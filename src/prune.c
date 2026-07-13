#include "common.h"

#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define COPY_BUF_SIZE (1024 * 1024)

struct prune_args {
    const char * in_path;
    const char * out_path;
    const char * profile_path;
    int keep;
};

struct tensor_plan {
    enum mc_tensor_class cls;
    int keep;
    int * kept;
    size_t chunk_size;
    size_t in_size;
};

static void parse_args(int argc, char ** argv, struct prune_args * out) {
    if (argc < 6) {
        mc_die("usage: moecut prune <in.gguf> <out.gguf> --profile <profile.json> --keep N");
    }
    out->in_path = argv[1];
    out->out_path = argv[2];
    out->profile_path = NULL;
    out->keep = -1;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            out->profile_path = argv[++i];
        } else if (strcmp(argv[i], "--keep") == 0 && i + 1 < argc) {
            char * end = NULL;
            const long keep = strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != 0 || keep <= 0 || keep > INT_MAX) {
                mc_die("--keep requires a positive integer");
            }
            out->keep = (int) keep;
        } else {
            mc_die("unknown prune argument '%s'", argv[i]);
        }
    }
    if (!out->profile_path) mc_die("prune requires --profile <profile.json>");
    if (out->keep <= 0) mc_die("prune requires --keep N with N > 0");
    if (strcmp(out->in_path, out->out_path) == 0) {
        mc_die("input and output paths must be different");
    }
}

static void set_same_int_type(struct gguf_context * dst, const struct gguf_context * src, const char * key, int64_t val) {
    const int64_t id = gguf_find_key(src, key);
    if (id < 0) {
        gguf_set_val_i64(dst, key, val);
        return;
    }
    switch (gguf_get_kv_type(src, id)) {
        case GGUF_TYPE_UINT8:  gguf_set_val_u8 (dst, key, (uint8_t)  val); break;
        case GGUF_TYPE_INT8:   gguf_set_val_i8 (dst, key, (int8_t)   val); break;
        case GGUF_TYPE_UINT16: gguf_set_val_u16(dst, key, (uint16_t) val); break;
        case GGUF_TYPE_INT16:  gguf_set_val_i16(dst, key, (int16_t)  val); break;
        case GGUF_TYPE_UINT32: gguf_set_val_u32(dst, key, (uint32_t) val); break;
        case GGUF_TYPE_INT32:  gguf_set_val_i32(dst, key, (int32_t)  val); break;
        case GGUF_TYPE_UINT64: gguf_set_val_u64(dst, key, (uint64_t) val); break;
        case GGUF_TYPE_INT64:  gguf_set_val_i64(dst, key, val); break;
        default:
            mc_die("KV '%s' has non-integer type %s", key, gguf_type_name(gguf_get_kv_type(src, id)));
    }
}

static int cmp_int(const void * a, const void * b) {
    const int x = *(const int *) a;
    const int y = *(const int *) b;
    return (x > y) - (x < y);
}

static int * kept_from_profile(const struct mc_layer_profile * lp, int keep) {
    int * kept = malloc((size_t) keep * sizeof(kept[0]));
    if (!kept) mc_die("out of memory");
    for (int i = 0; i < keep; i++) kept[i] = lp->rank[i];

    // The profile ranks experts hottest-first; tensor data must be copied in
    // original expert-id order so the pruned model has compact, stable rows.
    qsort(kept, (size_t) keep, sizeof(kept[0]), cmp_int);
    return kept;
}

static int expert_dim_for_tensor(enum mc_tensor_class cls, const struct ggml_tensor * t, int expected) {
    if (cls == MC_TENSOR_EXPS) {
        if (t->ne[2] == expected) return 2;
        return -1;
    }
    if (cls == MC_TENSOR_ROUTER) {
        if (t->ne[1] == expected) return 1;
        if (t->ne[0] == expected && ggml_blck_size(t->type) == 1) return 0;
        return -1;
    }
    return -1;
}

static size_t tensor_chunk_size(const struct ggml_tensor * t, int expert_dim) {
    if (expert_dim == 2) return t->nb[2];
    if (expert_dim == 1) return t->nb[1];
    if (expert_dim == 0 && ggml_blck_size(t->type) == 1) return ggml_type_size(t->type);
    return 0;
}

static void checked_seek(FILE * f, uint64_t off, const char * path) {
    if (fseeko(f, (off_t) off, SEEK_SET) != 0) {
        mc_die("%s: seek failed: %s", path, strerror(errno));
    }
}

static void checked_write(FILE * f, const void * p, size_t n, const char * path) {
    if (n > 0 && fwrite(p, 1, n, f) != n) {
        mc_die("%s: write failed: %s", path, strerror(errno));
    }
}

static void copy_exact(FILE * in, FILE * out, uint64_t off, size_t n, void * buf, const char * in_path, const char * out_path) {
    checked_seek(in, off, in_path);
    size_t left = n;
    while (left > 0) {
        const size_t want = left < COPY_BUF_SIZE ? left : COPY_BUF_SIZE;
        const size_t got = fread(buf, 1, want, in);
        if (got != want) {
            mc_die("%s: truncated input while reading tensor data", in_path);
        }
        checked_write(out, buf, got, out_path);
        left -= got;
    }
}

static void write_padding(FILE * out, size_t bytes_written, size_t alignment, const char * out_path) {
    static const unsigned char zero[GGUF_DEFAULT_ALIGNMENT] = { 0 };
    const size_t pad = (alignment - (bytes_written % alignment)) % alignment;
    if (pad > sizeof(zero)) {
        mc_die("internal error: padding buffer too small");
    }
    checked_write(out, zero, pad, out_path);
}

int mc_cmd_prune(int argc, char ** argv) {
    struct prune_args args;
    parse_args(argc, argv, &args);

    struct mc_profile * profile = mc_profile_load(args.profile_path);
    struct ggml_context * in_gctx = NULL;
    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = &in_gctx,
    };
    struct gguf_context * in_gguf = gguf_init_from_file(args.in_path, params);
    if (!in_gguf || !in_gctx) {
        mc_die("cannot read GGUF '%s'", args.in_path);
    }

    const size_t alignment = gguf_get_alignment(in_gguf);
    if (alignment != GGUF_DEFAULT_ALIGNMENT) {
        mc_die("unsupported GGUF alignment %zu in '%s' (currently only %d is supported)",
               alignment, args.in_path, GGUF_DEFAULT_ALIGNMENT);
    }

    const char * arch = mc_arch(in_gguf);
    char key[256];
    int64_t expert_count = -1;
    int64_t expert_used_count = 1;
    int64_t expert_group_count = 1;
    snprintf(key, sizeof(key), "%s.expert_count", arch);
    if (!mc_kv_int(in_gguf, key, &expert_count) || expert_count <= 0) {
        mc_die("cannot read positive %s expert_count", arch);
    }
    if (expert_count > INT_MAX) {
        mc_die("%s.expert_count is too large: %" PRId64, arch, expert_count);
    }
    snprintf(key, sizeof(key), "%s.expert_used_count", arch);
    mc_kv_int(in_gguf, key, &expert_used_count);
    snprintf(key, sizeof(key), "%s.expert_group_count", arch);
    mc_kv_int(in_gguf, key, &expert_group_count);
    if (expert_group_count > 1) {
        mc_die("group-limited routing is not implemented yet (%s.expert_group_count=%" PRId64 ")", arch, expert_group_count);
    }
    if (args.keep < expert_used_count) {
        mc_die("--keep %d is invalid: router uses top-%" PRId64 " experts", args.keep, expert_used_count);
    }
    if (args.keep >= expert_count) {
        mc_die("--keep %d must be smaller than current expert_count %" PRId64, args.keep, expert_count);
    }

    const int64_t n_tensors = gguf_get_n_tensors(in_gguf);
    struct tensor_plan * plans = calloc((size_t) n_tensors, sizeof(plans[0]));
    if (!plans) mc_die("out of memory");

    struct gguf_context * out_gguf = gguf_init_empty();
    gguf_set_kv(out_gguf, in_gguf);
    snprintf(key, sizeof(key), "%s.expert_count", arch);
    set_same_int_type(out_gguf, in_gguf, key, args.keep);

    struct ggml_context * out_gctx = ggml_init((struct ggml_init_params) {
        .mem_size = (size_t) n_tensors * ggml_tensor_overhead() + GGML_MEM_ALIGN,
        .mem_buffer = NULL,
        .no_alloc = true,
    });
    if (!out_gctx) mc_die("cannot allocate tensor metadata context");

    int pruned_tensors = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(in_gguf, i);
        const struct ggml_tensor * in_t = ggml_get_tensor(in_gctx, name);
        if (!in_t) mc_die("internal error: tensor '%s' missing from ggml context", name);

        struct tensor_plan * p = &plans[i];
        p->in_size = ggml_nbytes(in_t);
        int layer = -1;
        p->cls = mc_classify_tensor(name, &layer);

        int64_t out_ne[GGML_MAX_DIMS];
        for (int d = 0; d < GGML_MAX_DIMS; d++) out_ne[d] = in_t->ne[d];

        if (p->cls == MC_TENSOR_EXPS || p->cls == MC_TENSOR_ROUTER) {
            const struct mc_layer_profile * lp = mc_profile_layer(profile, layer);
            if (!lp) mc_die("profile has no layer %d required by tensor '%s'", layer, name);
            if (lp->n_expert != expert_count) {
                mc_die("profile layer %d has n_expert=%d but model has %" PRId64, layer, lp->n_expert, expert_count);
            }
            const int expert_dim = expert_dim_for_tensor(p->cls, in_t, (int) expert_count);
            if (expert_dim < 0) {
                mc_die("cannot identify expert axis for tensor '%s' shape [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]",
                       name, in_t->ne[0], in_t->ne[1], in_t->ne[2], in_t->ne[3]);
            }
            p->keep = args.keep;
            p->kept = kept_from_profile(lp, args.keep);
            p->chunk_size = tensor_chunk_size(in_t, expert_dim);
            if (p->chunk_size == 0 || p->in_size != p->chunk_size * (size_t) expert_count) {
                // This check is intentionally strict: pruning relies on each
                // expert being one contiguous byte slab, including quant blocks.
                mc_die("tensor '%s' does not consist of %" PRId64 " equal contiguous expert slabs", name, expert_count);
            }
            out_ne[expert_dim] = args.keep;
            pruned_tensors++;
        }

        struct ggml_tensor * out_t = ggml_new_tensor(out_gctx, in_t->type, GGML_MAX_DIMS, out_ne);
        ggml_set_name(out_t, name);
        gguf_add_tensor(out_gguf, out_t);
    }

    if (!gguf_write_to_file(out_gguf, args.out_path, true)) {
        mc_die("cannot write GGUF metadata to '%s'", args.out_path);
    }

    FILE * in = fopen(args.in_path, "rb");
    if (!in) mc_die("cannot open '%s': %s", args.in_path, strerror(errno));
    FILE * out = fopen(args.out_path, "ab");
    if (!out) mc_die("cannot append '%s': %s", args.out_path, strerror(errno));
    void * buf = malloc(COPY_BUF_SIZE);
    if (!buf) mc_die("out of memory");

    for (int64_t i = 0; i < n_tensors; i++) {
        struct tensor_plan * p = &plans[i];
        const uint64_t src_base = (uint64_t) gguf_get_data_offset(in_gguf) + (uint64_t) gguf_get_tensor_offset(in_gguf, i);
        if (p->cls == MC_TENSOR_EXPS || p->cls == MC_TENSOR_ROUTER) {
            for (int k = 0; k < p->keep; k++) {
                const uint64_t off = src_base + (uint64_t) p->kept[k] * (uint64_t) p->chunk_size;
                copy_exact(in, out, off, p->chunk_size, buf, args.in_path, args.out_path);
            }
            write_padding(out, p->chunk_size * (size_t) p->keep, alignment, args.out_path);
        } else {
            copy_exact(in, out, src_base, p->in_size, buf, args.in_path, args.out_path);
            write_padding(out, p->in_size, alignment, args.out_path);
        }
    }

    if (fclose(out) != 0) mc_die("%s: close failed: %s", args.out_path, strerror(errno));
    if (fclose(in) != 0) mc_die("%s: close failed: %s", args.in_path, strerror(errno));
    free(buf);

    printf("pruned %" PRId64 " tensors (%d expert/router tensors), expert_count %" PRId64 " -> %d\n",
           n_tensors, pruned_tensors, expert_count, args.keep);

    for (int64_t i = 0; i < n_tensors; i++) free(plans[i].kept);
    free(plans);
    ggml_free(out_gctx);
    gguf_free(out_gguf);
    gguf_free(in_gguf);
    ggml_free(in_gctx);
    mc_profile_free(profile);
    return 0;
}
