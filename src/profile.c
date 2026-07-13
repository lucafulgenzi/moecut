#include "common.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct expert_acc {
    double energy;
    double freq;
    double score;
};

struct layer_acc {
    int layer;
    int n_expert;
    int n_sources;
    struct expert_acc * e;
};

struct rank_item {
    int expert;
    double score;
};

static bool strip_suffix(const char * s, const char * suffix, char * out, size_t out_sz) {
    const size_t n = strlen(s);
    const size_t m = strlen(suffix);
    if (n <= m || strcmp(s + n - m, suffix) != 0) {
        return false;
    }
    if (n - m + 1 > out_sz) {
        mc_die("tensor name too long while stripping suffix '%s'", suffix);
    }
    memcpy(out, s, n - m);
    out[n - m] = 0;
    return true;
}

static struct layer_acc * get_layer(struct layer_acc ** layers, int * n, int * cap, int layer, int n_expert) {
    for (int i = 0; i < *n; i++) {
        if ((*layers)[i].layer == layer) {
            if ((*layers)[i].n_expert != n_expert) {
                mc_die("imatrix layer %d has inconsistent expert counts", layer);
            }
            return &(*layers)[i];
        }
    }
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 16;
        *layers = realloc(*layers, (size_t) *cap * sizeof((*layers)[0]));
        if (!*layers) mc_die("out of memory");
    }
    struct layer_acc * l = &(*layers)[(*n)++];
    l->layer = layer;
    l->n_expert = n_expert;
    l->n_sources = 0;
    l->e = calloc((size_t) n_expert, sizeof(l->e[0]));
    if (!l->e) mc_die("out of memory");
    return l;
}

static int cmp_layer_acc(const void * a, const void * b) {
    const struct layer_acc * x = a;
    const struct layer_acc * y = b;
    return x->layer - y->layer;
}

static int cmp_rank_desc(const void * a, const void * b) {
    const struct rank_item * x = a;
    const struct rank_item * y = b;
    return (y->score > x->score) - (y->score < x->score);
}

static void read_f32(FILE * f, const char * path, uint64_t off, float * out, size_t n) {
    if (fseeko(f, (off_t) off, SEEK_SET) != 0) {
        mc_die("%s: seek failed: %s", path, strerror(errno));
    }
    if (n > 0 && fread(out, sizeof(float), n, f) != n) {
        mc_die("%s: truncated input while reading imatrix tensor data", path);
    }
}

static void add_imatrix_source(
        struct layer_acc ** layers, int * n_layers, int * cap_layers,
        const struct gguf_context * gguf, struct ggml_context * gctx,
        FILE * f, const char * path, int64_t counts_id, const struct ggml_tensor * counts_t) {
    char base[GGML_MAX_NAME];
    if (!strip_suffix(ggml_get_name(counts_t), ".counts", base, sizeof(base))) {
        return;
    }

    int layer = -1;
    if (mc_classify_tensor(base, &layer) != MC_TENSOR_EXPS) {
        return;
    }
    if (counts_t->type != GGML_TYPE_F32 || counts_t->ne[1] != 1 || counts_t->ne[2] != 1 || counts_t->ne[3] != 1) {
        mc_warn("skipping %s: expected F32 [n_expert] counts tensor", ggml_get_name(counts_t));
        return;
    }

    if (counts_t->ne[0] > INT_MAX) {
        mc_die("too many experts in %s: %" PRId64, ggml_get_name(counts_t), counts_t->ne[0]);
    }
    const int n_expert = (int) counts_t->ne[0];
    struct layer_acc * l = get_layer(layers, n_layers, cap_layers, layer, n_expert);
    float * counts = malloc((size_t) n_expert * sizeof(counts[0]));
    if (!counts) mc_die("out of memory");
    read_f32(f, path,
             (uint64_t) gguf_get_data_offset(gguf) + (uint64_t) gguf_get_tensor_offset(gguf, counts_id),
             counts, (size_t) n_expert);

    char sum_name[GGML_MAX_NAME + 16];
    const int sum_name_len = snprintf(sum_name, sizeof(sum_name), "%s.in_sum2", base);
    const struct ggml_tensor * sum_t = NULL;
    int64_t sum_id = -1;
    if (sum_name_len < 0 || sum_name_len >= (int) sizeof(sum_name)) {
        mc_warn("ignoring %s: derived in_sum2 tensor name is too long", base);
    } else {
        sum_t = ggml_get_tensor(gctx, sum_name);
        sum_id = gguf_find_tensor(gguf, sum_name);
    }
    int64_t row_size = 0;
    uint64_t sum_base = 0;
    if (sum_t && sum_id >= 0) {
        if (sum_t->type == GGML_TYPE_F32 && sum_t->ne[1] == n_expert) {
            row_size = sum_t->ne[0];
            sum_base = (uint64_t) gguf_get_data_offset(gguf) + (uint64_t) gguf_get_tensor_offset(gguf, sum_id);
        } else {
            mc_warn("ignoring %s: expected F32 [row_size, n_expert]", sum_name);
        }
    }

    double total_counts = 0.0;
    for (int e = 0; e < n_expert; e++) {
        if (counts[e] > 0.0f) total_counts += counts[e];
    }
    if (total_counts <= 0.0) {
        mc_warn("skipping %s: all counts are zero", ggml_get_name(counts_t));
        free(counts);
        return;
    }

    double mean_energy = 0.0;
    int active = 0;
    double * energy = calloc((size_t) n_expert, sizeof(energy[0]));
    if (!energy) mc_die("out of memory");
    float * row = row_size > 0 ? malloc((size_t) row_size * sizeof(row[0])) : NULL;
    if (row_size > 0 && !row) mc_die("out of memory");
    for (int e = 0; e < n_expert; e++) {
        if (counts[e] <= 0.0f) {
            continue;
        }
        if (row_size > 0) {
            // Sum2 can be large in real imatrix files. Read one expert row at
            // a time so profiling does not mirror the full GGUF blob in RAM.
            read_f32(f, path, sum_base + (uint64_t) e * (uint64_t) row_size * sizeof(float), row, (size_t) row_size);
            double s = 0.0;
            for (int64_t c = 0; c < row_size; c++) {
                s += row[c];
            }
            energy[e] = s / ((double) counts[e] * (double) row_size);
        } else {
            energy[e] = 1.0;
        }
        mean_energy += energy[e];
        active++;
    }
    mean_energy = active ? mean_energy / active : 1.0;
    if (mean_energy <= 0.0) mean_energy = 1.0;

    for (int e = 0; e < n_expert; e++) {
        const double freq = counts[e] > 0.0f ? counts[e] / total_counts : 0.0;
        const double norm_energy = energy[e] / mean_energy;

        // This mirrors the validated Python metric: routing traffic dominates,
        // while activation energy is a square-root tie breaker for hot experts.
        const double score = freq * sqrt(norm_energy > 0.0 ? norm_energy : 0.0);
        l->e[e].energy += norm_energy;
        l->e[e].freq += freq;
        l->e[e].score += score;
    }
    l->n_sources++;
    free(row);
    free(energy);
    free(counts);
}

int mc_cmd_profile(int argc, char ** argv) {
    if (argc != 2) {
        mc_die("usage: moecut profile <imatrix.gguf>");
    }

    struct ggml_context * gctx = NULL;
    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = &gctx,
    };
    struct gguf_context * gguf = gguf_init_from_file(argv[1], params);
    if (!gguf || !gctx) {
        mc_die("cannot read imatrix GGUF '%s'", argv[1]);
    }
    FILE * f = fopen(argv[1], "rb");
    if (!f) mc_die("cannot open imatrix GGUF '%s': %s", argv[1], strerror(errno));

    struct layer_acc * layers = NULL;
    int n_layers = 0;
    int cap_layers = 0;
    const int64_t n_tensors = gguf_get_n_tensors(gguf);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf, i);
        struct ggml_tensor * t = ggml_get_tensor(gctx, name);
        if (!t) mc_die("internal error: tensor '%s' missing from ggml context", name);
        add_imatrix_source(&layers, &n_layers, &cap_layers, gguf, gctx, f, argv[1], i, t);
    }
    if (n_layers == 0) {
        mc_die("%s: no MoE expert .counts tensors found", argv[1]);
    }
    qsort(layers, (size_t) n_layers, sizeof(layers[0]), cmp_layer_acc);

    printf("{\n");
    for (int i = 0; i < n_layers; i++) {
        struct layer_acc * l = &layers[i];
        struct rank_item * rank = malloc((size_t) l->n_expert * sizeof(rank[0]));
        if (!rank) mc_die("out of memory");

        double entropy = 0.0;
        for (int e = 0; e < l->n_expert; e++) {
            if (l->n_sources > 0) {
                l->e[e].freq /= l->n_sources;
                l->e[e].energy /= l->n_sources;
                l->e[e].score /= l->n_sources;
            }
            if (l->e[e].freq > 0.0) {
                entropy -= l->e[e].freq * log(l->e[e].freq);
            }
            rank[e].expert = e;
            rank[e].score = l->e[e].score;
        }
        entropy = l->n_expert > 1 ? entropy / log((double) l->n_expert) : 0.0;
        qsort(rank, (size_t) l->n_expert, sizeof(rank[0]), cmp_rank_desc);

        const int cold_n = l->n_expert / 4;
        double cold_traffic = 0.0;
        for (int j = l->n_expert - cold_n; j < l->n_expert; j++) {
            if (j >= 0) cold_traffic += l->e[rank[j].expert].freq;
        }

        printf("  \"%d\": {\n", l->layer);
        printf("    \"n_expert\": %d,\n", l->n_expert);
        printf("    \"sources\": %d,\n", l->n_sources);
        printf("    \"entropy\": %.9g,\n", entropy);
        printf("    \"cold_25_traffic\": %.9g,\n", cold_traffic);
        printf("    \"rank\": [");
        for (int e = 0; e < l->n_expert; e++) printf("%s%d", e ? ", " : "", rank[e].expert);
        printf("],\n");
        printf("    \"score\": [");
        for (int e = 0; e < l->n_expert; e++) printf("%s%.9g", e ? ", " : "", l->e[e].score);
        printf("],\n");
        printf("    \"freq\": [");
        for (int e = 0; e < l->n_expert; e++) printf("%s%.9g", e ? ", " : "", l->e[e].freq);
        printf("],\n");
        printf("    \"energy\": [");
        for (int e = 0; e < l->n_expert; e++) printf("%s%.9g", e ? ", " : "", l->e[e].energy);
        printf("]\n");
        printf("  }%s\n", i + 1 == n_layers ? "" : ",");
        free(rank);
    }
    printf("}\n");

    for (int i = 0; i < n_layers; i++) free(layers[i].e);
    free(layers);
    if (fclose(f) != 0) mc_die("%s: close failed: %s", argv[1], strerror(errno));
    gguf_free(gguf);
    ggml_free(gctx);
    return 0;
}
