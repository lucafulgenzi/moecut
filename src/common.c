#include "common.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

// ---- errors ----------------------------------------------------------------

_Noreturn void mc_die(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "moecut: error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

void mc_warn(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "moecut: warning: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

// ---- GGUF KV helpers --------------------------------------------------------

bool mc_kv_int(const struct gguf_context * ctx, const char * key, int64_t * out) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return false;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  *out = gguf_get_val_u8 (ctx, id); return true;
        case GGUF_TYPE_INT8:   *out = gguf_get_val_i8 (ctx, id); return true;
        case GGUF_TYPE_UINT16: *out = gguf_get_val_u16(ctx, id); return true;
        case GGUF_TYPE_INT16:  *out = gguf_get_val_i16(ctx, id); return true;
        case GGUF_TYPE_UINT32: *out = gguf_get_val_u32(ctx, id); return true;
        case GGUF_TYPE_INT32:  *out = gguf_get_val_i32(ctx, id); return true;
        case GGUF_TYPE_UINT64: *out = (int64_t) gguf_get_val_u64(ctx, id); return true;
        case GGUF_TYPE_INT64:  *out = gguf_get_val_i64(ctx, id); return true;
        default:
            mc_die("KV '%s' has non-integer type %s", key, gguf_type_name(gguf_get_kv_type(ctx, id)));
    }
}

const char * mc_arch(const struct gguf_context * ctx) {
    const int64_t id = gguf_find_key(ctx, "general.architecture");
    if (id < 0) {
        mc_die("cannot read general.architecture: not a model GGUF?");
    }
    return gguf_get_val_str(ctx, id);
}

// ---- MoE tensor name matching ----------------------------------------------

enum mc_tensor_class mc_classify_tensor(const char * name, int * layer) {
    // all per-expert tensors live under "blk.<N>."
    if (strncmp(name, "blk.", 4) != 0) {
        return MC_TENSOR_OTHER;
    }
    const char * p = name + 4;
    if (!isdigit((unsigned char) *p)) {
        return MC_TENSOR_OTHER;
    }
    char * end = NULL;
    const long l = strtol(p, &end, 10);
    if (end == p || *end != '.') {
        return MC_TENSOR_OTHER;
    }
    const char * rest = end + 1;

    // exact suffix match: "_shexp" variants and everything else must NOT match
    static const char * const exps[] = {
        "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
        "ffn_gate_exps.bias",   "ffn_up_exps.bias",   "ffn_down_exps.bias",
    };
    static const char * const router[] = {
        "ffn_gate_inp.weight", "ffn_gate_inp.bias",
        "ffn_exp_probs_b.bias", "exp_probs_b.bias",
    };
    for (size_t i = 0; i < sizeof(exps)/sizeof(exps[0]); i++) {
        if (strcmp(rest, exps[i]) == 0) {
            if (layer) *layer = (int) l;
            return MC_TENSOR_EXPS;
        }
    }
    for (size_t i = 0; i < sizeof(router)/sizeof(router[0]); i++) {
        if (strcmp(rest, router[i]) == 0) {
            if (layer) *layer = (int) l;
            return MC_TENSOR_ROUTER;
        }
    }
    return MC_TENSOR_OTHER;
}

// ---- profile JSON parsing ----------------------------------------------------
// Minimal recursive-descent parser for the profile schema written by
// `moecut profile` / expert_profile.py: {"<layer>": {"n_expert": int,
// "rank": [ints], ...}, ...}. Unknown keys are skipped.

struct js {
    const char * start;
    const char * p;
    const char * end;
    const char * path; // for error messages
};

static void js_err(struct js * j, const char * what) {
    mc_die("%s: invalid profile JSON: %s at byte %zd", j->path, what, (ssize_t)(j->p - j->start));
}

static void js_ws(struct js * j) {
    while (j->p < j->end && isspace((unsigned char) *j->p)) j->p++;
}

static bool js_lit(struct js * j, char c) {
    js_ws(j);
    if (j->p < j->end && *j->p == c) { j->p++; return true; }
    return false;
}

static void js_expect(struct js * j, char c) {
    if (!js_lit(j, c)) {
        char msg[64];
        snprintf(msg, sizeof(msg), "expected '%c'", c);
        js_err(j, msg);
    }
}

// parse a JSON string, return malloc'd copy (escapes kept verbatim except \")
static char * js_string(struct js * j) {
    js_expect(j, '"');
    const char * start = j->p;
    while (j->p < j->end && *j->p != '"') {
        if (*j->p == '\\') j->p++;
        j->p++;
    }
    if (j->p >= j->end) js_err(j, "unterminated string");
    char * s = strndup(start, (size_t)(j->p - start));
    j->p++; // closing quote
    return s;
}

static double js_number(struct js * j) {
    js_ws(j);
    char * end = NULL;
    const double v = strtod(j->p, &end);
    if (end == j->p) js_err(j, "expected number");
    j->p = end;
    return v;
}

static void js_skip_value(struct js * j) {
    js_ws(j);
    if (j->p >= j->end) js_err(j, "unexpected end");
    switch (*j->p) {
        case '"': free(js_string(j)); return;
        case '{':
            j->p++;
            js_ws(j);
            if (js_lit(j, '}')) return;
            do { free(js_string(j)); js_expect(j, ':'); js_skip_value(j); } while (js_lit(j, ','));
            js_expect(j, '}');
            return;
        case '[':
            j->p++;
            js_ws(j);
            if (js_lit(j, ']')) return;
            do { js_skip_value(j); } while (js_lit(j, ','));
            js_expect(j, ']');
            return;
        case 't': case 'f': case 'n':
            while (j->p < j->end && isalpha((unsigned char) *j->p)) j->p++;
            return;
        default:
            js_number(j);
            return;
    }
}

// parse array of numbers; returns malloc'd doubles, count in *n
static double * js_num_array(struct js * j, int * n) {
    js_expect(j, '[');
    int cap = 64, cnt = 0;
    double * v = malloc(cap * sizeof(double));
    js_ws(j);
    if (!js_lit(j, ']')) {
        do {
            if (cnt == cap) { cap *= 2; v = realloc(v, cap * sizeof(double)); }
            v[cnt++] = js_number(j);
        } while (js_lit(j, ','));
        js_expect(j, ']');
    }
    *n = cnt;
    return v;
}

static int cmp_layer(const void * a, const void * b) {
    const struct mc_layer_profile * la = a, * lb = b;
    return la->layer - lb->layer;
}

struct mc_profile * mc_profile_load(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        mc_die("cannot open profile '%s'", path);
    }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * buf = malloc((size_t) sz + 1);
    if (fread(buf, 1, (size_t) sz, f) != (size_t) sz) {
        mc_die("cannot read profile '%s'", path);
    }
    buf[sz] = 0;
    fclose(f);

    struct js j = { buf, buf, buf + sz, path };
    struct mc_profile * prof = calloc(1, sizeof(*prof));
    int cap = 16;
    prof->layers = malloc(cap * sizeof(prof->layers[0]));

    js_expect(&j, '{');
    js_ws(&j);
    if (!js_lit(&j, '}')) {
        do {
            char * key = js_string(&j);
            js_expect(&j, ':');
            char * kend = NULL;
            const long layer = strtol(key, &kend, 10);
            if (kend == key || *kend != 0) {
                // non-layer top-level key (future metadata): skip
                free(key);
                js_skip_value(&j);
                continue;
            }
            free(key);

            struct mc_layer_profile lp = { (int) layer, 0, NULL };
            int rank_n = 0;
            js_expect(&j, '{');
            js_ws(&j);
            if (!js_lit(&j, '}')) {
                do {
                    char * k = js_string(&j);
                    js_expect(&j, ':');
                    if (strcmp(k, "rank") == 0) {
                        int n = 0;
                        double * v = js_num_array(&j, &n);
                        lp.rank = malloc(n * sizeof(int));
                        for (int i = 0; i < n; i++) lp.rank[i] = (int) v[i];
                        rank_n = n;
                        if (lp.n_expert == 0) lp.n_expert = n;
                        free(v);
                    } else if (strcmp(k, "n_expert") == 0) {
                        lp.n_expert = (int) js_number(&j);
                    } else {
                        js_skip_value(&j);
                    }
                    free(k);
                } while (js_lit(&j, ','));
                js_expect(&j, '}');
            }
            if (!lp.rank || lp.n_expert <= 0) {
                mc_die("%s: layer %d has no 'rank' array", path, lp.layer);
            }
            if (rank_n != lp.n_expert) {
                mc_die("%s: layer %d rank length %d does not match n_expert %d", path, lp.layer, rank_n, lp.n_expert);
            }
            // rank must be a permutation of 0..n_expert-1
            char * seen = calloc((size_t) lp.n_expert, 1);
            for (int i = 0; i < lp.n_expert; i++) {
                const int e = lp.rank[i];
                if (e < 0 || e >= lp.n_expert || seen[e]) {
                    mc_die("%s: layer %d 'rank' is not a permutation of 0..%d", path, lp.layer, lp.n_expert - 1);
                }
                seen[e] = 1;
            }
            free(seen);

            if (prof->n_layers == cap) { cap *= 2; prof->layers = realloc(prof->layers, cap * sizeof(prof->layers[0])); }
            prof->layers[prof->n_layers++] = lp;
        } while (js_lit(&j, ','));
        js_expect(&j, '}');
    }
    free(buf);

    if (prof->n_layers == 0) {
        mc_die("%s: no layers in profile", path);
    }
    qsort(prof->layers, (size_t) prof->n_layers, sizeof(prof->layers[0]), cmp_layer);
    return prof;
}

void mc_profile_free(struct mc_profile * p) {
    if (!p) return;
    for (int i = 0; i < p->n_layers; i++) {
        free(p->layers[i].rank);
    }
    free(p->layers);
    free(p);
}

const struct mc_layer_profile * mc_profile_layer(const struct mc_profile * p, int layer) {
    for (int i = 0; i < p->n_layers; i++) {
        if (p->layers[i].layer == layer) return &p->layers[i];
    }
    return NULL;
}
