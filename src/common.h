// common.h — shared helpers for moecut: errors, GGUF KV access, MoE tensor
// name matching, profile (keep-list) loading.
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "ggml.h"
#include "gguf.h"

// ---- errors ----------------------------------------------------------------

// print "moecut: error: ..." to stderr and exit(1)
_Noreturn void mc_die(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void mc_warn(const char * fmt, ...) __attribute__((format(printf, 1, 2)));

// ---- GGUF KV helpers --------------------------------------------------------

// read an integer-typed KV regardless of exact width; false if key missing
bool mc_kv_int(const struct gguf_context * ctx, const char * key, int64_t * out);

// "general.architecture" or die
const char * mc_arch(const struct gguf_context * ctx);

// ---- MoE tensor name matching ----------------------------------------------

enum mc_tensor_class {
    MC_TENSOR_OTHER = 0,   // dense / shared-expert / anything we never touch
    MC_TENSOR_EXPS,        // blk.N.ffn_{gate,up,down}_exps.{weight,bias}: expert = outermost dim
    MC_TENSOR_ROUTER,      // blk.N.ffn_gate_inp.{weight,bias}, blk.N.[ffn_]exp_probs_b.bias
};

// classify a tensor name; on EXPS/ROUTER match, *layer is set
enum mc_tensor_class mc_classify_tensor(const char * name, int * layer);

// ---- expert profile (output of `moecut profile`, input of `moecut prune`) ---

struct mc_layer_profile {
    int      layer;       // blk index
    int      n_expert;
    int    * rank;        // [n_expert] expert ids, hottest first
};

struct mc_profile {
    int                       n_layers;
    struct mc_layer_profile * layers;   // sorted by .layer ascending
};

// parse a profile JSON (schema of expert_profile.py / `moecut profile`); dies on error
struct mc_profile * mc_profile_load(const char * path);
void mc_profile_free(struct mc_profile * p);

// find layer entry or NULL
const struct mc_layer_profile * mc_profile_layer(const struct mc_profile * p, int layer);
