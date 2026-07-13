#include "common.h"

#include <inttypes.h>
int mc_cmd_info(int argc, char ** argv) {
    if (argc != 2) {
        mc_die("usage: moecut info <model.gguf>");
    }

    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = NULL,
    };
    struct gguf_context * ctx = gguf_init_from_file(argv[1], params);
    if (!ctx) {
        mc_die("cannot read GGUF '%s'", argv[1]);
    }

    const char * arch = mc_arch(ctx);
    char key[256];
    int64_t expert_count = -1;
    int64_t expert_used_count = -1;
    int64_t expert_group_count = -1;

    snprintf(key, sizeof(key), "%s.expert_count", arch);
    mc_kv_int(ctx, key, &expert_count);
    snprintf(key, sizeof(key), "%s.expert_used_count", arch);
    mc_kv_int(ctx, key, &expert_used_count);
    snprintf(key, sizeof(key), "%s.expert_group_count", arch);
    mc_kv_int(ctx, key, &expert_group_count);

    int n_exps = 0;
    int n_router = 0;
    int max_layer = -1;
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n_tensors; i++) {
        int layer = -1;
        enum mc_tensor_class cls = mc_classify_tensor(gguf_get_tensor_name(ctx, i), &layer);
        if (cls == MC_TENSOR_EXPS) n_exps++;
        if (cls == MC_TENSOR_ROUTER) n_router++;
        if (layer > max_layer) max_layer = layer;
    }

    printf("file: %s\n", argv[1]);
    printf("architecture: %s\n", arch);
    printf("gguf_version: %" PRIu32 "\n", gguf_get_version(ctx));
    printf("alignment: %zu\n", gguf_get_alignment(ctx));
    printf("kv_count: %" PRId64 "\n", gguf_get_n_kv(ctx));
    printf("tensor_count: %" PRId64 "\n", n_tensors);
    printf("layer_count_detected: %d\n", max_layer + 1);
    if (expert_count >= 0) printf("expert_count: %" PRId64 "\n", expert_count);
    if (expert_used_count >= 0) printf("expert_used_count: %" PRId64 "\n", expert_used_count);
    if (expert_group_count >= 0) printf("expert_group_count: %" PRId64 "\n", expert_group_count);
    printf("expert_tensors_detected: %d\n", n_exps);
    printf("router_tensors_detected: %d\n", n_router);

    gguf_free(ctx);
    return 0;
}
