#include "common.h"

#include <string.h>

int mc_cmd_info(int argc, char ** argv);
int mc_cmd_profile(int argc, char ** argv);
int mc_cmd_prune(int argc, char ** argv);

static void usage(void) {
    fprintf(stderr,
        "usage:\n"
        "  moecut info <model.gguf>\n"
        "  moecut profile <imatrix.gguf>\n"
        "  moecut prune <in.gguf> <out.gguf> --profile <profile.json> --keep N\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }

    if (strcmp(argv[1], "info") == 0) {
        return mc_cmd_info(argc - 1, argv + 1);
    }
    if (strcmp(argv[1], "profile") == 0) {
        return mc_cmd_profile(argc - 1, argv + 1);
    }
    if (strcmp(argv[1], "prune") == 0) {
        return mc_cmd_prune(argc - 1, argv + 1);
    }

    usage();
    return 2;
}
