# Agent Notes

`moecut` is a small C tool for profiling and pruning MoE experts inside GGUF
models. It is not a model runner, quantizer, or GGUF parser project. Keep the
production path focused on byte-level GGUF rewriting with ggml's public C API.

The main promise is direct pruning of already-quantized GGUFs. Expert tensors
are sliced only along whole expert slabs, so quantization blocks must never be
split or interpreted.

## Goals

* Keep the tool a single C11 command-line binary with no runtime dependencies
  beyond the vendored ggml build.
* Use `vendor/ggml` for GGUF metadata, tensor shapes, type traits, and writing.
  Do not grow a parallel GGUF parser in moecut.
* Preserve the direct-quantized path: Q4/Q8/etc. inputs should be copied and
  compacted as bytes, not dequantized and requantized.
* Keep pruning conservative. If a model layout is not clearly supported, refuse
  with a specific error instead of guessing.
* Keep every layer at the same expert count after pruning. Which experts are
  kept may differ per layer; the number kept may not.
* Refuse group-limited routing until proper per-group pruning is implemented.

## Quality Rules

* Keep the implementation small and readable. Remove dead code, unused helpers,
  speculative abstractions, and flags that do not serve the current release path.
* Comment important code where the tensor layout, byte slicing, alignment, or
  safety check is not obvious from the local code.
* Prefer comments next to the implementation over long design documents.
* Keep public helpers narrow. Command code can know about command arguments;
  shared code should only expose genuinely shared GGUF/profile behavior.
* Do not introduce C++ in `src/`. ggml may compile C++ internally, but moecut's
  own code stays C11.
* Do not copy multi-GB tensor data into memory. Prune should stream from input
  to output in bounded chunks.
* Check write and read errors on every file operation that can touch model data.
  Never assume a large output write succeeded.

## Tensor Rules

* Expert FFN tensors are expected as fused 3D tensors named
  `blk.N.ffn_{gate,up,down}_exps.weight` with expert on `ne[2]`.
* Router tensors are expected as `blk.N.ffn_gate_inp.weight` with the expert
  axis matching the model expert count.
* Shared experts such as `ffn_*_shexp` and dense tensors are not pruned.
* The metadata key `{arch}.expert_count` must be patched to the new count.
* `--keep` must remain at least `{arch}.expert_used_count`.
* GGUF alignment handling is important. If output metadata cannot preserve the
  input layout safely, refuse rather than producing a questionable file.

## Safety

* Do not default large generated files to `/tmp`; it may be tmpfs. Use the CWD
  or an explicit user path.
* Do not run multi-GB validation jobs unless the user explicitly asks for them.
* Do not mutate `vendor/ggml` casually. Treat ggml upgrades as explicit tasks,
  record the pinned commit, and rerun tests after changing it.
* Do not use destructive git commands unless the user explicitly requests them.

## Layout

* `src/main.c`: CLI dispatch.
* `src/info.c`: cheap GGUF/model inspection.
* `src/profile.c`: imatrix GGUF profiling and profile JSON output.
* `src/prune.c`: metadata rewrite plus streaming tensor byte slicing.
* `src/common.{c,h}`: shared errors, GGUF KV helpers, tensor classification,
  and profile loading.
* `tests/`: tiny synthetic GGUF generation and byte-level smoke tests.
* `vendor/ggml`: git submodule pinned to the ggml API used by this project.

This list is not a substitute for reading the code before changing it.

## Testing

Use CMake for normal validation:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j2
bash tests/run_tests.sh
```

The smoke test creates tiny GGUF fixtures, runs `info`, `profile`, and `prune`,
then verifies the pruned bytes. For changes to pruning, this test is mandatory.

Before calling a change done, also consider what was not tested: real imatrix
files, quantized fixtures, group routing, and end-to-end perplexity validation
are outside the current smoke test.
