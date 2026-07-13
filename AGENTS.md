# CLAUDE.md — moecut (working name)

Single-binary C tool for domain-aware expert profiling and pruning of MoE models in
GGUF format. Successor to the validated Python experiment
[moe-scalpel](https://github.com/lucafulgenzi/moe-scalpel) — that repo stays frozen as
the experiment record (see its `docs/LOGBOOK.md`); this project is the product,
rebuilt from scratch in C.

## Why this tool exists (30 seconds of context)

Expert routing in MoE models is domain-dependent: on OLMoE-1B-7B the cold-tail overlap
between a coding profile and a prose profile is only **24.6%** — a domain-specific
profile finds a genuinely different cut than a generic one. Pruning the 25% coldest
experts for a domain cost only +0.5pp perplexity on-domain (KLD 0.055 vs 0.016 for
quantization alone, 94.4% same top token) while shrinking Q4 from 4.3 → 3.2 GB. The
damage curve is geometric (~2× mean KLD per 8 experts removed) and tail-concentrated.
Off-domain the pruned model is much worse by design (PPL +79%): pruned models are
*domain models*. All numbers: moe-scalpel logbook.

## The killer feature that justifies the rewrite

**Prune already-quantized GGUFs directly** (Q4_K_M, Q8_0, ...), not just F16.
The Python experiment restricted to F16 out of caution, but the restriction is
unnecessary: quantization blocks live *within* tensor rows, while expert slicing cuts
along the outermost dimension — whole-expert removal never splits a quant block.

Why it matters: users of large MoEs (Qwen3-30B-A3B, GPT-OSS, DeepSeek) never download
F16 (60+ GB); they download quants. Direct quantized pruning means: download Q4 from
HF → profile on your domain → cut → smaller Q4. No F16, no requantization, no
post-prune imatrix. It also makes big-model experiments feasible on 32 GB RAM machines.

## Language & architecture decisions (already made)

- **C** (C11). Rationale: zero runtime dependencies (single static-ish binary), byte-level
  file surgery is C's home turf, mmap for 60 GB files, and upstream-compatibility —
  if this ever becomes a PR to llama.cpp `tools/`, it must be C/C++.
- **Do NOT write a GGUF parser from scratch.** Use ggml's C API (`gguf.h`:
  `gguf_init_from_file`, `gguf_get_*`, writer functions), consumed as a **git
  submodule** (see Repository layout below). Study `tools/gguf-split` in llama.cpp —
  it is the existing example of byte-level GGUF manipulation and the pattern to follow
  (including its handling of tensor data alignment, default 32 bytes).
- CLI shape: `moecut profile <imatrix.gguf> [--compare <imatrix2.gguf>]`,
  `moecut prune <in.gguf> <out.gguf> --profile <p> --keep N`, `moecut info <model.gguf>`.
  Profile output: JSON to stdout or file.
- The Python scripts in moe-scalpel are the **reference implementation** for
  differential testing: same inputs must produce the same keep-lists and (for F16)
  bit-identical sliced tensors.

## Technical knowledge dump (hard-won, verified on llama.cpp source)

### GGUF/ggml tensor layout for MoE
- Expert FFN tensors are fused 3D: `blk.N.ffn_{gate,up,down}_exps.weight`,
  ggml dims `[ne0=n_embd (or n_ff for down), ne1=n_ff (or n_embd), ne2=n_expert]`.
  **The expert is the outermost dimension**: expert `e` occupies one contiguous slab of
  `ne1 * row_bytes` where `row_bytes = ne0 / block_size(type) * type_size(type)`.
  Pruning = copying the kept slabs. Works identically for quantized types.
- Router: `blk.N.ffn_gate_inp.weight`, dims `[n_embd, n_expert]` — one row per expert.
  Usually F32. Slice rows with the same keep-list.
- Optional per-expert bias: `blk.N.ffn_exp_probs_b.bias` (`[n_expert]`, DeepSeek-style).
  Also `ffn_gate_inp.bias` may exist. Slice per element.
- Shared experts (`ffn_*_shexp`) and all dense tensors: never touched.
- Metadata to patch: `{arch}.expert_count` (global hparam — hence **every layer must
  keep the same NUMBER of experts**; WHICH experts may differ per layer).
  Constraint: `keep >= {arch}.expert_used_count` (router does top-k).

### imatrix file format (GGUF-based, produced by llama-imatrix)
- Per weight tensor touched by `mul_mat_id`, two data tensors:
  `<name>.in_sum2` — F32 `[row_size, n_expert]` (per-expert, per-column squared
  activation sums) and `<name>.counts` — F32 `[n_expert]` (activation counts).
- KV: `imatrix.chunk_count`, `imatrix.chunk_size`, `imatrix.datasets`.
- Profile metrics that worked: per-expert freq (counts/total), energy
  (sum2/counts/cols, normalized per layer), score = freq^1.0 * energy^0.5,
  per-layer routing entropy, JS divergence between domains, cold-tail overlap.
  Go/no-go thresholds used: entropy < 0.85 green / > 0.95 red; cold-25% traffic < 3%
  green / > 8% red; overlap < 50% green / > 80% hypothesis-falsified.
- llama-imatrix runs fine on quantized models → the profile→prune loop works without
  any F16 artifact.

### Group-limited routing (P2, not P0)
DeepSeek-style models set `{arch}.expert_group_count` (`n_expert_groups`); the router
selects groups first, then experts within groups. Naive pruning breaks group structure.
Support = uniform cut per group (same keep count within each group, keep-lists chosen
per group). Until implemented: detect and refuse, with a clear message.

### Validation pipeline (external, document in README)
`llama-perplexity --kl-divergence-base ref.bin` (on the unpruned model) then
`--kl-divergence` (on the pruned). Key metrics and how to read them: median vs mean
KLD (typical-token vs tail damage), same-top-p. See moe-scalpel `docs/METRICS.md`.
Always evaluate on held-out on-domain data AND off-domain data.

### Pitfalls learned the hard way
- Reference logits and imatrix files are multi-GB. **Never default outputs to /tmp**
  (tmpfs): a full tmpfs silently truncates and you lose runs. Default to CWD or a
  user-given path; check write errors on every chunk.
- After pruning an F16 that will then be quantized, the pre-prune imatrix is invalid
  (expert indices shifted) — a fresh one is required. Not applicable to the direct
  quantized-prune path (no requantization happens).
- GGUFReader-style APIs expose numpy-reversed dims; in C with gguf.h you get ggml
  dims directly — one source of off-by-one-axis bugs eliminated, but be careful when
  cross-checking against the Python reference.
- OLMoE F16 ≈ 14 GB, 16 layers: partial offload with `-ngl 6` fits 8 GB VRAM.
  Sanity numbers for regression: keep=48 prune of the F16 is 14 → 11 GB (−23%),
  Q4_K_M 4.3 → 3.2 GB.

## Testing strategy

1. **Synthetic fixtures** (port `tests/make_synthetic.py` idea): tiny GGUF where each
   expert's values encode its id → slicing verified bit by bit. Add a quantized
   synthetic fixture (quantize the tiny model with llama-quantize in CI or commit it).
2. **Differential vs Python reference**: same imatrix in → same profile JSON out;
   same F16 + profile in → identical output tensors.
3. **End-to-end on OLMoE** (manual, documented): reproduce the known keep=48 numbers.
4. `moecut info` output on both original and pruned model as a cheap invariant check
   (tensor count, shapes, expert_count).

## Priorities

- **P0**: `prune` on F16 *and* quantized inputs, per-layer keep-lists, metadata patch,
  refuse-with-message on group routing. `profile` with the metrics above. `info`.
- **P1**: profile/prune round-trip UX (prune reads profile JSON), `--dry-run` with
  size forecast, clean errors (file too small = truncated input, etc.).
- **P2**: group-limited routing support; per-group keep-lists.
- **P3**: upstream-quality polish; consider proposing per-expert stats display as a
  `llama-imatrix --show-statistics` extension upstream (data already collected there).

## Repository layout & build

```
moecut/
├── CLAUDE.md
├── CMakeLists.txt
├── src/
│   ├── main.c            # CLI dispatch (info / profile / prune)
│   ├── info.c
│   ├── profile.c
│   ├── prune.c
│   └── common.{c,h}      # shared GGUF helpers, keep-list handling, error macros
├── vendor/
│   └── ggml/             # git submodule -> https://github.com/ggml-org/ggml
├── tests/
│   ├── make_synthetic.py # fixture generator (ported from moe-scalpel)
│   └── run_tests.sh      # differential tests vs the Python reference
└── docs/
    └── LOGBOOK.md        # start it on day one
```

Setup:

```bash
git init moecut && cd moecut
git submodule add https://github.com/ggml-org/ggml vendor/ggml
```

CMake: `add_subdirectory(vendor/ggml)` and link the `ggml` target (this provides the
gguf API and the type traits for quantized formats; no GPU backends needed —
`-DGGML_CUDA=OFF` and friends, CPU-only is fine since moecut never runs inference).
Include paths come from the ggml target itself. Pin the submodule to a known-good
commit and record it; treat ggml API changes as an explicit upgrade task, not drift.
Anyone cloning must use `git clone --recursive` (document it in the README).

**Note on ggml vs llama.cpp:** the full llama.cpp repo is NOT a dependency of this
project. It is needed only as (a) study material — `tools/gguf-split/gguf-split.cpp` —
and (b) external validation toolchain (llama-imatrix, llama-quantize,
llama-perplexity for end-to-end tests). It lives outside this repo.

## Environment

CachyOS (Arch), fish shell, RTX 3070 Ti 8 GB, CUDA build of llama.cpp available at
`~/src/llama.cpp` (adjust) — used for validation only, see note above. Build system:
CMake ≥ 3.14 (matches ggml). C compiler: gcc/clang, `-Wall -Wextra`, sanitizers in
debug builds (ASan is non-negotiable for a tool that does pointer arithmetic over
multi-GB mmapped files). moecut's own code is C11; the vendored ggml compiles with
its own settings (it contains C++ — that's fine, it stays inside the submodule
boundary and moecut only consumes the C API).
