# moecut

`moecut` is a single-binary C tool for profiling and pruning Mixture-of-Experts
models in GGUF format.

The goal is to prune already-quantized GGUFs directly: download a Q4/Q8 MoE
model, collect an imatrix profile on your domain, then write a smaller GGUF
without going through F16, requantization, or a new imatrix pass.

This repository is early-stage. The current implementation has a working
`info`, `profile`, and conservative `prune` path covered by synthetic GGUF
smoke tests.

## Why

MoE routing is domain-dependent. A coding workload and a prose workload may use
different cold experts, so the useful cut is often domain-specific. `moecut`
turns that observation into a direct GGUF surgery tool: keep the hot experts for
your domain and remove cold expert slabs from the model file.

Quantized pruning works because GGUF quantization blocks live inside tensor rows,
while MoE expert removal cuts along the outer expert dimension. A whole-expert
copy does not split quantization blocks.

## Current Commands

```bash
moecut info <model.gguf>
moecut profile <imatrix.gguf>
moecut prune <in.gguf> <out.gguf> --profile <profile.json> --keep N
```

`profile` writes JSON to stdout. `prune` reads that JSON and keeps the top `N`
experts per layer according to the profile rank.

`profile` reads imatrix tensor data in bounded chunks; it does not load the full
imatrix blob into memory.

## Download GGUF Models

The helper script `scripts/hf-gguf-download.sh` downloads one GGUF file from a
Hugging Face model repository.

Requirements:

* Bash 4 or newer.
* `jq`.
* `curl` or `wget`.
* Standard Linux userland tools such as `mkdir`.

Install the script dependencies on Debian/Ubuntu:

```bash
sudo apt install jq curl
```

Or, if you prefer `wget`:

```bash
sudo apt install jq wget
```

Pass a repository id, a Hugging Face URL, or a search string:

```bash
scripts/hf-gguf-download.sh unsloth/Qwen3-30B-A3B-GGUF
scripts/hf-gguf-download.sh https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF
scripts/hf-gguf-download.sh "Qwen3 30B GGUF"
```

The script lists available `.gguf` files, shows the quantization label inferred
from each filename, and prompts for the file to download. Use `--out DIR` to
choose an output directory, `--revision REV` for a non-`main` branch, and
`HF_TOKEN` or `--token TOKEN` for gated/private repositories.

To download a specific quantization, pass the exact GGUF filename:

```bash
scripts/hf-gguf-download.sh \
  --file Qwen3-30B-A3B-Q4_K_M.gguf \
  lmstudio-community/Qwen3-30B-A3B-GGUF
```

The same works with Hugging Face's `show_file_info` URLs:

```bash
scripts/hf-gguf-download.sh \
  'https://huggingface.co/lmstudio-community/Qwen3-30B-A3B-GGUF?show_file_info=Qwen3-30B-A3B-Q4_K_M.gguf'
```

## Supported Path

Currently supported:

* F16 and quantized GGUF tensor byte copying, as long as expert slabs are
  contiguous in the expected ggml layout.
* Per-layer keep lists with the same number of experts kept in every layer.
* Metadata patching for `{arch}.expert_count`.
* Router tensor slicing.
* Refusal for group-limited routing.

Not yet complete:

* Real quantized fixture in CI.
* `--dry-run` size forecast.
* `profile --compare`.
* Group-limited routing support.
* Full differential testing against the frozen Python reference.

## Build

Clone with submodules:

```bash
git clone --recursive <repo-url>
cd moecut
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

Configure and build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Debug builds enable AddressSanitizer and UndefinedBehaviorSanitizer for moecut's
own code:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

The binary is written to:

```bash
build/moecut
```

## Tests

Run the synthetic smoke test:

```bash
bash tests/run_tests.sh
```

The test generator creates tiny GGUF files where expert values encode expert
ids. The verifier checks that pruning keeps the expected expert slabs, slices
router rows, preserves dense tensors, and writes a GGUF that can be reopened.

## Example Workflow

Generate an imatrix with llama.cpp on your target domain:

```bash
llama-imatrix -m model.Q4_K_M.gguf -f domain.txt -o domain.imatrix.gguf
```

Create a moecut profile:

```bash
build/moecut profile domain.imatrix.gguf > domain.profile.json
```

Inspect the model:

```bash
build/moecut info model.Q4_K_M.gguf
```

Prune to a fixed expert count:

```bash
build/moecut prune model.Q4_K_M.gguf model.domain.Q4_K_M.gguf \
  --profile domain.profile.json \
  --keep 48
```

Choose `--keep` carefully. It must be at least the model router's top-k expert
use count.

## Validation

Always validate pruned models on held-out data. A pruned model is a domain model:
it should be evaluated on the target domain and on at least one off-domain set so
you can see the intended tradeoff.

Recommended external validation with llama.cpp:

```bash
llama-perplexity -m original.gguf -f heldout.txt --kl-divergence-base ref.bin
llama-perplexity -m pruned.gguf   -f heldout.txt --kl-divergence ref.bin
```

Track perplexity, KL divergence, and same-top-token rate. Median KLD describes
typical-token drift; mean KLD catches tail damage.

## Development Notes

`moecut` uses `vendor/ggml` for GGUF reading/writing and ggml tensor type traits.
The full llama.cpp tree is not a build dependency; it is useful for producing
imatrix files and validating output models.

Keep output paths explicit for large files. Do not write multi-GB artifacts to
`/tmp` unless you know it is not tmpfs.
