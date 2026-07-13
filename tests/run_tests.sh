#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMPDIR="${TMPDIR:-$ROOT/build/test-fixtures}"

# The Codex/CI command runner may execute under ptrace-like supervision, where
# LeakSanitizer aborts before tests run. ASan/UBSan still catch invalid accesses.
export LSAN_OPTIONS="${LSAN_OPTIONS:-detect_leaks=0}"

mkdir -p "$TMPDIR"
python3 "$ROOT/tests/make_synthetic.py" "$TMPDIR"

"$ROOT/build/moecut" info "$TMPDIR/synthetic-model.gguf" > "$TMPDIR/info.txt"
"$ROOT/build/moecut" profile "$TMPDIR/synthetic-imatrix.gguf" > "$TMPDIR/profile.json"
"$ROOT/build/moecut" prune "$TMPDIR/synthetic-model.gguf" "$TMPDIR/pruned.gguf" --profile "$TMPDIR/profile.json" --keep 2 > "$TMPDIR/prune.txt"
"$ROOT/build/moecut" info "$TMPDIR/pruned.gguf" > "$TMPDIR/pruned-info.txt"
python3 "$ROOT/tests/verify_synthetic.py" "$TMPDIR/pruned.gguf"

grep -q "expert_count: 4" "$TMPDIR/info.txt"
grep -q "expert_count: 2" "$TMPDIR/pruned-info.txt"
grep -q '"rank": \[1, 2, 3, 0\]' "$TMPDIR/profile.json"

echo "synthetic smoke tests passed"
