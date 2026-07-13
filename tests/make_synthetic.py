#!/usr/bin/env python3
"""Create tiny GGUF fixtures for moecut smoke tests.

This is deliberately a small test writer, not a production GGUF implementation.
The product code consumes ggml/gguf APIs; tests use explicit bytes so failures are
easy to inspect and do not depend on another helper binary.
"""

import argparse
import struct
from pathlib import Path

GGUF_VERSION = 3
GGUF_ALIGN = 32

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8
GGML_TYPE_F32 = 0


def pad32(data: bytearray) -> None:
    data.extend(b"\0" * ((GGUF_ALIGN - (len(data) % GGUF_ALIGN)) % GGUF_ALIGN))


def put_str(out: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    out.extend(struct.pack("<Q", len(b)))
    out.extend(b)


def tensor_bytes(values: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


def write_gguf(path: Path, kvs: list[tuple[str, int, object]], tensors: list[tuple[str, list[int], int, bytes]]) -> None:
    tensor_offsets: list[int] = []
    off = 0
    for _, _, _, data in tensors:
        tensor_offsets.append(off)
        off += len(data)
        off += (GGUF_ALIGN - (off % GGUF_ALIGN)) % GGUF_ALIGN

    out = bytearray()
    out.extend(b"GGUF")
    out.extend(struct.pack("<Iqq", GGUF_VERSION, len(tensors), len(kvs)))

    for key, typ, val in kvs:
        put_str(out, key)
        out.extend(struct.pack("<i", typ))
        if typ == GGUF_TYPE_STRING:
            put_str(out, val)
        elif typ == GGUF_TYPE_UINT32:
            out.extend(struct.pack("<I", val))
        else:
            raise ValueError(f"unsupported test KV type {typ}")

    for (name, dims, typ, _), offset in zip(tensors, tensor_offsets):
        put_str(out, name)
        out.extend(struct.pack("<I", len(dims)))
        for d in dims:
            out.extend(struct.pack("<q", d))
        out.extend(struct.pack("<iQ", typ, offset))

    pad32(out)
    for _, _, _, data in tensors:
        out.extend(data)
        pad32(out)

    path.write_bytes(out)


def make_model(path: Path) -> None:
    kvs = [
        ("general.architecture", GGUF_TYPE_STRING, "test"),
        ("test.expert_count", GGUF_TYPE_UINT32, 4),
        ("test.expert_used_count", GGUF_TYPE_UINT32, 2),
    ]

    expert_values: list[float] = []
    for expert in range(4):
        # Expert slabs are contiguous: ne0 * ne1 floats per expert.
        for row in range(3):
            for col in range(2):
                expert_values.append(expert * 100.0 + row * 10.0 + col)

    router_values: list[float] = []
    for expert in range(4):
        for col in range(2):
            router_values.append(expert * 10.0 + col)

    tensors = [
        ("blk.0.ffn_gate_exps.weight", [2, 3, 4], GGML_TYPE_F32, tensor_bytes(expert_values)),
        ("blk.0.ffn_gate_inp.weight", [2, 4], GGML_TYPE_F32, tensor_bytes(router_values)),
        ("dense.weight", [2, 2], GGML_TYPE_F32, tensor_bytes([1.0, 2.0, 3.0, 4.0])),
    ]
    write_gguf(path, kvs, tensors)


def make_imatrix(path: Path) -> None:
    kvs = [
        ("general.architecture", GGUF_TYPE_STRING, "imatrix"),
        ("imatrix.chunk_count", GGUF_TYPE_UINT32, 1),
        ("imatrix.chunk_size", GGUF_TYPE_UINT32, 8),
    ]
    tensors = [
        ("blk.0.ffn_gate_exps.weight.counts", [4], GGML_TYPE_F32, tensor_bytes([1.0, 10.0, 5.0, 2.0])),
        ("blk.0.ffn_gate_exps.weight.in_sum2", [2, 4], GGML_TYPE_F32, tensor_bytes([1.0, 1.0, 10.0, 10.0, 5.0, 5.0, 2.0, 2.0])),
    ]
    write_gguf(path, kvs, tensors)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_model(args.out_dir / "synthetic-model.gguf")
    make_imatrix(args.out_dir / "synthetic-imatrix.gguf")


if __name__ == "__main__":
    main()
