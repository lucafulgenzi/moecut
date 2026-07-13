#!/usr/bin/env python3
"""Verify byte-level behavior of the synthetic pruned GGUF."""

import struct
import sys
from pathlib import Path

GGUF_ALIGN = 32
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8
GGML_TYPE_F32 = 0


def align32(x: int) -> int:
    return x + ((GGUF_ALIGN - (x % GGUF_ALIGN)) % GGUF_ALIGN)


def get_str(data: bytes, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<Q", data, off)
    off += 8
    s = data[off : off + n].decode("utf-8")
    return s, off + n


def read(path: Path):
    data = path.read_bytes()
    if data[:4] != b"GGUF":
        raise AssertionError("not a GGUF file")
    version, n_tensors, n_kv = struct.unpack_from("<Iqq", data, 4)
    if version != 3:
        raise AssertionError(f"unexpected GGUF version {version}")
    off = 24
    kv = {}
    for _ in range(n_kv):
        key, off = get_str(data, off)
        (typ,) = struct.unpack_from("<i", data, off)
        off += 4
        if typ == GGUF_TYPE_STRING:
            val, off = get_str(data, off)
        elif typ == GGUF_TYPE_UINT32:
            (val,) = struct.unpack_from("<I", data, off)
            off += 4
        else:
            raise AssertionError(f"unsupported synthetic KV type {typ}")
        kv[key] = val

    tensors = {}
    for _ in range(n_tensors):
        name, off = get_str(data, off)
        (n_dims,) = struct.unpack_from("<I", data, off)
        off += 4
        dims = list(struct.unpack_from("<" + "q" * n_dims, data, off))
        off += 8 * n_dims
        typ, tensor_off = struct.unpack_from("<iQ", data, off)
        off += 12
        tensors[name] = (dims, typ, tensor_off)

    data_base = align32(off)
    return data, kv, tensors, data_base


def f32_tensor(data: bytes, data_base: int, meta) -> list[float]:
    dims, typ, tensor_off = meta
    if typ != GGML_TYPE_F32:
        raise AssertionError("expected F32 tensor")
    n = 1
    for d in dims:
        n *= d
    return list(struct.unpack_from("<" + "f" * n, data, data_base + tensor_off))


def main() -> None:
    data, kv, tensors, data_base = read(Path(sys.argv[1]))
    assert kv["test.expert_count"] == 2

    exps = tensors["blk.0.ffn_gate_exps.weight"]
    assert exps[0] == [2, 3, 2]
    assert f32_tensor(data, data_base, exps) == [
        100.0, 101.0, 110.0, 111.0, 120.0, 121.0,
        200.0, 201.0, 210.0, 211.0, 220.0, 221.0,
    ]

    router = tensors["blk.0.ffn_gate_inp.weight"]
    assert router[0] == [2, 2]
    assert f32_tensor(data, data_base, router) == [10.0, 11.0, 20.0, 21.0]

    dense = tensors["dense.weight"]
    assert dense[0] == [2, 2]
    assert f32_tensor(data, data_base, dense) == [1.0, 2.0, 3.0, 4.0]


if __name__ == "__main__":
    main()
