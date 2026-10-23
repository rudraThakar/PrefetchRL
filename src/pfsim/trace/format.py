from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Trace:
    pc: np.ndarray
    addr: np.ndarray
    gap: np.ndarray
    is_write: np.ndarray
    dep: np.ndarray
    meta: dict[str, Any]

    def __len__(self) -> int:
        return len(self.addr)


DTYPES = {
    "pc": np.dtype("uint64"), "addr": np.dtype("uint64"),
    "gap": np.dtype("uint16"), "is_write": np.dtype("uint8"),
    "dep": np.dtype("uint8"),
}


def validate(trace: Trace) -> None:
    n = len(trace)
    for name, dtype in DTYPES.items():
        value = getattr(trace, name)
        if value.ndim != 1 or len(value) != n:
            raise ValueError(f"{name} must be a one-dimensional array of length {n}")
        if value.dtype != dtype:
            raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if n and np.any(trace.addr == 0):
        raise ValueError("trace addresses must be non-zero")
    if np.any(trace.is_write > 1) or np.any(trace.dep > 1):
        raise ValueError("is_write and dep must contain only 0 or 1")


def save_trace(trace: Trace, path: str | Path) -> None:
    validate(trace)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pc=trace.pc, addr=trace.addr, gap=trace.gap,
                       is_write=trace.is_write, dep=trace.dep,
                       meta=np.array(json.dumps(trace.meta, sort_keys=True)))


def load_trace(path: str | Path) -> Trace:
    with np.load(path, allow_pickle=False) as data:
        raw_meta = data["meta"].item()
        trace = Trace(*(data[k].copy() for k in ("pc", "addr", "gap", "is_write", "dep")),
                      meta=json.loads(str(raw_meta)))
    validate(trace)
    return trace

