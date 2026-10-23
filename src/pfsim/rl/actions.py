from __future__ import annotations

import numpy as np
from functools import lru_cache

from .features import History

PATTERNS = ("P1", "P2", "P4", "M1", "PC_D1", "PC_D2", "PAGE_D1")
REACHES = (1, 2, 4, 8, 16)
N_ACTIONS = 36


def decode(action: int) -> tuple[str | None, int]:
    if action == 0: return None, 0
    value = action - 1
    return PATTERNS[value // 5], REACHES[value % 5]


def delta_for(pattern: str, h: History) -> int:
    return {"P1": 1, "P2": 2, "P4": 4, "M1": -1,
            "PC_D1": h.d1, "PC_D2": h.d2, "PAGE_D1": h.pd1}[pattern]


@lru_cache(maxsize=8192)
def _valid_bytes(offset: int, d1: int, d2: int, pd1: int,
                 patterns: tuple[str, ...], reaches: tuple[int, ...]) -> bytes:
    valid = np.zeros(N_ACTIONS, dtype=bool); valid[0] = True
    deltas = {"P1": 1, "P2": 2, "P4": 4, "M1": -1,
              "PC_D1": d1, "PC_D2": d2, "PAGE_D1": pd1}
    for pidx, pattern in enumerate(PATTERNS):
        if pattern not in patterns: continue
        delta = deltas[pattern]
        if delta == 0 or abs(delta) > 63: continue
        for ridx, reach in enumerate(REACHES):
            if reach not in reaches: continue
            target = offset + reach * delta
            if 0 <= target < 64:
                valid[1 + 5 * pidx + ridx] = True
    return valid.tobytes()


def valid_mask(line: int, h: History, patterns=PATTERNS, reaches=REACHES) -> np.ndarray:
    return np.frombuffer(_valid_bytes(line & 63, h.d1, h.d2, h.pd1,
                                      tuple(patterns), tuple(reaches)), dtype=bool).copy()
