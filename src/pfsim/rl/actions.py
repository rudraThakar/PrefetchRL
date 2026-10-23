from __future__ import annotations

import numpy as np

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


def valid_mask(line: int, h: History, patterns=PATTERNS, reaches=REACHES) -> np.ndarray:
    valid = np.zeros(N_ACTIONS, dtype=bool); valid[0] = True
    page = line >> 6
    for pidx, pattern in enumerate(PATTERNS):
        if pattern not in patterns: continue
        delta = delta_for(pattern, h)
        if delta == 0 or abs(delta) > 63: continue
        for ridx, reach in enumerate(REACHES):
            if reach not in reaches: continue
            target = line + reach * delta
            if target >= 0 and target >> 6 == page:
                valid[1 + 5 * pidx + ridx] = True
    return valid

