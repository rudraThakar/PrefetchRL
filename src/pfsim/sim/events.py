from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class OutcomeKind(str, Enum):
    USEFUL = "useful"
    USELESS = "useless"
    REDUNDANT = "redundant"
    DROPPED = "dropped"
    POLLUTION = "pollution"


@dataclass(frozen=True)
class SysState:
    bw_util: float
    acc_ema: float
    pol_ema: float
    pf_occupancy: float
    stall_since_last: float = 0.0


@dataclass(frozen=True)
class DemandEvent:
    now: float
    pc: int
    line: int
    page: int
    offset: int
    is_write: bool
    l2_hit: bool
    l2_prefetch_hit: bool
    l2_idx: int
    sys: SysState


@dataclass(frozen=True)
class PrefetchRequest:
    line: int
    tag: Any
    trigger_page: int


@dataclass(frozen=True)
class OutcomeEvent:
    kind: OutcomeKind
    tag: Any
    u_issue: float = 0.0
    wait_cycles: float = 0.0
    reason: str | None = None

