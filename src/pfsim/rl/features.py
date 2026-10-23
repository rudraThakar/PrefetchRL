from __future__ import annotations

from dataclasses import dataclass

from pfsim.sim.events import DemandEvent

ROWS = (289, 4913, 4352, 1088, 144, 272)


def quantize_delta(delta: int) -> int:
    if delta == 0: return 0
    magnitude = abs(delta)
    cls = magnitude if magnitude <= 4 else 5 if magnitude <= 8 else 6 if magnitude <= 16 else 7 if magnitude <= 64 else 8
    return cls if delta > 0 else 8 + cls


@dataclass(frozen=True)
class History:
    pc_hash: int
    d1: int
    d2: int
    d3: int
    pd1: int


def feature_indices(ev: DemandEvent, h: History, pc_bits: int = 8) -> tuple[int, ...]:
    q1, q2, q3 = quantize_delta(h.d1), quantize_delta(h.d2), quantize_delta(h.d3)
    bw = min(3, int(ev.sys.bw_util * 4))
    acc = 0 if ev.sys.acc_ema < .25 else 1 if ev.sys.acc_ema < .5 else 2 if ev.sys.acc_ema < .75 else 3
    occupancy = 0 if ev.sys.pf_occupancy < 1/3 else 1 if ev.sys.pf_occupancy < 2/3 else 2
    pollution = 0 if ev.sys.pol_ema < .02 else 1 if ev.sys.pol_ema < .10 else 2
    return (q1 * 17 + q2,
            q1 * 289 + q2 * 17 + q3,
            (h.pc_hash >> (10 - pc_bits)) * 17 + q1,
            ev.offset * 17 + quantize_delta(h.pd1),
            ((bw * 4 + acc) * 3 + occupancy) * 3 + pollution,
            (q1 * 4 + bw) * 4 + acc)

