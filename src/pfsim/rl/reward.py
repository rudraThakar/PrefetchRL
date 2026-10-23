from __future__ import annotations

from dataclasses import dataclass

from pfsim.sim.events import OutcomeEvent, OutcomeKind


@dataclass
class RewardConfig:
    c_use: float = 1.0
    t_min: float = 0.2
    c_w0: float = 0.3
    c_w1: float = 1.0
    c_red: float = 0.05
    c_drop: float = 0.10
    c_none_miss: float = 0.05
    c_pol: float = 0.6


def reward_for(out: OutcomeEvent, cfg: RewardConfig, miss_lat_ref: float) -> float:
    if out.kind == OutcomeKind.USEFUL:
        lateness = min(1.0, out.wait_cycles / miss_lat_ref)
        return cfg.c_use * (cfg.t_min + (1 - cfg.t_min) * (1 - lateness))
    if out.kind == OutcomeKind.USELESS:
        return -(cfg.c_w0 + (cfg.c_w1 - cfg.c_w0) * out.u_issue)
    if out.kind == OutcomeKind.REDUNDANT: return -cfg.c_red
    if out.kind == OutcomeKind.DROPPED: return -cfg.c_drop
    if out.kind == OutcomeKind.POLLUTION: return -cfg.c_pol
    raise ValueError(out.kind)

