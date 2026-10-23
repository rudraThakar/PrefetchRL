from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from pfsim.sim.events import DemandEvent, OutcomeEvent, OutcomeKind, PrefetchRequest
from .features import quantize_delta
from .reward import RewardConfig, reward_for

PATTERNS = (1, 2, 4, -1, "d1")
REACHES = (2, 8)
N_STATES = 544
N_ACTIONS = 11


@dataclass
class _SmallDecision:
    ident: int
    state: int
    action: int
    reward: float | None = None
    next_state: int | None = None
    next_valid: np.ndarray | None = None
    updated: bool = False


class QTableSmallPrefetcher:
    """The section 6.11 single-table, 544-state comparison agent."""
    def __init__(self, alpha: float = .05, gamma: float = .6, epsilon: float = .01,
                 q0: float = .2, miss_lat_ref: float = 254.0,
                 reward: RewardConfig | None = None):
        self.alpha, self.gamma, self.epsilon, self.q0 = alpha, gamma, epsilon, q0
        self.miss_lat_ref = miss_lat_ref; self.reward_cfg = reward or RewardConfig()
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.q = np.full((N_STATES, N_ACTIONS), self.q0, dtype=np.float32)
        self.pc_table: list[tuple[int, int, int, int] | None] = [None] * 1024
        self.decisions: dict[int, _SmallDecision] = {}; self.order: deque[int] = deque()
        self.previous = None; self.next_id = self.updates = 0; self.reward_sum = 0.0
        self.action_counts = np.zeros(N_ACTIONS, dtype=np.int64)

    def _context(self, ev: DemandEvent) -> tuple[int, int, int]:
        index = ((ev.pc >> 2) ^ (ev.pc >> 12)) & 1023
        old = self.pc_table[index]
        if old is None or old[0] != ev.pc: d1 = d2 = 0
        else:
            d1 = max(-1023, min(1023, ev.line - old[1])); d2 = old[2]
        self.pc_table[index] = (ev.pc, ev.line, d1, d2)
        bw = min(3, int(ev.sys.bw_util * 4))
        acc = 0 if ev.sys.acc_ema < .25 else 1 if ev.sys.acc_ema < .5 else 2 if ev.sys.acc_ema < .75 else 3
        state = (((quantize_delta(d1) * 2 + int(d1 == d2)) * 4 + bw) * 4 + acc)
        return state, d1, d2

    def _valid(self, line: int, d1: int) -> np.ndarray:
        valid = np.zeros(N_ACTIONS, dtype=bool); valid[0] = True
        for action in range(1, N_ACTIONS):
            pattern = PATTERNS[(action - 1) // 2]
            delta = d1 if pattern == "d1" else pattern
            target = line + REACHES[(action - 1) % 2] * delta
            valid[action] = delta != 0 and abs(delta) <= 63 and target >= 0 and target >> 6 == line >> 6
        return valid

    def _finalize(self, decision: _SmallDecision) -> None:
        if decision.updated or decision.reward is None or (self.gamma and decision.next_state is None): return
        target = decision.reward
        if self.gamma and decision.next_state is not None:
            choices = np.flatnonzero(decision.next_valid)
            target += self.gamma * float(self.q[decision.next_state, choices].max())
        self.q[decision.state, decision.action] += self.alpha * (target - self.q[decision.state, decision.action])
        decision.updated = True; self.updates += 1; self.reward_sum += decision.reward

    def on_demand(self, ev: DemandEvent) -> list[PrefetchRequest]:
        state, d1, _ = self._context(ev); valid = self._valid(ev.line, d1)
        if self.previous is not None:
            self.previous.next_state = state; self.previous.next_valid = valid; self._finalize(self.previous)
        choices = np.flatnonzero(valid)
        if self.rng.random() < self.epsilon: action = int(self.rng.choice(choices))
        else:
            values = self.q[state, choices]; action = int(self.rng.choice(choices[values == values.max()]))
        decision = _SmallDecision(self.next_id, state, action); self.next_id += 1
        self.decisions[decision.ident] = decision; self.order.append(decision.ident)
        while len(self.order) > 8192: self.decisions.pop(self.order.popleft(), None)
        self.previous = decision; self.action_counts[action] += 1
        if action == 0:
            decision.reward = -self.reward_cfg.c_none_miss if not ev.l2_hit else 0.0
            self._finalize(decision); return []
        pattern = PATTERNS[(action - 1) // 2]; delta = d1 if pattern == "d1" else int(pattern)
        reach = REACHES[(action - 1) % 2]
        return [PrefetchRequest(ev.line + reach * delta, decision.ident, ev.page)]

    def on_outcome(self, out: OutcomeEvent) -> None:
        decision = self.decisions.get(out.tag)
        if decision is None: return
        if out.kind == OutcomeKind.POLLUTION:
            self.q[decision.state, decision.action] += self.alpha * (-self.reward_cfg.c_pol)
        else:
            decision.reward = reward_for(out, self.reward_cfg, self.miss_lat_ref); self._finalize(decision)

    def storage_bits(self) -> int:
        return N_STATES * N_ACTIONS * 16 + 1024 * (16 + 32 + 22)

    def stats(self) -> dict[str, Any]:
        return {"mean_reward": self.reward_sum / self.updates if self.updates else 0.0,
                "epsilon": self.epsilon, "updates": self.updates,
                "action_counts": self.action_counts.tolist()}

