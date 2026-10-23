from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pfsim.sim.events import DemandEvent, OutcomeEvent, OutcomeKind, PrefetchRequest
from .actions import N_ACTIONS, decode, delta_for, valid_mask
from .features import History, ROWS, feature_indices
from .reward import RewardConfig, reward_for


@dataclass
class AgentConfig:
    alpha: float = .05
    gamma: float = .6
    eps_0: float = .12
    eps_min: float = .01
    half_life: int = 150_000
    q0: float = .2
    delta_max: float = 2.0
    pc_bits: int = 8
    double_q: bool = True
    learn: bool = True
    explore: bool = True
    explore_mode: str = "eps_decay"
    history_size: int = 8192
    reaches: tuple[int, ...] = (1, 2, 4, 8, 16)
    patterns: tuple[str, ...] = ("P1", "P2", "P4", "M1", "PC_D1", "PC_D2", "PAGE_D1")
    feature_drop: tuple[str, ...] = ()
    reward_pollution: bool = True
    reward_bw_term: bool = True
    reward_graded: bool = True
    miss_lat_ref: float = 254.0
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class _Decision:
    ident: int
    state: tuple[int, ...]
    action: int
    valid: np.ndarray
    reward: float | None = None
    next_state: tuple[int, ...] | None = None
    next_valid: np.ndarray | None = None
    updated: bool = False


class RLPrefetcher:
    def __init__(self, config: AgentConfig | None = None):
        self.cfg = config or AgentConfig()
        self._loaded_weights: tuple[list[np.ndarray], list[np.ndarray]] | None = None
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        names = tuple(f"f{i}" for i in range(1, 7))
        unknown = set(self.cfg.feature_drop) - set(names)
        if unknown: raise ValueError(f"unknown feature tables: {sorted(unknown)}")
        self.active_features = tuple(i for i, name in enumerate(names) if name not in self.cfg.feature_drop)
        if not self.active_features: raise ValueError("at least one feature table is required")
        if not 1 <= self.cfg.pc_bits <= 10: raise ValueError("pc_bits must be between 1 and 10")
        configured_rows = list(ROWS); configured_rows[2] = (1 << self.cfg.pc_bits) * 17
        self.configured_rows = tuple(configured_rows)
        self.active_rows = tuple(self.configured_rows[i] for i in self.active_features)
        init = self.cfg.q0 / len(self.active_features)
        if self._loaded_weights is None:
            self.weights_a = [np.full((rows, N_ACTIONS), init, np.float32) for rows in self.active_rows]
            self.weights_b = ([np.full((rows, N_ACTIONS), init, np.float32) for rows in self.active_rows]
                              if self.cfg.double_q else self.weights_a)
        else:
            self.weights_a = [value.copy() for value in self._loaded_weights[0]]
            self.weights_b = ([value.copy() for value in self._loaded_weights[1]]
                              if self.cfg.double_q else self.weights_a)
        self.pc_table: list[tuple[int, int, int, int, int] | None] = [None] * 1024
        self.page_table: list[tuple[int, int, int] | None] = [None] * 256
        self.decisions: dict[int, _Decision] = {}; self.order: deque[int] = deque()
        self.previous: _Decision | None = None; self.next_id = self.steps = self.updates = 0
        self.reward_sum = 0.0; self.action_counts = np.zeros(N_ACTIONS, dtype=np.int64)

    def _history(self, ev: DemandEvent) -> History:
        ph = ((ev.pc >> 2) ^ (ev.pc >> 12)) & 1023
        old = self.pc_table[ph]
        if old is None or old[0] != ev.pc:
            d1 = d2 = d3 = 0
        else:
            _, last, old_d1, old_d2, _ = old
            d1 = max(-1023, min(1023, ev.line - last)); d2, d3 = old_d1, old_d2
        self.pc_table[ph] = (ev.pc, ev.line, d1, d2, d3)
        pi = ev.page & 255; page_old = self.page_table[pi]
        if page_old is None or page_old[0] != ev.page:
            pd1 = 0
        else:
            pd1 = max(-63, min(63, ev.offset - page_old[1]))
        self.page_table[pi] = (ev.page, ev.offset, pd1)
        return History(ph, d1, d2, d3, pd1)

    def _q(self, weights, state):
        value = np.zeros(N_ACTIONS, dtype=np.float32)
        for table, row in zip(weights, state): value += table[row]
        return value

    def _epsilon(self):
        if self.cfg.explore_mode == "eps_fixed": return self.cfg.eps_min
        if self.cfg.explore_mode == "softmax": return 0.0
        return max(self.cfg.eps_min, self.cfg.eps_0 * .5 ** (self.steps / self.cfg.half_life))

    def _select(self, state, valid):
        candidates = np.flatnonzero(valid)
        values = (self._q(self.weights_a, state) + self._q(self.weights_b, state)) / 2
        if self.cfg.explore and self.cfg.explore_mode == "softmax":
            progress = min(1.0, self.steps / (2 * self.cfg.half_life))
            temperature = .5 + (.05 - .5) * progress
            logits = (values[candidates] - values[candidates].max()) / temperature
            probabilities = np.exp(logits); probabilities /= probabilities.sum()
            return int(self.rng.choice(candidates, p=probabilities))
        if self.cfg.explore_mode not in {"eps_decay", "eps_fixed", "softmax"}:
            raise ValueError(f"unknown exploration mode {self.cfg.explore_mode!r}")
        if self.cfg.explore and self.rng.random() < self._epsilon():
            return int(self.rng.choice(candidates))
        best = candidates[values[candidates] == values[candidates].max()]
        return int(self.rng.choice(best))

    def _update(self, decision: _Decision, correction: float | None = None):
        if not self.cfg.learn: return
        update_a = not self.cfg.double_q or bool(self.rng.integers(2))
        x = self.weights_a if update_a else self.weights_b
        y = self.weights_b if update_a else self.weights_a
        current = float(self._q(x, decision.state)[decision.action])
        if correction is not None:
            error = correction
        else:
            target = float(decision.reward)
            if self.cfg.gamma > 0 and decision.next_state is not None:
                qx = self._q(x, decision.next_state)
                choices = np.flatnonzero(decision.next_valid)
                star = int(choices[np.argmax(qx[choices])])
                target += self.cfg.gamma * float(self._q(y, decision.next_state)[star])
            error = target - current
        delta = float(np.clip(error, -self.cfg.delta_max, self.cfg.delta_max))
        count = len(self.active_features)
        for table, row in zip(x, decision.state):
            table[row, decision.action] += self.cfg.alpha / count * delta
        self.updates += 1

    def _finalize(self, decision):
        if (decision.reward is not None and (decision.next_state is not None or self.cfg.gamma == 0)
                and not decision.updated):
            self._update(decision); decision.updated = True
            self.reward_sum += decision.reward

    def on_demand(self, ev: DemandEvent):
        history = self._history(ev)
        all_features = feature_indices(ev, history, self.cfg.pc_bits)
        state = tuple(all_features[i] for i in self.active_features)
        valid = valid_mask(ev.line, history, self.cfg.patterns, self.cfg.reaches)
        if self.previous is not None:
            self.previous.next_state = state; self.previous.next_valid = valid
            self._finalize(self.previous)
        action = self._select(state, valid); self.steps += 1; self.action_counts[action] += 1
        decision = _Decision(self.next_id, state, action, valid); self.next_id += 1
        self.decisions[decision.ident] = decision; self.order.append(decision.ident)
        while len(self.order) > self.cfg.history_size:
            self.decisions.pop(self.order.popleft(), None)
        self.previous = decision
        if action == 0:
            decision.reward = -self.cfg.reward.c_none_miss if not ev.l2_hit else 0.0
            self._finalize(decision)
            return []
        pattern, reach = decode(action); delta = delta_for(pattern, history)
        return [PrefetchRequest(ev.line + reach * delta, decision.ident, ev.page)]

    def on_outcome(self, out: OutcomeEvent) -> None:
        decision = self.decisions.get(out.tag)
        if decision is None: return
        if out.kind == OutcomeKind.POLLUTION:
            if self.cfg.reward_pollution:
                self._update(decision, -self.cfg.reward.c_pol)
        else:
            reward_cfg = self.cfg.reward
            if not self.cfg.reward_bw_term or not self.cfg.reward_graded:
                from dataclasses import replace
                reward_cfg = replace(reward_cfg,
                    c_w1=reward_cfg.c_w0 if not self.cfg.reward_bw_term else reward_cfg.c_w1,
                    t_min=1.0 if not self.cfg.reward_graded else reward_cfg.t_min)
            decision.reward = reward_for(out, reward_cfg, self.cfg.miss_lat_ref)
            self._finalize(decision)

    def storage_bits(self) -> int:
        weight_sets = 2 if self.cfg.double_q else 1
        weights = sum(self.active_rows) * N_ACTIONS * 16 * weight_sets
        history = 1024 * (16 + 32 + 33) + 256 * (16 + 6 + 7 + 8)
        return weights + history

    def stats(self) -> dict[str, Any]:
        return {"mean_reward": self.reward_sum / self.updates if self.updates else 0.0,
                "epsilon": self._epsilon(), "updates": self.updates,
                "action_counts": self.action_counts.tolist()}

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"a_{i}": value for i, value in enumerate(self.weights_a)}
        if self.cfg.double_q:
            arrays.update({f"b_{i}": value for i, value in enumerate(self.weights_b)})
        arrays["active_features"] = np.asarray(self.active_features, dtype=np.int16)
        np.savez_compressed(path, **arrays)

    def load(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            active = tuple(int(v) for v in data["active_features"])
            expected = tuple(i for i in range(6) if f"f{i + 1}" not in self.cfg.feature_drop)
            if active != expected:
                raise ValueError(f"checkpoint features {active} do not match configuration {expected}")
            a = [data[f"a_{i}"].copy() for i in range(len(active))]
            b = ([data[f"b_{i}"].copy() for i in range(len(active))]
                 if self.cfg.double_q else a)
        configured_rows = list(ROWS); configured_rows[2] = (1 << self.cfg.pc_bits) * 17
        expected_shapes = [(configured_rows[i], N_ACTIONS) for i in active]
        if [value.shape for value in a] != expected_shapes or [value.shape for value in b] != expected_shapes:
            raise ValueError("checkpoint weight shapes do not match configuration")
        self._loaded_weights = (a, b)
        self.reset(0)
