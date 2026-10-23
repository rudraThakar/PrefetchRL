from collections import deque

from pfsim.sim.events import PrefetchRequest
from .base import BasePrefetcher


class BestOffsetPrefetcher(BasePrefetcher):
    OFFSETS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25,
               27, 30, 32, 36, 40, 45, 48, 50, 54, 60)

    def __init__(self, degree: int = 1, roundmax: int = 100,
                 scoremax: int = 31, badscore: int = 1, reference_latency: float = 254):
        self.degree, self.roundmax = degree, roundmax
        self.scoremax, self.badscore = scoremax, badscore
        self.reference_latency = reference_latency
        self.reset(0)

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.rr_order: deque[int] = deque()
        self.rr = set()
        self.pending: deque[tuple[float, int]] = deque()
        self.scores = [0] * len(self.OFFSETS)
        self.test_idx = self.round = 0
        self.active = None

    def _end_phase(self):
        best = max(range(len(self.OFFSETS)), key=self.scores.__getitem__)
        self.active = self.OFFSETS[best] if self.scores[best] > self.badscore else None
        self.scores = [0] * len(self.OFFSETS); self.round = self.test_idx = 0

    def on_demand(self, ev):
        while self.pending and self.pending[0][0] <= ev.now:
            _, line = self.pending.popleft()
            if line not in self.rr:
                if len(self.rr_order) >= 256:
                    self.rr.remove(self.rr_order.popleft())
                self.rr.add(line); self.rr_order.append(line)
        offset = self.OFFSETS[self.test_idx]
        prior = ev.line - offset
        if prior in self.rr and (prior >> 6) == ev.page:
            self.scores[self.test_idx] += 1
        reached = self.scores[self.test_idx] >= self.scoremax
        self.test_idx += 1
        if self.test_idx == len(self.OFFSETS):
            self.test_idx = 0; self.round += 1
        if reached or self.round >= self.roundmax:
            self._end_phase()
        self.pending.append((ev.now + self.reference_latency, ev.line))
        if self.active is None:
            return []
        return [PrefetchRequest(ev.line + k * self.active, None, ev.page)
                for k in range(1, self.degree + 1)]

    def storage_bits(self) -> int:
        return 256 * 16 + 26 * 5 + 64
