from __future__ import annotations

from collections import deque


class DramChannel:
    def __init__(self, latency: float, transfer_cycles: int, window_cycles: int = 4096):
        self.latency = latency
        self.transfer_cycles = transfer_cycles
        self.window_cycles = window_cycles
        self.channel_free = 0.0
        self.requests: deque[tuple[float, float]] = deque()
        self.total_busy = 0.0
        self.reads = {"demand": 0, "prefetch": 0}

    def request(self, arrival: float, kind: str) -> float:
        start = max(arrival, self.channel_free)
        end = start + self.transfer_cycles
        self.channel_free = end
        self.requests.append((start, end))
        self.total_busy += self.transfer_cycles
        self.reads[kind] += 1
        return end + self.latency

    def utilization(self, now: float) -> float:
        lower = now - self.window_cycles
        while self.requests and self.requests[0][1] <= lower:
            self.requests.popleft()
        busy = 0.0
        for start, end in self.requests:
            if start >= now:
                break
            busy += max(0.0, min(end, now) - max(start, lower))
        return min(1.0, busy / self.window_cycles)

