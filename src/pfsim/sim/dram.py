from __future__ import annotations

from bisect import bisect_right


class DramChannel:
    def __init__(self, latency: float, transfer_cycles: int, window_cycles: int = 4096):
        self.latency = latency
        self.transfer_cycles = transfer_cycles
        self.window_cycles = window_cycles
        self.channel_free = 0.0
        self.starts: list[float] = []
        self.ends: list[float] = []
        self.total_busy = 0.0
        self.reads = {"demand": 0, "prefetch": 0}

    def request(self, arrival: float, kind: str) -> float:
        start = max(arrival, self.channel_free)
        end = start + self.transfer_cycles
        self.channel_free = end
        self.starts.append(start); self.ends.append(end)
        self.total_busy += self.transfer_cycles
        self.reads[kind] += 1
        return end + self.latency

    def utilization(self, now: float) -> float:
        def busy_before(moment: float) -> float:
            completed = bisect_right(self.ends, moment)
            busy = completed * self.transfer_cycles
            if completed < len(self.starts) and self.starts[completed] < moment:
                busy += min(moment, self.ends[completed]) - self.starts[completed]
            return busy
        busy = busy_before(now) - busy_before(now - self.window_cycles)
        return min(1.0, max(0.0, busy / self.window_cycles))
