from pfsim.sim.events import PrefetchRequest
from .base import BasePrefetcher


class FutureLookaheadPrefetcher(BasePrefetcher):
    """Diagnostic policy backed by a previously recorded L2 demand stream."""
    def __init__(self, demand_lines: list[int], lookahead: int = 16):
        if lookahead <= 0:
            raise ValueError("lookahead must be positive")
        self.demand_lines = demand_lines
        self.lookahead = lookahead

    def on_demand(self, ev):
        target_index = ev.l2_idx - 1 + self.lookahead
        if target_index >= len(self.demand_lines):
            return []
        target = self.demand_lines[target_index]
        return [PrefetchRequest(target, None, target >> 6)]

    def stats(self):
        return {"uses_future": True, "lookahead": self.lookahead}

