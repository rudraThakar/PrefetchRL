from pfsim.sim.events import PrefetchRequest
from .base import BasePrefetcher


class NextLinePrefetcher(BasePrefetcher):
    def __init__(self, degree: int = 2):
        self.degree = degree

    def on_demand(self, ev):
        return [PrefetchRequest(ev.line + k, None, ev.page) for k in range(1, self.degree + 1)]

