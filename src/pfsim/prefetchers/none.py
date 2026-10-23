from .base import BasePrefetcher


class NoPrefetcher(BasePrefetcher):
    def on_demand(self, ev):
        return []

