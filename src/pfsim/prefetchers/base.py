from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pfsim.sim.events import DemandEvent, OutcomeEvent, PrefetchRequest


class Prefetcher(Protocol):
    def reset(self, seed: int) -> None: ...
    def on_demand(self, ev: DemandEvent) -> list[PrefetchRequest]: ...
    def on_outcome(self, out: OutcomeEvent) -> None: ...
    def storage_bits(self) -> int: ...
    def stats(self) -> dict: ...


class BasePrefetcher:
    def reset(self, seed: int) -> None:
        self.seed = seed
    def on_outcome(self, out: OutcomeEvent) -> None:
        pass
    def storage_bits(self) -> int:
        return 0
    def stats(self) -> dict:
        return {}


def make_prefetcher(name: str, **kwargs) -> Prefetcher:
    from .bop import BestOffsetPrefetcher
    from .next_line import NextLinePrefetcher
    from .none import NoPrefetcher
    from .stride_pc import StridePCPrefetcher
    from pfsim.rl.agent import RLPrefetcher
    from pfsim.rl.qtable_small import QTableSmallPrefetcher
    choices = {"none": NoPrefetcher, "next_line": NextLinePrefetcher,
               "stride_pc": StridePCPrefetcher, "bop": BestOffsetPrefetcher}
    choices["agent"] = RLPrefetcher
    choices["qtable_small"] = QTableSmallPrefetcher
    try:
        return choices[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown prefetcher {name!r}") from exc
