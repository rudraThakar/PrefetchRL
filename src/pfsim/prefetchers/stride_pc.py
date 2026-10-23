from dataclasses import dataclass

from pfsim.sim.events import PrefetchRequest
from .base import BasePrefetcher


@dataclass
class _StrideEntry:
    tag: int = -1
    last_line: int = 0
    stride: int = 0
    conf: int = 0


class StridePCPrefetcher(BasePrefetcher):
    def __init__(self, degree: int = 4, min_conf: int = 2):
        self.degree = degree
        self.min_conf = min_conf
        self.reset(0)

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.table = [_StrideEntry() for _ in range(256)]

    def on_demand(self, ev):
        entry = self.table[(ev.pc >> 2) & 255]
        if entry.tag != ev.pc:
            entry.tag, entry.last_line, entry.stride, entry.conf = ev.pc, ev.line, 0, 0
            return []
        stride = ev.line - entry.last_line
        entry.last_line = ev.line
        if stride == 0:
            return []
        if stride == entry.stride:
            entry.conf = min(3, entry.conf + 1)
        else:
            entry.conf = max(0, entry.conf - 1)
            if entry.conf == 0:
                entry.stride = stride
        if entry.conf >= self.min_conf and entry.stride:
            return [PrefetchRequest(ev.line + k * entry.stride, None, ev.page)
                    for k in range(1, self.degree + 1)]
        return []

    def storage_bits(self) -> int:
        return 256 * (16 + 32 + 16 + 2)

