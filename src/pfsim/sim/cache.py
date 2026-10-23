from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class Entry:
    line: int
    ready: float
    pf: bool = False
    used: bool = False
    resolved: bool = False
    pf_id: int = -1


class SetAssociativeCache:
    def __init__(self, size_kib: int, ways: int, latency: float, line_bytes: int = 64):
        lines = size_kib * 1024 // line_bytes
        if ways <= 0 or lines < ways or lines % ways:
            raise ValueError("cache size must contain an integral number of sets")
        self.ways = ways
        self.set_count = lines // ways
        self.latency = float(latency)
        self._sets: list[OrderedDict[int, Entry]] = [OrderedDict() for _ in range(self.set_count)]

    def _set(self, line: int) -> OrderedDict[int, Entry]:
        return self._sets[line % self.set_count]

    def lookup(self, line: int) -> Entry | None:
        return self._set(line).get(line)

    def contains(self, line: int) -> bool:
        return line in self._set(line)

    def touch(self, line: int) -> None:
        bucket = self._set(line)
        if line in bucket:
            bucket.move_to_end(line)

    def install(self, line: int, entry: Entry) -> Entry | None:
        bucket = self._set(line)
        if line in bucket:
            bucket[line] = entry
            bucket.move_to_end(line)
            return None
        victim = bucket.popitem(last=False)[1] if len(bucket) >= self.ways else None
        bucket[line] = entry
        return victim

    def clear(self) -> None:
        for bucket in self._sets:
            bucket.clear()

    def entries(self):
        for bucket in self._sets:
            yield from bucket.values()

