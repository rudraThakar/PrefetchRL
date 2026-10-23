from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from typing import Any

from pfsim.config import SimConfig
from pfsim.prefetchers.base import Prefetcher
from pfsim.trace.format import Trace

from .cache import Entry, SetAssociativeCache
from .dram import DramChannel
from .events import DemandEvent, OutcomeEvent, OutcomeKind, PrefetchRequest, SysState


@dataclass
class SimulationResult:
    records: int
    instructions: int
    cycles: float
    ipc: float
    l2_accesses: int
    l2_misses: int
    useful: int
    useless: int
    redundant: int
    dropped: int
    issued: int
    unresolved: int
    pollution: int
    pollution_benign: int
    useful_after_expiry: int
    late: int
    dram_reads: int
    dram_demand_reads: int
    dram_prefetch_reads: int
    avg_bw_util: float
    storage_bits: int
    returned_requests: int

    @property
    def accuracy(self) -> float:
        resolved = self.useful + self.useless
        return self.useful / resolved if resolved else 0.0

    @property
    def late_fraction(self) -> float:
        return self.late / self.useful if self.useful else 0.0

    @property
    def pollution_rate(self) -> float:
        return self.pollution / self.l2_accesses if self.l2_accesses else 0.0

    @property
    def redundant_rate(self) -> float:
        return self.redundant / self.returned_requests if self.returned_requests else 0.0

    @property
    def dropped_rate(self) -> float:
        return self.dropped / self.returned_requests if self.returned_requests else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(accuracy=self.accuracy, late_fraction=self.late_fraction,
                    pollution_rate=self.pollution_rate,
                    redundant_rate=self.redundant_rate, dropped_rate=self.dropped_rate)
        return data


class Simulator:
    def __init__(self, config: SimConfig, prefetcher: Prefetcher):
        self.cfg, self.prefetcher = config, prefetcher

    def _reset(self, seed: int):
        c = self.cfg
        self.l1 = SetAssociativeCache(c.l1.size_kib, c.l1.ways, c.l1.latency)
        self.l2 = SetAssociativeCache(c.l2.size_kib, c.l2.ways, c.l2.latency)
        self.llc = SetAssociativeCache(c.llc.size_kib, c.llc.ways, c.llc.latency)
        self.shadow = SetAssociativeCache(c.l2.size_kib, c.l2.ways, 0)
        self.dram = DramChannel(c.dram_latency, c.transfer_cycles, c.bw_window_cycles)
        self.prefetcher.reset(seed)
        self.l2_idx = self.next_pf_id = 0
        self.book: dict[int, dict[str, Any]] = {}
        self.expiry: deque[tuple[int, int]] = deque()
        self.victims: OrderedDict[int, tuple[int, int]] = OrderedDict()
        self.acc_ema = 0.5; self.pol_ema = 0.0
        self.counts = {k: 0 for k in ("l2_accesses", "l2_misses", "useful", "useless", "redundant",
                       "dropped", "issued", "pollution", "pollution_benign",
                       "useful_after_expiry", "late", "returned")}
        self.measure = False
        self.demand_lines: list[int] = []
        self.intervals: list[dict[str, Any]] = []
        self._interval_base = {"l2": 0, "inst": 0, "now": 0.0, "useful": 0,
                               "useless": 0, "misses": 0}

    def _record_interval(self, inst: int, now: float) -> None:
        c, base = self.counts, self._interval_base
        accesses = c["l2_accesses"] - base["l2"]
        if accesses <= 0:
            return
        resolved = (c["useful"] - base["useful"]) + (c["useless"] - base["useless"])
        stats = self.prefetcher.stats()
        self.intervals.append({
            "l2_idx": self.l2_idx,
            "l2_accesses": accesses,
            "window_ipc": (inst - base["inst"]) / (now - base["now"]) if now > base["now"] else 0.0,
            "window_accuracy": (c["useful"] - base["useful"]) / resolved if resolved else 0.0,
            "window_coverage_proxy": (c["useful"] - base["useful"]) / accesses,
            "mean_reward": float(stats.get("mean_reward", 0.0)),
            "epsilon": float(stats.get("epsilon", 0.0)),
        })
        self._interval_base = {"l2": c["l2_accesses"], "inst": inst, "now": now,
                               "useful": c["useful"], "useless": c["useless"],
                               "misses": c["l2_misses"]}

    def _emit(self, kind: OutcomeKind, tag: Any, *, pf_id: int | None = None,
              wait: float = 0, reason: str | None = None):
        info = self.book.get(pf_id, {}) if pf_id is not None else {}
        measured = info.get("measured", self.measure)
        if measured:
            if kind != OutcomeKind.POLLUTION:
                self.counts[kind.value] += 1
            else:
                self.counts["pollution"] += 1
            if kind == OutcomeKind.USEFUL and wait > 0:
                self.counts["late"] += 1
        event = OutcomeEvent(kind, info.get("tag", tag), info.get("u_issue", 0.0), wait, reason)
        self.prefetcher.on_outcome(event)
        if kind in (OutcomeKind.USEFUL, OutcomeKind.USELESS) and pf_id is not None:
            info["resolved"] = True
        alpha = self.cfg.acc_ema_weight
        if kind in (OutcomeKind.USEFUL, OutcomeKind.USELESS):
            value = 1.0 if kind == OutcomeKind.USEFUL else 0.0
            self.acc_ema += alpha * (value - self.acc_ema)
            self.pol_ema *= 1 - self.cfg.pol_ema_weight
        elif kind == OutcomeKind.POLLUTION:
            self.pol_ema += self.cfg.pol_ema_weight * (1 - self.pol_ema)

    def _expire(self):
        while self.expiry and self.expiry[0][1] + self.cfg.expiry_accesses <= self.l2_idx:
            pf_id, _ = self.expiry.popleft(); info = self.book.get(pf_id)
            if info is not None and not info.get("resolved"):
                entry = self.l2.lookup(info["line"])
                if entry is not None and entry.pf_id == pf_id:
                    entry.resolved = True
                self._emit(OutcomeKind.USELESS, info["tag"], pf_id=pf_id, reason="expired")

    def _install_l2(self, entry: Entry, prefetch_fill: bool, pf_id: int = -1):
        victim = self.l2.install(entry.line, entry)
        if victim is None:
            return
        if victim.pf and not victim.used and not victim.resolved:
            victim.resolved = True
            self._emit(OutcomeKind.USELESS, None, pf_id=victim.pf_id, reason="evicted")
        elif prefetch_fill:
            self.victims[victim.line] = (pf_id, self.l2_idx)
            self.victims.move_to_end(victim.line)
            while len(self.victims) > self.cfg.victim_filter_size:
                self.victims.popitem(last=False)

    def _check_pollution(self, line: int):
        blamed = self.victims.get(line)
        if blamed is None:
            return
        pf_id, idx = blamed
        del self.victims[line]
        if self.l2_idx - idx > self.cfg.pollution_window:
            return
        if self.cfg.pollution.counterfactual and not self.shadow.contains(line):
            if self.measure: self.counts["pollution_benign"] += 1
            return
        self._emit(OutcomeKind.POLLUTION, None, pf_id=pf_id)

    def _issue(self, req: PrefetchRequest, now: float):
        self.counts["returned"] += int(self.measure)
        line = int(req.line)
        if line < 0 or (self.cfg.pf_same_page_only and (line >> 6) != req.trigger_page):
            self._emit(OutcomeKind.DROPPED, req.tag, reason="cross_page"); return
        if self.l2.contains(line):
            self._emit(OutcomeKind.REDUNDANT, req.tag); return
        inflight = sum(e.pf and not e.used and e.ready > now for e in self.l2.entries())
        if inflight >= self.cfg.pf_mshr:
            self._emit(OutcomeKind.DROPPED, req.tag, reason="mshr"); return
        llc_entry = self.llc.lookup(line)
        if llc_entry is not None:
            ready = now + self.llc.latency + max(0.0, llc_entry.ready - now)
        else:
            if self.dram.channel_free - now > self.cfg.pf_backlog_cycles:
                self._emit(OutcomeKind.DROPPED, req.tag, reason="bandwidth"); return
            ready = self.dram.request(now + self.llc.latency, "prefetch")
            self.llc.install(line, Entry(line, ready))
        pf_id = self.next_pf_id; self.next_pf_id += 1
        self.book[pf_id] = {"tag": req.tag, "line": line, "u_issue": self.dram.utilization(now),
                            "issue_idx": self.l2_idx, "measured": self.measure, "resolved": False}
        self._install_l2(Entry(line, ready, pf=True, pf_id=pf_id), True, pf_id)
        self.expiry.append((pf_id, self.l2_idx))
        if self.measure: self.counts["issued"] += 1

    def _demand(self, pc: int, addr: int, is_write: bool, now: float, stall: float) -> float:
        line = addr >> 6
        e1 = self.l1.lookup(line)
        if e1 is not None:
            self.l1.touch(line)
            return self.l1.latency + max(0.0, e1.ready - now)
        self.l2_idx += 1; self.demand_lines.append(line)
        if self.measure: self.counts["l2_accesses"] += 1
        e2 = self.l2.lookup(line)
        if e2 is not None:
            wait = max(0.0, e2.ready - now); ready = now + self.l2.latency + wait
            if e2.pf and not e2.used:
                e2.used = True
                if not e2.resolved:
                    e2.resolved = True; self._emit(OutcomeKind.USEFUL, None, pf_id=e2.pf_id, wait=wait)
                elif self.measure:
                    self.counts["useful_after_expiry"] += 1
            self.l2.touch(line)
        else:
            if self.measure: self.counts["l2_misses"] += 1
            self._check_pollution(line)
            e3 = self.llc.lookup(line)
            if e3 is not None:
                ready = now + self.llc.latency + max(0.0, e3.ready - now); self.llc.touch(line)
            else:
                ready = self.dram.request(now + self.llc.latency, "demand")
                self.llc.install(line, Entry(line, ready))
            self._install_l2(Entry(line, ready), False)
        # The shadow sees exactly the demand sequence and is updated after pollution checking.
        if self.shadow.contains(line): self.shadow.touch(line)
        else: self.shadow.install(line, Entry(line, 0))
        self._expire()
        inflight_pf = sum(e.pf and not e.used and e.ready > now for e in self.l2.entries())
        event = DemandEvent(now, pc, line, line >> 6, line & 63, is_write,
                            e2 is not None, bool(e2 is not None and e2.pf), self.l2_idx,
                            SysState(self.dram.utilization(now), self.acc_ema, self.pol_ema,
                                     inflight_pf / self.cfg.pf_mshr, stall))
        for request in self.prefetcher.on_demand(event):
            self._issue(request, now + self.cfg.pf_issue_latency)
        self.l1.install(line, Entry(line, ready))
        return ready - now

    def run(self, trace: Trace, seed: int = 0) -> SimulationResult:
        self._reset(seed)
        now = 0.0; inst = 0; last_done = 0.0
        outstanding: deque[tuple[int, float]] = deque()
        warmup = int(len(trace) * self.cfg.warmup_fraction)
        measured_inst_start = measured_now_start = 0.0
        for index, (pc, addr, gap, wr, dep) in enumerate(zip(
                trace.pc.tolist(), trace.addr.tolist(), trace.gap.tolist(),
                trace.is_write.tolist(), trace.dep.tolist())):
            if index == warmup:
                self.measure = True; measured_inst_start = inst; measured_now_start = now
                self.dram.total_busy = 0.0
                self.dram.reads = {"demand": 0, "prefetch": 0}
                self._interval_base = {"l2": 0, "inst": inst, "now": now, "useful": 0,
                                       "useless": 0, "misses": 0}
            before_stall = now
            now += (gap + 1) / self.cfg.width; inst += gap + 1
            if dep and last_done > now: now = last_done
            while outstanding and outstanding[0][1] <= now: outstanding.popleft()
            while outstanding and inst - outstanding[0][0] >= self.cfg.rob_size:
                now = max(now, outstanding[0][1]); outstanding.popleft()
                while outstanding and outstanding[0][1] <= now: outstanding.popleft()
            inflight = [item for item in outstanding if item[1] > now]
            while len(inflight) >= self.cfg.mshr_demand:
                now = min(done for _, done in inflight)
                inflight = [item for item in outstanding if item[1] > now]
            nominal = before_stall + (gap + 1) / self.cfg.width
            before_l2 = self.counts["l2_accesses"]
            latency = self._demand(pc, addr, bool(wr), now, max(0.0, now - nominal))
            if (self.measure and self.counts["l2_accesses"] != before_l2
                    and self.counts["l2_accesses"] % self.cfg.stats_interval == 0):
                self._record_interval(inst, now)
            last_done = now + latency
            if latency > self.cfg.l1.latency:
                outstanding.append((inst, last_done))
        cycles_end = max([now] + [done for _, done in outstanding])
        measured_cycles = cycles_end - measured_now_start
        measured_inst = inst - int(measured_inst_start)
        if self.measure and self.counts["l2_accesses"] > self._interval_base["l2"]:
            self._record_interval(inst, now)
        unresolved = sum(info["measured"] and not info["resolved"] for info in self.book.values())
        c = self.counts
        return SimulationResult(len(trace) - warmup, measured_inst, measured_cycles,
            measured_inst / measured_cycles if measured_cycles else 0.0,
            c["l2_accesses"], c["l2_misses"],
            c["useful"], c["useless"], c["redundant"], c["dropped"], c["issued"], unresolved,
            c["pollution"], c["pollution_benign"], c["useful_after_expiry"], c["late"],
            self.dram.reads["demand"] + self.dram.reads["prefetch"], self.dram.reads["demand"],
            self.dram.reads["prefetch"], min(1.0, self.dram.total_busy / measured_cycles) if measured_cycles else 0,
            self.prefetcher.storage_bits(), c["returned"])
