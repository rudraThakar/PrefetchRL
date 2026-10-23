from __future__ import annotations

from typing import Any

from .simulator import SimulationResult


def comparative_metrics(run: SimulationResult, baseline: SimulationResult) -> dict[str, Any]:
    """Return the section 3.9 metrics that require a no-prefetch baseline."""
    speedup = run.ipc / baseline.ipc if baseline.ipc else 0.0
    coverage = ((baseline.l2_misses - run.l2_misses) / baseline.l2_misses
                if baseline.l2_misses else 0.0)
    overprediction = ((run.dram_reads - baseline.dram_reads) / baseline.dram_reads
                      if baseline.dram_reads else 0.0)
    return {**run.to_dict(), "speedup": speedup, "coverage": coverage,
            "overprediction": overprediction}

