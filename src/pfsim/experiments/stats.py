from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def geometric_mean(values: Iterable[float]) -> float:
    values = tuple(values)
    return math.exp(sum(math.log(max(float(value), 1e-12)) for value in values) / len(values)) if values else 0.0


def bootstrap_gmean_ci(by_workload: dict[str, list[float]], *, samples: int = 2000,
                       seed: int = 0) -> tuple[float, float]:
    """Resample seeds within each workload, then recompute the workload geometric mean."""
    if not by_workload: return 0.0, 0.0
    rng = np.random.default_rng(seed); estimates = np.empty(samples, dtype=float)
    groups = [np.asarray(values, dtype=float) for _, values in sorted(by_workload.items())]
    for index in range(samples):
        means = [float(rng.choice(group, size=len(group), replace=True).mean()) for group in groups]
        estimates[index] = geometric_mean(means)
    low, high = np.percentile(estimates, (2.5, 97.5))
    return float(low), float(high)


def paired_outcomes(agent: dict[tuple[str, int], float], baseline: dict[tuple[str, int], float],
                    tie_fraction: float = .01) -> dict[str, int]:
    counts = {"win": 0, "tie": 0, "loss": 0}
    for key in sorted(set(agent) & set(baseline)):
        ratio = agent[key] / baseline[key] if baseline[key] else 0.0
        counts["tie" if abs(ratio - 1) <= tie_fraction else "win" if ratio > 1 else "loss"] += 1
    return counts
