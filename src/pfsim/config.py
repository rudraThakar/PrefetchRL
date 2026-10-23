from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CacheConfig:
    size_kib: int
    ways: int
    latency: float


@dataclass
class PollutionConfig:
    counterfactual: bool = True


@dataclass
class SimConfig:
    freq_ghz: float = 4.0
    width: int = 4
    rob_size: int = 256
    mshr_demand: int = 32
    pf_mshr: int = 16
    l1: CacheConfig = field(default_factory=lambda: CacheConfig(32, 8, 4.0))
    l2: CacheConfig = field(default_factory=lambda: CacheConfig(256, 8, 14.0))
    llc: CacheConfig = field(default_factory=lambda: CacheConfig(2048, 16, 34.0))
    dram_latency: float = 200.0
    peak_bw_gbps: float = 12.8
    pf_backlog_factor: float = 32.0
    bw_window_cycles: int = 4096
    bw_bucket_cycles: int = 64
    pf_same_page_only: bool = True
    pf_issue_latency: float = 0.0
    expiry_accesses: int = 1024
    pollution_window: int = 4096
    victim_filter_size: int = 4096
    warmup_fraction: float = 0.1
    stats_interval: int = 50_000
    acc_ema_weight: float = 1 / 64
    pol_ema_weight: float = 1 / 64
    pollution: PollutionConfig = field(default_factory=PollutionConfig)

    @property
    def transfer_cycles(self) -> int:
        import math
        return math.ceil(64.0 / self.peak_bw_gbps * self.freq_ghz)

    @property
    def pf_backlog_cycles(self) -> float:
        return self.pf_backlog_factor * self.transfer_cycles

    def with_bandwidth(self, gbps: float) -> "SimConfig":
        data = asdict(self)
        data["peak_bw_gbps"] = gbps
        return sim_config_from_dict(data)

    def digest(self) -> str:
        encoded = yaml.safe_dump(asdict(self), sort_keys=True).encode()
        return sha256(encoded).hexdigest()[:16]


def sim_config_from_dict(data: dict[str, Any]) -> SimConfig:
    values = dict(data)
    for name in ("l1", "l2", "llc"):
        if name in values and not isinstance(values[name], CacheConfig):
            values[name] = CacheConfig(**values[name])
    if "pollution" in values and not isinstance(values["pollution"], PollutionConfig):
        values["pollution"] = PollutionConfig(**values["pollution"])
    return SimConfig(**values)


def load_config(path: str | Path | None = None) -> SimConfig:
    if path is None:
        return SimConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        return sim_config_from_dict(yaml.safe_load(handle) or {})


def load_agent_config(path: str | Path | None = None):
    from pfsim.rl.agent import AgentConfig
    from pfsim.rl.reward import RewardConfig
    if path is None:
        return AgentConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    values = dict(values)
    if "reaches" in values and "low_accuracy_max_reach" not in values:
        values["low_accuracy_max_reach"] = max(values["reaches"])
    if "reward" in values:
        values["reward"] = RewardConfig(**values["reward"])
    for key in ("reaches", "patterns", "feature_drop", "reset_boundaries"):
        if key in values: values[key] = tuple(values[key])
    return AgentConfig(**values)


def overlay_agent_config(base, path: str | Path):
    """Apply a partial YAML variant to a complete agent configuration."""
    from dataclasses import asdict
    from pfsim.rl.agent import AgentConfig
    from pfsim.rl.reward import RewardConfig
    with Path(path).open("r", encoding="utf-8") as handle:
        overlay = yaml.safe_load(handle) or {}
    values = asdict(base)
    reward = values.pop("reward")
    reward.update(overlay.pop("reward", {}))
    values.update(overlay)
    if "reaches" in overlay and "low_accuracy_max_reach" not in overlay:
        values["low_accuracy_max_reach"] = max(overlay["reaches"])
    for key in ("reaches", "patterns", "feature_drop", "reset_boundaries"):
        if key in values: values[key] = tuple(values[key])
    values["reward"] = RewardConfig(**reward)
    return AgentConfig(**values)
