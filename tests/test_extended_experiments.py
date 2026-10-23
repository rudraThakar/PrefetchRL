import csv
from dataclasses import replace

from pfsim.config import SimConfig
from pfsim.experiments.extended import run_e2, run_e3


def test_e2_small_sweep_writes_plot(tmp_path):
    cfg = replace(SimConfig(), warmup_fraction=0, stats_interval=50)
    runs, summary = run_e2(cfg, tmp_path, records=100, workloads=("seq_stream",),
                           bandwidths=(12.8,), prefetchers=("none", "next_line"), force=True)
    assert runs.exists() and summary.exists()
    assert (tmp_path / "E2" / "fig_speedup_bandwidth.png").stat().st_size > 0


def test_e3_small_sweep_writes_variants(tmp_path):
    cfg = replace(SimConfig(), warmup_fraction=0, stats_interval=50)
    runs, _ = run_e3(cfg, tmp_path, records=80, workloads=("seq_stream",),
                     bandwidths=(12.8,), variants=("full", "no_reach"), agent_seeds=(0,), force=True)
    with runs.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["variant"] for row in rows} == {"full", "no_reach"}
    assert (tmp_path / "E3" / "fig_ablation_speedup.png").exists()
