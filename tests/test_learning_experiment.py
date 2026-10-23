from dataclasses import replace

from pfsim.config import SimConfig
from pfsim.experiments.learning import recovery_times, run_e4


def test_recovery_time_calculation():
    rows = [{"l2_idx": index, "window_ipc": ipc} for index, ipc in
            ((400, .5), (410, .7), (420, .95), (450, 1.0))]
    result = recovery_times(rows, [100, 200, 300, 400], 100)
    assert result[0]["recovery_accesses"] == 20


def test_e4_small_run_writes_learning_figure(tmp_path):
    cfg = replace(SimConfig(), warmup_fraction=0, stats_interval=50)
    runs, recovery = run_e4(cfg, tmp_path, records=200, workloads=("phase_mix",),
                            agent_seeds=(0,), force=True)
    assert runs.exists() and recovery.exists()
    assert (tmp_path / "E4" / "fig_learning_phase_mix_seed0.png").exists()
