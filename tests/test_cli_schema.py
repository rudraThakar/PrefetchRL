import json

from pfsim.cli import main
from pfsim.experiments.runner import RUN_FIELDS
from pfsim.trace import generate, save_trace


def test_run_json_contains_runs_schema(tmp_path, capsys):
    trace_path = tmp_path / "trace.npz"; save_trace(generate("seq_stream", 0, 50), trace_path)
    main(["run", "--trace", str(trace_path), "--prefetcher", "none"])
    output = capsys.readouterr().out
    payload = json.loads(output[output.index("{"):])
    assert set(RUN_FIELDS) <= set(payload)
