import json
import os
import subprocess
import sys
from pathlib import Path


def test_missing_offline_model_is_a_failed_run(tmp_path):
    output = tmp_path / "failed-run.json"
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        HF_HOME=str(tmp_path / "empty-home"),
        HF_HUB_CACHE=str(tmp_path / "empty-cache"),
    )
    run = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--model", "qwen", "--output", str(output)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert run.returncode != 0
    record = json.loads(output.read_text())
    assert record["errors"] == 5
    assert record["hits"] == 0
