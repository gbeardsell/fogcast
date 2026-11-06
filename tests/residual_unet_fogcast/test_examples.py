"""Run both public example scripts against the local source without network I/O."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("example", ["historical", "current"])
def test_example_script(example, synthetic_bundle, historical_inputs, tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = root / "examples" / (example + ".py")
    output = tmp_path / "result"
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        'import socket\ndef blocked(*args, **kwargs):\n    raise AssertionError("network forbidden")\n'
        'socket.socket.connect = blocked\nsocket.socket.connect_ex = blocked\n')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(guard), str(root / "src")]),
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, str(script), "--assets", str(synthetic_bundle),  # noqa: PLW1510
        "--offline-inputs", str(historical_inputs), "--output", str(output)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads((output / "forecast.json").read_text())
    assert record["issue_time"] == "2026-06-14T14:00:00+00:00"
    assert record["download"] == {"response_bytes": 0, "requests": 0}
    with np.load(output / "forecast.npz", allow_pickle=False) as arrays:
        assert arrays["clear_probability"].shape == (10, 400, 400)
        # The synthetic members have constant logits -3, 1, and 2.
        logits = np.array([-3., 1., 2.])
        expected_probability = (1 / (1 + np.exp(-logits))).mean()
        np.testing.assert_allclose(arrays["clear_probability"], expected_probability, rtol=1e-6)
    assert (output / "forecast.png").is_file()
