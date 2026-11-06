"""Bundle integrity and end-to-end forecast output contracts without network I/O."""
import json
from argparse import Namespace

import numpy as np
import pytest


def test_asset_corruption_is_rejected_before_loading_weights(tmp_path):
    from fogcast.forecast import InferenceAssets
    (tmp_path / "model.pt").write_bytes(b"different")
    manifest = {"schema": "fogcast-baseline-inference-v1", "files": {"model.pt":{"bytes": 9, "sha256": "0" * 64}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        InferenceAssets(tmp_path)


@pytest.mark.parametrize("legacy_metadata", [False, True])
def test_fixed_average_pooling_bundle_roundtrip(tmp_path, legacy_metadata):
    import torch

    from fogcast.forecast import InferenceAssets, sha256
    from fogcast.model import NBMResidualUNet

    shape = (400, 400)
    arrays = {"reference_coefficients": np.zeros((*shape, 11), np.float32),
                  "offsets": np.zeros(160000), "eligible_offsets": np.ones(160000, bool),
                  "classifier_coefficients": np.zeros(4), "classifier_k": np.array(1.),
                  "land_fraction": np.zeros(shape), "elevation_normalized": np.zeros(shape),
                  "latitude": np.full(shape, 37.7), "longitude": np.full(shape, -122.4)}
    np.savez_compressed(tmp_path / "preprocessing.npz", **arrays)
    state = NBMResidualUNet().state_dict()
    torch.save(state, tmp_path / "model.pt")
    manifest = {"schema": "fogcast-baseline-inference-v1", "files": {
        name: {"bytes": (tmp_path / name).stat().st_size, "sha256": sha256(tmp_path / name)}
        for name in ("model.pt", "preprocessing.npz")}}
    if legacy_metadata:
        manifest["shortcut_downsampling"] = "avg_pool"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    previous = torch.get_num_threads()
    try:
        assets = InferenceAssets(tmp_path)
        assert all(torch.equal(value, assets.model.state_dict()[name]) for name, value in state.items())
    finally:
        torch.set_num_threads(previous)


def test_bundle_with_other_shortcut_architecture_is_rejected(tmp_path):
    from fogcast import InferenceAssets
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": "fogcast-baseline-inference-v1", "shortcut_downsampling": "strided_conv"}))
    with pytest.raises(ValueError, match="average-pooling"):
        InferenceAssets(tmp_path)


def test_offline_replay_is_network_free_and_writes_ten_georeferenced_probabilities(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    from fogcast import forecast
    shape = (400, 400)
    sample = {"issue_time": np.array("2026-06-01T20:00:00Z"), "land_fraction": np.zeros(shape, np.float32),
                  "elevation_normalized": np.zeros(shape, np.float32), "current_solar_cosine": np.full(shape, .7, np.float32),
                  "target_solar_cosine": np.zeros((10, *shape), np.float32), "nbm_cloud_cover": np.full((11, 100, 100), .2, np.float32)}
    source = tmp_path / "source.npz"
    np.savez_compressed(source, **sample)
    class FixtureAssets:
        def __init__(self, *_args, **_kwargs):
            self.arrays = {"latitude": np.full(shape, 37.7), "longitude": np.full(shape, -122.4)}
            self.manifest = {"schema": "fixture-only", "files": {}}
        def predict(self, item): return np.full((10, *shape), .25, np.float32)
    def no_network(*_args, **_kwargs): raise AssertionError("offline replay used the network")
    monkeypatch.setattr(forecast, "InferenceAssets", FixtureAssets)
    monkeypatch.setattr(forecast.DownloadBudget, "request", no_network)
    args = Namespace(output=str(tmp_path / "result"), cache_dir=str(tmp_path / "cache"), assets="fixture",
                     timeout_seconds=30, maximum_download_mib=1, threads=1, offline_inputs=str(source), as_of=None)
    record = forecast.run(args)
    assert record["provenance"]["input_mode"] == "offline_replay"
    assert record["grid"]["row_order"] == "south_up" and len(record["valid_times"]) == 10
    with np.load(tmp_path / "result/forecast.npz", allow_pickle=False) as result:
        assert result["clear_probability"].shape == (10, 400, 400)
        assert result["valid_times"][-1] == "2026-06-02T06:00:00+00:00"
        assert np.all(np.diff(result["x_m"]) == 500)
    assert (tmp_path / "result/forecast.png").is_file()
    with pytest.raises(FileExistsError): forecast.run(args)
