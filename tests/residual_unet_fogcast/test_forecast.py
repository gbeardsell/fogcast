"""Model I bundle, probability averaging, and complete offline output checks."""
from argparse import Namespace
import json

import numpy as np
import pytest
import torch

from fogcast.residual_unet_fogcast import forecast
from fogcast.residual_unet_fogcast.dataset import ArrayForecastDataset
from fogcast.residual_unet_fogcast.dataset import forecast_collate
from fogcast.residual_unet_fogcast.losses import forecast_loss


def test_corrupt_bundle_is_rejected_before_loading_models(tmp_path):
    (tmp_path / "preprocessing.npz").write_bytes(b"corrupt")
    manifest = dict(schema="fogcast-model-i-inference-v1", members=forecast.MODEL_MEMBERS,
        files={"preprocessing.npz": dict(bytes=0, sha256="0"*64)})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        forecast.InferenceAssets(tmp_path)


def test_bundle_requires_the_three_selected_members(tmp_path):
    manifest = dict(schema="fogcast-model-i-inference-v1", members=forecast.MODEL_MEMBERS[:2])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="seeds 42/43/44"):
        forecast.InferenceAssets(tmp_path)


def test_bundle_rejects_another_shortcut_architecture(tmp_path):
    manifest = dict(schema="fogcast-model-i-inference-v1", members=forecast.MODEL_MEMBERS,
        shortcut_downsampling="strided_conv")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="average-pooling"):
        forecast.InferenceAssets(tmp_path)


def test_three_member_mean_and_offline_outputs(synthetic_bundle, historical_inputs, tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("offline inference contacted the network")
    monkeypatch.setattr(forecast.DownloadBudget, "request", no_network)
    args = Namespace(assets=str(synthetic_bundle), offline_inputs=str(historical_inputs),
        output=str(tmp_path / "result"), cache_dir=str(tmp_path / "cache"),
        timeout_seconds=90, maximum_download_mib=1, threads=1, as_of=None)
    record = forecast.run(args)
    expected = torch.tensor([-3., 1., 2.]).sigmoid().double().mean().float().item()
    assert abs(expected - .5) > .01  # Averaging logits would incorrectly give .5.
    with np.load(tmp_path / "result/forecast.npz", allow_pickle=False) as result:
        probabilities = result["clear_probability"]
        assert probabilities.shape == (10, 400, 400)
        np.testing.assert_array_equal(probabilities, np.full_like(probabilities, expected))
        assert result["latitude"].shape == result["longitude"].shape == (400, 400)
        assert result["valid_times"][0] == "2026-06-14T15:00:00+00:00"
        assert result["valid_times"][-1] == "2026-06-15T00:00:00+00:00"
    assert record["download"] == dict(response_bytes=0, requests=0)
    assert record["provenance"]["input_mode"] == "offline_replay"
    for name, spec in record["files"].items():
        assert forecast.sha256(tmp_path / "result" / name) == spec["sha256"]
    assert (tmp_path / "result/forecast.png").is_file()
    with pytest.raises(FileExistsError):
        forecast.run(args)


def test_model_i_labels_and_geometry_are_central(historical_inputs):
    sample, _ = forecast.load_prepared(historical_inputs)
    # A sunny halo must not change supervision for the entirely dark target area.
    sample["target_solar_cosine"][:] = 1
    sample["target_solar_cosine"][:, 100:500, 100:500] = 0
    columns = np.array([400*r+c for r in range(4) for c in range(4)], np.int64)
    geometry = dict(fine_shape=(400, 400), row_indices=np.zeros(16, np.int64),
        col_indices=columns, weights=np.full(16, 1/16, np.float32),
        coverage=np.ones(1, np.float32), native_area=np.array([4_000_000.]),
        covered_area=np.array([4_000_000.]))
    sample["target_coarse"] = np.zeros((10, 1), np.float32)
    sample["valid_coarse"] = np.ones((10, 1), bool)
    item = ArrayForecastDataset([sample], native_geometry=geometry)[0]
    assert item["x_fine"].shape == (5, 600, 600)
    assert item["x_coarse"].shape == (25, 150, 150)
    assert item["target_fine"].shape == item["valid_fine"].shape == (10, 400, 400)
    assert item["coarse_supervision"].all()
    assert item["coarse_supervision"].dtype == torch.bool
    logits = torch.zeros((1, 10, 400, 400), requires_grad=True)
    result = forecast_loss(logits, forecast_collate([item]))
    torch.testing.assert_close(result.loss, torch.tensor(.25 * np.log(2), dtype=torch.float32))
    assert result.statistics.coarse_count.tolist() == [1] * 10
    result.loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_expired_preparation_stops_before_inference(historical_inputs, tmp_path, monkeypatch):
    class Assets:
        def __init__(self, *args, **kwargs): pass
        def predict(self, *args): raise AssertionError("inference ran after deadline")
    class Expired:
        def __init__(self, *args, **kwargs): pass
        def remaining(self): raise TimeoutError("deadline")
    monkeypatch.setattr(forecast, "InferenceAssets", Assets)
    monkeypatch.setattr(forecast, "DownloadBudget", Expired)
    args = Namespace(assets="fixture", offline_inputs=str(historical_inputs),
        output=str(tmp_path / "result"), cache_dir=str(tmp_path / "cache"),
        timeout_seconds=1, maximum_download_mib=1, threads=1, as_of=None)
    with pytest.raises(TimeoutError):
        forecast.run(args)
