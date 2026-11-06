"""Model I bundle, probability averaging, and complete offline output checks."""
import json
from argparse import Namespace

import numpy as np
import pytest
import torch

from fogcast import forecast
from fogcast.dataset import ArrayForecastDataset, forecast_collate
from fogcast.losses import forecast_loss


def test_corrupt_bundle_is_rejected_before_loading_models(tmp_path):
    (tmp_path / "preprocessing.npz").write_bytes(b"corrupt")
    manifest = {"schema": "fogcast-model-i-inference-v1", "members": forecast.MODEL_MEMBERS,
        "files": {"preprocessing.npz": {"bytes": 0, "sha256": "0"*64}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        forecast.InferenceAssets(tmp_path)


def test_bundle_requires_the_three_selected_members(tmp_path):
    manifest = {"schema": "fogcast-model-i-inference-v1", "members": forecast.MODEL_MEMBERS[:2]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="seeds 42/43/44"):
        forecast.InferenceAssets(tmp_path)


def test_bundle_rejects_another_shortcut_architecture(tmp_path):
    manifest = {"schema": "fogcast-model-i-inference-v1", "members": forecast.MODEL_MEMBERS,
        "shortcut_downsampling": "strided_conv"}
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
    assert record["download"] == {"response_bytes": 0, "requests": 0}
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
    geometry = {"fine_shape": (400, 400), "row_indices": np.zeros(16, np.int64),
        "col_indices": columns, "weights": np.full(16, 1/16, np.float32),
        "coverage": np.ones(1, np.float32), "native_area": np.array([4_000_000.]),
        "covered_area": np.array([4_000_000.])}
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


def test_map_preview_registration_cloud_opacity_labels_and_probability_fixity(tmp_path, monkeypatch):
    """Southern clouds stay south; equal0.5 is clear; labels remain above clouds."""
    from datetime import UTC, datetime

    from PIL import Image
    base = Image.new("RGBA", (400,400), (20,30,40,255))
    labels = Image.new("RGBA", (400,400))
    labels.putpixel((250,300), (0,255,0,255))
    monkeypatch.setattr(forecast, "_preview_map", lambda: (base, labels, "Test map attribution"))
    probabilities = np.ones((10,400,400), np.float32)
    probabilities[0,:200] = 0  # Southern half is cloudy in the south-up array.
    probabilities[0,300,160] = .5
    before = probabilities.copy()
    probabilities.setflags(write=False)
    path = tmp_path / "preview.png"
    forecast.preview(path, probabilities, datetime(2026,6,14,14,tzinfo=UTC))
    with Image.open(path) as image:
        assert image.size == (2112,1136)
        assert image.getpixel((24+50,170+100)) == (20,30,40)
        assert image.getpixel((24+50,170+300)) == (145,155,162)
        assert image.getpixel((24+160,170+99)) == (20,30,40)
        assert image.getpixel((24+250,170+300)) == (7,250,8)
        assert image.getpixel((24+416+50,170+300)) == (20,30,40)
        assert image.info["Map attribution"] == "Test map attribution"
        assert "binary" in image.info["Display"]
    np.testing.assert_array_equal(probabilities, before)


def test_packaged_map_matches_forecast_footprint_and_has_transparent_labels():
    base, labels, attribution = forecast._preview_map()
    assert base.size == labels.size == (400,400)
    assert base.getchannel("A").getextrema() == (255,255)
    assert labels.getchannel("A").getextrema() == (0,255)
    assert "Esri" in attribution and "OpenStreetMap" in attribution


def test_preview_rejects_invalid_values_before_loading_maps(tmp_path, monkeypatch):
    from datetime import UTC, datetime
    def forbidden():
        raise AssertionError("invalid forecast reached map rendering")
    monkeypatch.setattr(forecast, "_preview_map", forbidden)
    issue = datetime(2026,6,14,14,tzinfo=UTC)
    with pytest.raises(ValueError, match="probability maps"):
        forecast.preview(tmp_path/"bad.png", np.zeros((1,400,400)), issue)
    for value in (np.nan, -.01, 1.01):
        probabilities = np.zeros((10,400,400), np.float32)
        probabilities[0,0,0] = value
        with pytest.raises(ValueError, match="probability maps"):
            forecast.preview(tmp_path/"bad.png", probabilities, issue)
