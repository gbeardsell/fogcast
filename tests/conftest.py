"""Synthetic fixtures for tests; the runnable examples use trained assets."""
import json

import numpy as np
import pytest
import torch
from fogcast.forecast import MODEL_MEMBERS, sha256
from fogcast.model import ResidualUNet


@pytest.fixture(scope="session")
def synthetic_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("synthetic-model-i")
    shape = (600, 600)
    arrays = dict(reference_coefficients=np.zeros((*shape, 11), np.float64),
        offsets=np.zeros(360000), eligible_offsets=np.ones(360000, bool),
        classifier_coefficients=np.zeros(4), classifier_k=np.array(1.),
        land_fraction=np.zeros(shape, np.float32), elevation_normalized=np.zeros(shape, np.float32),
        latitude=np.full(shape, 37.7), longitude=np.full(shape, -122.4))
    np.savez_compressed(directory / "preprocessing.npz", **arrays)
    for member, bias in zip(MODEL_MEMBERS, [-3., 1., 2.]):
        model = ResidualUNet()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.head.bias.fill_(bias)
        torch.save(model.state_dict(), directory / member["file"])
    names = ["preprocessing.npz", *(member["file"] for member in MODEL_MEMBERS)]
    manifest = dict(schema="fogcast-model-i-inference-v1", members=MODEL_MEMBERS,
        fixture_only=True, files={name: dict(bytes=(directory/name).stat().st_size,
            sha256=sha256(directory/name)) for name in names})
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


@pytest.fixture(scope="session")
def historical_inputs(tmp_path_factory):
    path = tmp_path_factory.mktemp("synthetic-inputs") / "inputs.npz"
    shape = (600, 600)
    np.savez_compressed(path, issue_time=np.array("2026-06-14T14:00:00Z"),
        land_fraction=np.zeros(shape, np.float32), elevation_normalized=np.zeros(shape, np.float32),
        current_solar_cosine=np.full(shape, .7, np.float32),
        target_solar_cosine=np.zeros((10, *shape), np.float32),
        nbm_cloud_cover=np.full((11, 150, 150), .2, np.float32))
    return path
