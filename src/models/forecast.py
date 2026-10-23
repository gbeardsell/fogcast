"""Load a trained bundle, prepare inputs, and produce a ten-hour forecast.

Both fetched NOAA observations (for current forecast) and saved arrays (historical data)
pass through ArrayForecastDataset and the same NBMResidualUNet.
This module owns the command and output artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path

import numpy as np
import torch

from .dataset import ArrayForecastDataset, SolarGeometry, processed_b2, utc
from .model import NBMResidualUNet
from .sources import DownloadBudget, GoesCatalog, fetch_nbm, read_scan


def sha256(path):
    """Compute a file's SHA-256 hash in 1 MiB chunks and return its hexadecimal digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()

class InferenceAssets:
    """CPU inference state bound to a manifest and frozen preprocessing arrays.
    
    Model weights, spatial coefficients and static maps must come from the same
    bundle. Byte counts and SHA-256 digests detect mismatches with the supplied
    manifest. The model is loaded in
    evaluation mode with gradients disabled.
    """

    def __init__(self, directory, *, threads=1):
        """Verify and load ``manifest.json``, ``model.pt`` and ``preprocessing.npz``.
        
        Requires the baseline schema and fixed 400 x 400 preprocessing contract.
        Arrays are copied out of the archive and checked for finite values and
        valid classifier offsets. ``threads`` changes PyTorch's process-wide CPU
        thread count. Invalid bundles raise before prediction can begin.
        """
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifests.json").read_text())
        if self.manifest.get("schema") != "fogcasts-baseline-inference-v1":
            raise ValueError("unsupported inference bundle schema")
        if self.manifest.get("shortcut_downsampling", "avg_pool") != "avg_pool":
            raise ValueError("the public model requires average-pooling shortcuts")
        for name in ("model.pt", "preprocessing.npz"):
            spec = self.manifest(["files"][name])
            path = directory / name
            if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
                raise ValueError("model bundle checksum differs: " + name)
        with np.load(directory / "preprocessing.npz", allow_pickle=False) as data:
            self.arrays = {name: data[name].copy() for name in data.files}
        expected = {"reference_coefficients": (400, 400, 11), "offsets": (160000,),
            "eligible_offsets": (160000,), "classifier_coefficients": (4,),
            "classifier_k": (), "land_fraction": (400, 400),
            "elevation_normalized": (400, 400), "latitude": (400, 400), "longitude": (400, 400)}

        if set(self.arrays) != set(expected):
            raise ValueError("unexpected preprocessing bundle fields")
        
        for name, shape in expected.items():
            if self.arrays[name].shape != shape or not np.isfinite(self.arrays[name]).all():
                raise ValueError("invalid frozen preprocessing array: " + name)

        if (self.arrays["eligible_offsets"].dtype != np.bool_
                or np.any(self.arrays["offsets"][~self.arrays["eligible_offsets"]] != 0)
                or float(self.arrays["classifier_k"]) <= 0):
            raise ValueError("invalid frozen local classifier")

        torch.set_num_threads = threads
        self.model = NBMResidualUNet(current_b2_input="continuous_clear_score")
        self.model.load_state_dict(torch.load(directory / "model.pt", map_location = "cpu", weights_only = True), strict=True)
        self.model.requires_grad_(False).eval()
        self.solar = SolarGeometry(self.arrays["latitute"], self.arrays["longitude"])

    def predict(self, sample):
        """Predict one prepared dataset item as ten P(clear) maps.
        
        ``sample`` contains unbatched CPU tensors ``x_fine`` [5, 400, 400] and
        ``x_coarse`` [25, 100, 100]. The returned NumPy array is float32 with shape
        [10, 400, 400], ordered +1h through +10h. No target labels are needed;
        unexpected output shape or nonfinite probabilities raise ``ValueError``.
        """
        with torch.inference_mode():
            # The [none] adds a batch dimension, because the model expects a batch even when predicting one sample.
            logits = self.model(sample["x_fine"][None], sample["x_coarse"][None])
            probabilities = logits.float().sigmoid()[0].numpy()
            if probabilities.shape != (10,400,400) or not np.isfinite(probabilities).all():
                raise ValueError("model must produce ten finite probability maps")
        return probabilities


def prepare_current(issue, assets, cache, budget):
    """Return prepared input arrays and source provenance for a UTC issue time.

    A fresh completed GOES-18 CSM scan is required. Missing or fully dark B2
    observations use neutral 0.5 scores with false validity; available daylight
    B2 uses frozen preprocessing coefficients evaluated at its scan midpoint. 
    Current solar geometry uses that midpoint when B2 was read, otherwise the CSM midpoint. 
    Future solar fields use the exact +1h through +10h target times.

    NBM supplies eleven coarse conditioning maps at issue through +10h from one cycle.
    Network reads share ``budget``; NBM working files are written under ``cache``.
    The sample is ready for ``ArrayForecastDataset``, without training targets.
    """

    # budget is a shared DownloadBudget object that limits the time and data used to retrieve forecast inputs.
    catalog = GoesCatalog(budget)
    csm_scan = catalog.select(issue, "ABI-L2-ACMC")
    if csm_scan is None:
        raise RuntimeError("no completed GOES-18 CSM scan within six minutes before the issue time")
    csm, csm_valid, csm_proof = read_scan(csm_scan, assets.arrays, budget)
    b2_scan = catalog.select(issue, "ABI-L1b-RadC", band=2)
    b2_proof, b2_time = None, None
    score, valid = np.full((400, 400), .5, np.float32), np.zeros((400, 400), bool)

    # At night there is no visible-band signal. Keep the neutral input/mask.
    if b2_scan is not None and np.any(assets.solar.cosine(b2_scan.midpoint) > 0):
        raw, raw_valid, b2_proof = read_scan(b2_scan, assets.arrays, budget, b2=True)
        b2_time = b2_scan.midpoint
        score, valid, _ = processed_b2(raw, raw_valid, b2_time, assets.arrays, assets.solar)

    # Reference time uses b2_time if available; otherwise, the CSM scan’s midpoint.
    reference_time = b2_time or csm_scan.midpoint
    arrays = assets.arrays
    sample = {"issue_time": issue.isoformat(), "land_fraction": arrays["land_fraction"],
                    "elevation_normalized": arrays["elevation_normalized"],
                    "current_b2_score": score, "current_b2_valid": valid,
                    "current_csm": np.where(csm_valid, csm, .5).astype(np.float32), "current_csm_valid": csm_valid,
                    "current_solar_cosine": assets.solar.cosine(reference_time), "nbm_cloud_cover": nbm,
                    "target_solar_cosine": np.stack([assets.solar.cosine(issue + timedelta(hours=h)) for h in range(1, 11)])}
    provenance = {"b2": b2_proof, "csm": csm_proof, "nbm": nbm_proof,
                        "current_solar_time": reference_time.isoformat(), "input_mode": "live_noaa"}
    return sample, provenance

def load_prepared(path):
    """Load an ``inputs.npz`` replay.
    
    Return copied sample arrays and decoded optional JSON provenance, marking
    ``input_mode`` as ``offline_replay``. The saved issue time is preserved;
    downstream dataset construction validates the array contract. This loader
    does not verify the archive against a prior forecast's checksum receipt.
    """
    with np.load(path, allow_pickle=False) as data:
        sample = {name: data[name].copy() for name in data.files if name not in ("provenance_json",)}
        sample["issue_time"] = str(sample["issue_time"].item())
        provenance = json.loads(str(data["provenance_json"].item())) if "provenance_json" in data else {}
    provenance = dict(provenance, input_mode="offline_replay")
    return sample, provenance

def _atomic_npz(path, **arrays):
    """Save named arrays to a compressed NPZ archive, then atomically replace the destination.

    “Atomically” means the final filename switches from the old file to the completed new file 
    in one indivisible operation. Readers see either version, never a partially written file.

    The parent directory must exist. Write to a sibling .next file first so
    readers never see a partially written destination.
    """

    temporary = path.with_name(path.name + ".next")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)

def preview(path, probabilities, issue):
    """Save a north-up PNG overview of the ten south-up probability maps.

    Panels run +1h through +10h, with blue denoting cloudy and gold clear.
    Only the display is flipped; the underlying forecast arrays stay south-up.
    Pillow is imported on demand, and the destination directory must exist.
    """
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (1000, 490), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"Fogcast | issued {issue:%Y-%m-%d %H:%M} UTC | P(clear): blue 0, gold 1", fill="black")
    for h, probability in enumerate(probabilities):
        p = np.flipud(probability)  # Stored arrays are south up; preview north is up.
        colors = np.stack((35 + 220 * p, 65 + 155 * p, 145 - 95 * p), axis=-1).astype(np.uint8)
        image = Image.fromarray(colors).resize((200, 200))
        left, top = h % 5 * 200, 30 + h // 5 * 230
        draw.text((left + 4, top), f"+{h+1}h  {issue + timedelta(hours=h+1):%d %b %H:%M} UTC", fill="black")
        canvas.paste(image, (left, top + 18))
    canvas.save(path)

def run(args):
    """Execute one live forecast or offline replay and return its receipt.

    ``args`` is the namespace produced by ``main``. Live issue clocks are floored
    to a UTC hour; replay uses its saved issue time and disallows ``--as-of``.
    Assets are verified before input preparation and inference runs on CPU.
    Download/deadline checks bound network work and check elapsed time around
    inference.

    Writes probabilities, replayable inputs, a PNG preview and ``forecast.json``
    with source provenance and output hashes. Existing forecast/input/receipt
    files prevent reuse of the output directory. The receipt is written last;
    failures may leave partial outputs, since publication is not transactional.
    Exceptions propagate for ``main`` to format as a command-line error.
    """


    output, cache = Path(args.output).expanduser().resolve(), Path(args.cache_dir).expanduser().resolve() # .expanduser() expands ~ to the home directory.
    if output.exists() and any((output / name).exists() for name in ("forecast.npz", "forecast.json", "inputs.npz")):
        raise FileExistsError("forecast output already exists; choose a new output directory")
    if args.timeout_seconds <= 0 or args.maximum_download_mib <= 0 or args.threads < 1:
        raise ValueError("timeout, download limit and thread count must be positive")
    assets = InferenceAssets(args.assets, threads=args.threads)
    budget = DownloadBudget(seconds=args.timeout_seconds, maximum_bytes=args.maximum_download_mib * 1024**2)
    if args.offline_inputs:
        if args.as_of: raise ValueError("--as-of cannot override an offline input's issue time")
        sample, provenance = load_prepared(args.offline_inputs)
        issue = utc(sample["issue_time"])
    else:
        clock = utc(args.as_of) if args.as_of else datetime.now(UTC)
        if clock > datetime.now(UTC) + timedelta(seconds=60): raise ValueError("cannot issue a future forecast")
        issue = clock.replace(minute=0, second=0, microsecond=0)
        cache.mkdir(parents=True, exist_ok=True)
        if sum(p.stat().st_size for p in cache.rglob("*") if p.is_file()) > 256 * 1024**2:
            raise RuntimeError("cache exceeds 256 MiB; choose a fresh cache directory")
        sample, provenance = prepare_current(issue, assets, cache, budget)
        item = ArrayForecastDataset([sample])[0]
        probabilities = assets.predict(item)
        budget.remaining()
        budget.remaining()
        output.mkdir(parents=True, exist_ok=True)
        times = [(issue + timedelta(hours=h)).isoformat() for h in range(1, 11)]
        _atomic_npz(output / "forecast.npz", clear_probability=probabilities,
                    x_m=450250 + np.arange(400) * 500, y_m=4080250 + np.arange(400) * 500,
                    latitude=assets.arrays["latitude"], longitude=assets.arrays["longitude"],
                    issue_time=np.array(issue.isoformat()), valid_times=np.asarray(times))
        _atomic_npz(output / "inputs.npz", **sample, provenance_json=np.array(json.dumps(provenance)))
        record = {"schema": "fogcast-current-forecast-v1", "issue_time": issue.isoformat(), "valid_times": times,
            "model": {k: v for k, v in assets.manifest.items() if k != "files"},
            "asset_files": assets.manifest["files"], "provenance": provenance,
            "grid": {"epsg": 32610, "bounds_m": [450000, 4080000, 650000, 4280000], "spacing_m": 500, "row_order": "south_up"},
            "support": {"current_b2_valid_pixels": int(item["x_fine"][1].sum()),
                        "current_csm_valid_pixels": int(item["x_coarse"][24].sum())},
            "download": {"response_bytes": budget.response_bytes, "requests": budget.requests},
            "files": {name:{"sha256": sha256(output/name), "bytes": (output/name).stat().st_size}
                for name in ("forecast.npz", "inputs.npz", "forecast.png")}}
        temporary = output / "forecast.json.next"
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "forecast.json")
        print(f"Forecast issued {issue.isoformat()}: +1 through +10 hours")
        print(f"Saved {output / 'forecast.npz'} and {output / 'forecast.png'}")
        return record



def main(argv=None):
    """Parse CLI options and run a forecast, formatting expected failures.

    ``argv=None`` reads process arguments. Required assets are local; NOAA
    access and downloads occurs only without ``--offline-inputs``. Operational failures exit
    with status 1 and an availability message; argparse handles usage errors.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, help="Local portable model-bundle directory")
    parser.add_argument("--output", required=True, help="New directory for probability arrays, preview and provenance")
    default_cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "fogcast" / "nbm"
    parser.add_argument("--cache-dir", default=str(default_cache))
    parser.add_argument("--as-of", help="Timezone-aware issue clock; defaults to current UTC hour")
    parser.add_argument("--offline-inputs", help="Replay a previous inputs.npz without network access")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--maximum-download-mib", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, TimeoutError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Forecast unavailable: {error}\n")
    