from calendar import isleap
from collections.abc import (
    Mapping,  # Key-based container, e.g. dict; hint: Mapping[str, Any]
    Sequence,  # Ordered, indexable container, e.g. list or tuple; hint: Sequence[int]
)
from datetime import UTC, datetime
from functools import lru_cache
from operator import index as integer_index
from typing import Any

import numpy as np
import torch
from scipy.special import expit
from torch.utils.data import Dataset

HORIZONS = 10
MINIMUM_COVERAGE = 0.99
ZSA_MAX = 82
NIGHT_SOLAR_THRESHOLD = np.float32(np.cos(np.deg2rad(ZSA_MAX)))
FINE_CHANNELS = (
    "current_b2_score", "current_b2_valid", "land_fraction",
    "elevation_normalized", "current_solar_cosine",
)
CONDITIONING_CHANNELS = (
    *(f"nbm_plus_{h:02d}h" for h in range(11)),
    *(f"solar_plus_{h:02d}h" for h in range(1, 11)),
    "annual_sine", "annual_cosine", "current_csm", "current_csm_valid",
)

def utc(value: str | datetime) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    return value.astimezone(UTC).replace(second=0, microsecond=0)

class SolarGeometry:
    """NOAA fractional-year equations, with a bounded cache of fine-grid cosines."""

    def __init__(self, latitude, longitude):
        self.latitude = np.asarray(latitude, np.float64)
        self.longitude = np.asarray(longitude, np.float64)
        # Cache computations per instance, after timestamp normalization.
        self._solar_geometry = lru_cache(maxsize=64)(self._solar_geometry)

    def solar_geometry(self, timestamp):
            """Normalize the timestamp before looking up cached solar fields."""
            return self._solar_geometry(utc(timestamp))

    def _solar_geometry(self, timestamp):
        """Compute solar zenith cosine using NOAA's approximate solar position equations.
        Also computes solar hour.
        Reference: https://gml.noaa.gov/grad/solcalc/solareqns.PDF"""
        
        time = utc(timestamp)
        hour = time.hour + time.minute / 60
        days_in_year = 366 if isleap(time.year) else 365
        gamma = 2 * np.pi / days_in_year * (time.timetuple().tm_yday - 1 + (hour - 12)/24)
        equation = 229.18 * (0.000075 + 0.001868 * np.cos(gamma)
            - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2 * gamma)
            - 0.040849 * np.sin(2 * gamma))
        declination = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
        solar_minutes = (hour * 60 + equation + 4 * self.longitude) % 1440
        solar_hour = solar_minutes / 60
        hour_angle = np.deg2rad(solar_minutes / 4 - 180)
        latitude = np.deg2rad(self.latitude)
        cosine = (
            np.sin(latitude) * np.sin(declination)
            + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
            )

        cosine = np.clip(cosine, -1, 1).astype(np.float32)
        solar_hour = solar_hour.astype(np.float32)

        # Prevent content from being modified accidentally
        cosine.setflags(write=False)
        solar_hour.setflags(write=False)

        return cosine, solar_hour

def processed_b2(raw, valid, timestamp, arrays, solar):
    """Exact continuous clear score, source validity and diagnostic binary clear."""
    time = utc(timestamp)
    cosine, hour = solar.solar_geometry(time)
    day = time.timetuple().tm_yday + (time.hour * 3600 + time.minute * 60) / 86_400
    c = cosine.astype(np.float64)
    phase = np.sin((hour - 12.0) * np.pi / 12.0)
    days_in_year = 366 if isleap(time.year) else 365
    annual = (day - 1.0) * (2.0 * np.pi / days_in_year)
    sine_day, cosine_day = np.sin(annual), np.cos(annual)
    basis = np.stack((np.ones_like(c), c, c * c, phase, c* phase,
                     np.broadcast_to(sine_day, c.shape), np.broadcast_to(cosine_day, c.shape),
                     c * sine_day, c * cosine_day, phase * sine_day, phase * cosine_day), axis = -1)
    reference = np.clip(np.sum(basis * arrays["reference_coefficients"], axis = -1,
                               dtype=np.float64), 0, 2).astype(np.float32)
    raw = np.asarray(raw, np.float32)
    residual = raw - reference

    basis = np.stack((np.ones_like(c), c, c * c, phase), axis = -1)
    threshold = np.sum(basis * arrays["classifier_coefficients"], axis = -1)
    threshold += arrays["offsets"].reshape(c.shape)
    good = (np.asarray(valid, bool) & np.isfinite(raw) & (raw >= 0) & (raw <= 2)
            & (c >= float(np.cos(np.deg2rad(ZSA_MAX)))) & np.isfinite(threshold))
    logits = float(arrays["classifier_k"]) * (residual.astype(np.float64) - threshold)
    score = expit(logits) # numerically robust implementation of the logistic sigmoid from scipy.special
    clear = np.where(good, 1 - score, 0.5).astype(np.float32)
    binary = np.where(good, clear >= 0.5, 255).astype(np.uint8)
    return clear, good, binary

def _array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """_array converts input data to a NumPy float32 array and checks its shape"""
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    return result

def _mask(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """_mask converts input data to a NumPy array and checks its shape"""
    result = np.asarray(value) #preserves type
    if result.shape != shape or result.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return result

def _check_values(values: np.ndarray, name: str, *, valid: Any = True,
                  low: float | None = 0, high: float | None = 1,
                  binary: bool = False) -> None:
    """Check valid entries for finiteness, specified bounds, and binary 0/1 values if required."""
    selected = values[np.broadcast_to(valid, values.shape)]
    bad = ~np.isfinite(selected)
    if low is not None:
        bad |= selected < low
    if high is not None:
        bad |= selected > high
    if binary:
        bad |= (selected != 0) & (selected != 1)
    if bad.any():
        detail = "binary 0/1" if binary else f"in [{low}, {high}]"
        raise ValueError(f"valid {name} values must be finite and {detail}")

def _satellite(sample: Mapping[str, Any], field: str, shape: tuple[int, ...]
               ) -> tuple[np.ndarray, np.ndarray]:
    """_satellite retrieves a satellite field and its validity mask, checks them, and fills missing or invalid pixels with 0.5"""
    
    if field == "current_b2_score":
        mask_name = "current_b2_valid"
    elif field =="current_csm":
        mask_name = "current_csm_valid"
    else:
        raise ValueError(f"{field} must be either current_b2_score or current_csm")
    
    if (field in sample) != (mask_name in sample):
        raise ValueError(f"both {field} and {mask_name} must be present, or both must be absent")
    if field not in sample:
        return np.full(shape, 0.5, np.float32), np.zeros(shape,bool)
    values = _array(sample[field], field, shape)
    valid = _mask(sample[mask_name], mask_name, shape)
    _check_values(values, field, valid=valid) #checks valid pixels have finite values between 0 and 1
    return np.where(valid, values, np.float32(0.5)), valid

def seasonal_fields(issue_time: str | datetime) -> tuple[float, float]:
    """Annual sine/cosine, using the UTC issue date and 365/366-day year."""
    if isinstance(issue_time, str):
        issue_time = datetime.fromisoformat(issue_time)
    if not isinstance(issue_time, datetime) or issue_time.utcoffset() is None:
        raise ValueError("issue_time must be a timezone-aware datetime or ISO timestamp")
    issue_time = issue_time.astimezone(UTC)
    year = issue_time.year
    days_in_year = 366 if isleap(year) else 365
    phase = 2 * np.pi * (issue_time.timetuple().tm_yday - 1) / days_in_year
    return float(np.sin(phase)), float(np.cos(phase))

def _indices(value: Any, name: str) -> np.ndarray:
    """Check the argument is a one-dimensional integer array"""
    result = np.asarray(value)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return result.astype(np.int64)

def _native_record(
    record: Mapping[str, Any],
    coarse_horizon: np.ndarray,
    *,
    fine_shape: tuple[int, int],
) -> dict[str, Any]:
    
    """Validate native CSM targets and geometry; enable coarse-only horizons."""

    # Require one boolean supervision decision per forecast horizon.
    coarse_horizon = _mask(coarse_horizon, "coarse_horizon", (HORIZONS,))

    # Identify the forecast horizon; reject booleans and out-of-range indices.
    horizon = integer_index(record["horizon_index"])
    if (
        isinstance(record["horizon_index"], (bool, np.bool_))
        or not 0 <= horizon < HORIZONS
    ):
        raise ValueError("native horizon_index must be an integer from 0 to 9")

    # Each entry represents one native satellite footprint.
    target = np.asarray(record["target"], dtype=np.float32)
    if target.ndim != 1:
        raise ValueError("native target must be one-dimensional")

    shape = target.shape

    # This mask describes coarse target validity, not fine target validity.
    native_valid = _mask(record["valid"], "native valid", shape)
    _check_values(
        target, "native CSM targets", valid=native_valid, binary=True
    )

    # Sparse overlap mapping:
    # rows[i] identifies a native footprint.
    # cols[i] identifies an overlapping pixel in the flattened fine grid.
    # weights[i] gives that fine pixel's contribution to the footprint average.
    rows = _indices(record["row_indices"], "row_indices")
    cols = _indices(record["col_indices"], "col_indices")
    weights = _array(record["weights"], "weights", rows.shape)

    if cols.shape != rows.shape:
        raise ValueError("native coordinate format row and column shapes differ")

    # Validate index bounds and require positive, finite overlap weights.
    pixels = fine_shape[0] * fine_shape[1]
    if (
        (rows < 0).any()
        or (rows >= target.size).any()
        or (cols < 0).any()
        or (cols >= pixels).any()
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError(
            "native coordinate format indices must be in bounds "
            "and weights positive and finite"
        )

    # One coverage fraction and pair of areas per native footprint.
    coverage, native_area, covered_area = (
        _array(record[name], name, shape)
        for name in ("coverage", "native_area", "covered_area")
    )

    # Check that supplied coverage agrees with the supplied areas.
    # The denominator floor avoids division by zero.
    calculated = covered_area / np.maximum(
        native_area, np.finfo(np.float32).tiny
    )
    comparable = np.isfinite(coverage) & np.isfinite(calculated)
    if (np.abs(coverage[comparable] - calculated[comparable]) > 2e-5).any():
        raise ValueError("native coverage does not match footprint areas")

    # Initially accept valid targets with sufficient coverage and usable areas.
    accepted = (
        native_valid
        & np.isfinite(coverage)
        & (coverage >= MINIMUM_COVERAGE)
        & (coverage <= 1 + 2e-5)
        & np.isfinite(native_area)
        & (native_area > 0)
        & np.isfinite(covered_area)
        & (covered_area > 0)
    )

    # Sum overlap weights separately for each native footprint.
    totals = np.bincount(rows, weights=weights, minlength=target.size)

    # Require at least one contribution and a coarse-only horizon.
    # Fine targets must already be disabled for this horizon by the caller.
    accepted &= (totals > 0) & coarse_horizon[horizon]

    # Accepted footprints must average their contributing fine pixels.
    if not np.allclose(totals[accepted], 1, atol=2e-5, rtol=2e-5):
        raise ValueError(
            "accepted native footprints require normalized area weights"
        )

    # Retain all footprints; the final validity mask selects usable ones.
    # Replace originally invalid targets with a harmless placeholder.
    arrays = {
        "target": np.where(native_valid, target, np.float32(0)),
        "valid": accepted,
        "row_indices": rows,
        "col_indices": cols,
        "weights": weights,
        "coverage": coverage,
        "native_area": native_area,
        "covered_area": covered_area,
    }

    # Copy arrays into independently owned storage and return CPU tensors.
    return {
        "horizon_index": horizon,
        **{
            name: torch.from_numpy(np.array(value, copy=True))
            for name, value in arrays.items()
        },
    }


class ArrayForecastDataset(Dataset):
    """Build 5 fine and 25 coarse channels from named prepared arrays."""

    def __init__(self, samples: Sequence[Mapping[str, Any]], *, fine_target_mode: str = "hard"):
        if fine_target_mode not in ("hard", "soft"):
            raise ValueError("fine_target_mode must be hard or soft")
        self.samples = samples
        self.fine_target_mode = fine_target_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Prepare one forecast sample, including inputs and supervision masks."""

        # Retrieve the prepared arrays for this sample.
        sample = self.samples[index]

        # Establish the fine-grid dimensions.
            # Dimensions must be positive and divisible by four.
        land = np.asarray(sample["land_fraction"], dtype=np.float32)
        shape = land.shape  # (H, W)
        
        if land.ndim !=2 or any(size <= 0 or size % 4 for size in shape):
            raise ValueError("fine grid must be two-dimensional with positive sizes divisible by four")
        
        coarse_shape = (shape[0] // 4, shape[1] // 4)
        target_shape = (HORIZONS, *shape) # (10, H, W)

        # Validate static fields.
        _check_values(land,"land_fraction") # Verify land has finite values between 0 and 1
        elevation = _array(sample["elevation_normalized"], "elevation_normalized", shape) # standardized elevation
        _check_values(elevation, "elevation_normalized", low=None, high=None) # Verify elevation has finite values

        # Solar cosine values must be finite and between -1 and 1.
        current_solar = _array(sample["current_solar_cosine"], "current_solar_cosine", shape)
        _check_values(current_solar, "current_solar_cosine", low=-1)
        future_solar = _array(sample["target_solar_cosine"], "target_solar_cosine", coarse_shape)
        _check_values(future_solar, "target_solar_cosine", low=-1)

        # NBM provides 11 cloud-cover channels on the coarse input grid.
        nbm = _array(sample["nbm_cloud_cover"], "nbm_cloud_cover", (11, *coarse_shape))
        _check_values(nbm, "nbm_cloud_cover") # Verify nbm has finite values

        # Retrieve current satellite observations and their validity masks.
        # Missing or invalid observations receive the placeholder value 0.5.
        b2, b2_valid = _satellite(sample, "current_b2_score", shape)
        csm, csm_valid = _satellite(sample, "current_csm", coarse_shape)

        # Stack five 2D fields into fine-grid channels: (5, H, W).
        fine = np.stack((b2,b2_valid,land,elevation,current_solar)).astype(np.float32)

         # Encode the issue date as seasonal sine and cosine values.
        sine, cosine = seasonal_fields(sample["issue_time]"])

        # Join existing channel groups into 25 conditioning channels:
        # 11 NBM + 10 future solar + 2 seasonal + CSM + CSM validity.
        coarse= np.concatenate((nbm, future_solar,
                                np.full((1, *coarse_shape), sine, np.float32),
                                np.full((1, *coarse_shape), cosine, np.float32),
                                csm[None], csm_valid[None])).astype(np.float32) # csm[None].shape = (1,100,100) vs csm.shape = (100,100)

        # Select coarse supervision for an entire horizon if any pixel
        # falls below the solar threshold. Shape: (10,).
        coarse_horizon = np.asarray((future_solar < NIGHT_SOLAR_THRESHOLD).any(axis=(-2,-1)), dtype=bool) # One decision boolean per horizon: shape (10,)

        # Expand the horizon decisions to (10, H, W).
        # Copy because broadcast_to returns a read-only view, whereas
        # torch.from_numpy expects writable underlying storage.
        coarse_only = np.broadcast_to(coarse_horizon[:, None, None], target_shape).copy() 

        # Fine targets and their validity mask must be supplied together.
        if ("target_fine" in sample) != ("valid_fine" in sample):
                    raise ValueError("provide both target_fine and valid_fine, or omit both")

        # If fine targets are absent, create placeholders marked invalid.
        valid = _mask(sample.get("valid_fine", np.zeros(target_shape, bool)), "valid_fine", target_shape)
        target = _array(sample.get("target_fine", np.zeros(target_shape, np.float32)), "target_fine", target_shape)

        # Disable fine supervision throughout coarse-only horizons.
        # This creates a new array without modifying the supplied mask.
        valid = valid & ~coarse_only

        # Check only active fine targets. Hard mode requires binary labels; soft mode permits [0, 1].
        _check_values(target, "fine targets", valid=valid, binary=self.fine_target_mode == "hard")

        # Replace inactive targets with zero; valid excludes them from loss.
        target = np.where(valid, target, np.float32(0))

        

        # Check AFter!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Prepare native CSM targets and their overlap geometry.
        # Allow at most one record per horizon.
        native, seen = [], set()

        for record in sample.get("native_csm", ()):
            prepared = _native_record(
                record, coarse_horizon, fine_shape=shape
            )
            horizon = prepared["horizon_index"]
    
            if horizon in seen:
                raise ValueError(
                    "duplicate native horizon record; "
                    "combine footprints in one operator"
                )
            seen.add(horizon)
    
            # Keep records containing at least one accepted footprint.
            if prepared["valid"].any():
                native.append(prepared)
    
        # Package this sample as CPU tensors plus metadata.
        result = {
            "x_fine": torch.from_numpy(fine),
            "x_cond": torch.from_numpy(coarse),
            "target_fine": torch.from_numpy(target),
            "valid_fine": torch.from_numpy(valid),
            "coarse_only": torch.from_numpy(coarse_only),
            "native_csm": native,
            "metadata": dict(sample.get("metadata", {})),
            "fine_target_mode": self.fine_target_mode,
        }
    
        # Soft training targets require separate binary evaluation labels.
        if self.fine_target_mode == "soft":
            if (
                "target_fine" in sample
                and "target_fine_binary" not in sample
            ):
                raise ValueError(
                    "soft targets require authoritative "
                    "target_fine_binary evaluation labels"
                )
    
            binary = _array(
                sample.get(
                    "target_fine_binary",
                    np.zeros(target_shape, np.float32),
                ),
                "target_fine_binary",
                target_shape,
            )
            _check_values(
                binary, "binary evaluation targets", valid=valid, binary=True
            )
    
            # Apply the same fine-supervision mask to evaluation labels.
            result["target_fine_binary"] = torch.from_numpy(
                np.where(valid, binary, np.float32(0))
            )
    
        elif "target_fine_binary" in sample:
            raise ValueError(
                "target_fine_binary is only used in soft target mode"
            )
    
        return result


def forecast_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine prepared samples into one batch for training or evaluation."""

    # Require at least one sample.
    if not batch:
        raise ValueError("cannot collate an empty batch")

    # All samples must use the same fine-target mode: "hard" or "soft".
    # The set contains the distinct modes present in the batch.
    modes = {sample["fine_target_mode"] for sample in batch}
    if len(modes) != 1:
        raise ValueError("cannot collate mixed fine target modes")

    # Stack matching tensors along a new first dimension: the batch axis.
    # Example: B tensors shaped (5, H, W) become (B, 5, H, W).
    keys = ("x_fine", "x_cond", "target_fine", "valid_fine", "coarse_only")
    result: dict[str, Any] = {
        name: torch.stack([sample[name] for sample in batch])
        for name in keys
    }

    # Store the shared target mode once for the whole batch.
    result["fine_target_mode"] = batch[0]["fine_target_mode"]

    # Soft mode also supplies separate binary labels for evaluation.
    if result["fine_target_mode"] == "soft":
        result["target_fine_binary"] = torch.stack(
            [sample["target_fine_binary"] for sample in batch]
        )

    # Preserve metadata as a list, in the same order as the samples.
    result["metadata"] = [sample["metadata"] for sample in batch]

    # Combine native CSM records into one list instead of stacking them,
    # because samples may have different numbers of records and footprints.
    # Copy each record's dictionary and add its sample's position in the batch.
    # Example: batch_index=1 identifies the second sample.
    result["native_csm"] = [
        dict(record, batch_index=b)
        for b, sample in enumerate(batch)
        for record in sample["native_csm"]
    ]

    return result


    
        



        


