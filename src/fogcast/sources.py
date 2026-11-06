"""NOAA source discovery, bounded downloads, calibration and spatial registration.

GOES supplies current B2/CSM observations; NBM supplies weather conditioning.
Fitted B2 processing and input packing live in dataset.py. Heavy decoding and
projection dependencies are imported where needed, so the GRIB subprocess never
imports pyproj. Both HTTP block and working-file caches remain bounded.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import numpy as np

from .dataset import utc

# Shared HTTP transport and GOES satellite observations.

class DownloadBudget:
    """Shared HTTP response-byte allowance and monotonic deadline for one run.

    All source readers use the same instance so many small requests cannot each
    claim a fresh allowance. Checks occur at request boundaries; this is not a
    watchdog for arbitrary computation. Counters describe responses handled by
    this object, not transport headers or bytes served from a local cache.
    """

    def __init__(self, *, seconds=600, maximum_bytes=128 * 1024**2):
        """Start the elapsed-time allowance now and initialize HTTP counters."""
        self.deadline = time.monotonic() + seconds
        self.maximum_bytes = maximum_bytes
        self.response_bytes = 0
        self.requests = 0

    def remaining(self):
        """Return seconds left, raising ``TimeoutError`` when the deadline passed."""
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("live forecast deadline reached")
        return remaining

    def request(self, url, *, method="GET", headers=None, maximum=1024**2):
        """Read one response within per-call and shared byte allowances.

        Return ``(body, lowercase_headers, status)``. HEAD requests return an
        empty body. Other requests read at most allowance + 1 bytes to detect
        oversized responses and count those bytes before raising. Socket timeout
        is the smaller of 30 seconds and the remaining run time. HTTP errors
        propagate; there is no automatic retry or Range validation here.
        """
        allowance = min(maximum, self.maximum_bytes - self.response_bytes)
        if allowance <= 0:
            raise RuntimeError("live download byte limit reached")
        request = Request(url, method=method, headers=headers or {})
        with urlopen(request, timeout=min(30, self.remaining())) as response:
            status, info = response.status, dict(response.headers)
            self.requests += 1
            body = b"" if method == "HEAD" else response.read(allowance + 1)
            self.response_bytes += len(body)
            if len(body) > allowance:
                raise RuntimeError("HTTP response exceeds the live download byte limit")
        return body, {k.lower(): v for k, v in info.items()}, status


class RangeReader(io.RawIOBase):
    """Expose a remote object as a seekable file for cropped NetCDF/HDF5 reads.

    A HEAD request binds object size and ETag. Every fetched block must match
    that ETag and its exact Content-Range, preventing mixed object revisions.
    The default LRU retains sixteen 256 KiB blocks; cache misses share the run's
    download budget. Reading a crop may still fetch extra HDF5 chunks/metadata.
    """

    def __init__(self, url, budget, *, block_size=256 * 1024, cache_blocks=16):
        """Bind a nonempty ETag-bearing object and start its cursor at zero."""
        super().__init__()
        self.mode = "rb"
        self.url, self.budget = url, budget
        _, headers, _ = budget.request(url, method="HEAD")
        self.size, self.etag = int(headers["content-length"]), headers.get("etag")
        if not self.etag or self.size <= 0:
            raise ValueError("GOES source requires a nonempty ETag-bound object")
        self.block_size, self.cache_blocks = block_size, cache_blocks
        self.position, self.blocks = 0, OrderedDict()

    def readable(self):
        """Advertise read support to file consumers such as h5netcdf."""
        return True

    def seekable(self):
        """Advertise random access backed by validated HTTP Range requests."""
        return True

    def tell(self):
        """Return the logical byte cursor without accessing the remote object."""
        return self.position

    def seek(self, offset, whence=0):
        """Move the byte cursor relative to start, current position or EOF.

        Seeking past EOF is allowed; negative positions and unknown origins
        raise ``ValueError``. No data is downloaded until a subsequent read.
        """
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError("invalid HTTP file seek")
        self.position = position
        return position

    def read(self, size=-1):
        """Return up to ``size`` bytes, fetching and validating missing blocks.

        Negative/None size reads to EOF; EOF returns empty bytes. The cursor
        advances as blocks are consumed, so a failed read may partially advance
        it. A server ignoring Range, changing ETag or truncating data is rejected.
        """
        if size is None or size < 0:
            size = self.size - self.position
        end = min(self.size, self.position + size)
        chunks = []
        while self.position < end:
            number = self.position // self.block_size
            start = number * self.block_size
            stop = min(self.size - 1, start + self.block_size - 1)
            if number not in self.blocks:
                body, headers, status = self.budget.request(self.url,
                    headers={"Range": f"bytes={start}-{stop}", "If-Match": self.etag},
                    maximum=stop - start + 1)
                expected = f"bytes {start}-{stop}/{self.size}"
                if status != 206 or headers.get("content-range") != expected or headers.get("etag") != self.etag:
                    raise ValueError("NOAA range response or source ETag changed")
                if len(body) != stop - start + 1:
                    raise ValueError("truncated NOAA range response")
                self.blocks[number] = body
            self.blocks.move_to_end(number)
            while len(self.blocks) > self.cache_blocks:
                self.blocks.popitem(last=False)
            take = min(end - self.position, stop - self.position + 1)
            offset = self.position - start
            chunks.append(self.blocks[number][offset:offset + take])
            self.position += take
        return b"".join(chunks)

    def readinto(self, buffer):
        """Fill a writable buffer from the cursor and return the byte count."""
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


@dataclass(frozen=True)
class Scan:
    """GOES-18 object identity with UTC acquisition and creation timestamps.

    ``band`` is present for channel-specific radiance products. Acquisition
    midpoint represents observation time; both scan end and creation time are
    checked separately when selecting data available by a forecast issue.
    """
    key: str
    scan_start: datetime
    scan_end: datetime
    created: datetime
    band: int | None = None

    @property
    def midpoint(self):
        """UTC observation time used for freshness and solar preprocessing."""
        return self.scan_start + (self.scan_end - self.scan_start) / 2

    @property
    def url(self):
        """Public GOES-18 object URL corresponding to the catalog key."""
        return "https://noaa-goes18.s3.amazonaws.com/" + self.key


def parse_scan(key):
    """Parse a GOES-18 NetCDF key into acquisition times and optional ABI band.

    Filename times encode year, day-of-year and fractional UTC seconds; excess
    fractional precision is truncated to microseconds. Other satellite keys or
    missing start/end/creation suffixes raise ``ValueError``. No network I/O occurs.
    """
    match = re.search(r"_s(\d{13,})_e(\d{13,})_c(\d{13,})\.nc$", key)
    if not match or "_G18_" not in key:
        raise ValueError("expected a GOES-18 NetCDF source key")
    def timestamp(digits):
        """Interpret year/day-of-year timestamps with microsecond precision."""
        return datetime.strptime(digits[:13], "%Y%j%H%M%S").replace(tzinfo=UTC,
                microsecond=int((digits[13:] + "000000")[:6]))
    band = re.search(r"C(\d{2})_G18_", key)
    
    start, end, created = match.groups()

    return Scan(
        key=key,
        scan_start=timestamp(start),
        scan_end=timestamp(end),
        created=timestamp(created),
        band=int(band[1]) if band else None,
    )


class GoesCatalog:
    """Per-run S3 listing cache and issue-time selection for GOES-18 products."""

    def __init__(self, budget):
        """Use the shared download budget and begin with an empty hourly cache."""
        self.budget, self.cache = budget, {}

    def hour(self, product, hour):
        """Return cached or paginated scan records under one product/hour prefix.

        ``hour`` should be a UTC hour boundary. Nonempty NetCDF objects are
        parsed into an immutable tuple. A listing is reused for this catalog's
        lifetime, so later arrivals do not refresh a previously listed hour.
        """
        key = product, hour
        if key in self.cache:
            return self.cache[key]
        prefix = f"{product}/{hour.year}/{hour.timetuple().tm_yday:03d}/{hour:%H}/"
        parameters = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        records = []
        while True:
            body, _, _ = self.budget.request("https://noaa-goes18.s3.amazonaws.com/?" + urlencode(parameters))
            root = ElementTree.fromstring(body)
            ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
            for item in root.findall("s:Contents", ns):
                name = item.findtext("s:Key", namespaces=ns)
                size_text = item.findtext("s:Size", namespaces=ns)

                if (
                    name is not None
                    and name.endswith(".nc")
                    and size_text is not None
                    and int(size_text) > 0
                ):
                    records.append(parse_scan(name))

            if root.findtext("s:IsTruncated", namespaces=ns) != "true": break

            token = root.findtext("s:NextContinuationToken", namespaces=ns)
            if not token:
                raise ValueError("GOES listing is truncated but has no continuation token")
            parameters["continuation-token"] = token
        self.cache[key] = tuple(records)
        return self.cache[key]

    def select(self, issue, product, band=None):
        """Select the freshest eligible scan, or ``None`` when none qualifies.

        Scan end and file creation must be no later than issue; the acquisition
        midpoint must be at most twenty minutes old. Listings cover the hours
        containing issue +/- 20 minutes. Equal midpoints prefer later creation
        times, and ``band`` optionally restricts channel-specific products.
        """
        issue = utc(issue)
        hours = {(issue + timedelta(minutes=n)).replace(minute=0, second=0, microsecond=0) for n in (-20, 20)}
        eligible = [s for hour in sorted(hours) for s in self.hour(product, hour)
                    if (band is None or s.band == band) and s.scan_end <= issue and s.created <= issue
                    and 0 <= (issue - s.midpoint).total_seconds() <= 1200]
        return min(eligible, key=lambda s: ((issue - s.midpoint).total_seconds(), -s.created.timestamp()), default=None)


def _window(axis, positions):
    """Return a source-axis slice spanning target extrema plus two-cell padding.

    Source coordinates may run in either direction. No source center inside
    the target extent raises ``ValueError`` instead of returning an empty crop.
    """
    indexes = np.flatnonzero((axis >= np.nanmin(positions)) & (axis <= np.nanmax(positions)))
    if not len(indexes): raise ValueError("Bay Area domain is outside the NOAA source")
    return slice(max(0, indexes[0] - 2), min(len(axis), indexes[-1] + 3))


def _names(metadata):
    """Normalize NetCDF flag meanings from bytes, text or an existing sequence."""
    names = metadata.get("flag_meanings", "")
    if isinstance(names, bytes): names = names.decode()
    return names.split() if isinstance(names, str) else list(names)


def decode_csm(categories, qa, attributes, qa_attributes):
    """Decode metadata-declared cloud classes as clear=1, cloudy=0, invalid=255.

    Category and quality arrays share a shape. Only explicitly good QA values
    are accepted; fill values, nonfinite entries and unrecognized categories
    remain invalid. Both clear/cloudy meanings and good-quality meanings must
    be declared in flag metadata, so numeric codes are never guessed. Probably
    clear/cloudy classes follow their respective binary class.
    """
    good_names = {"good_quality", "good_quality_qf", "valid_good_quality"}
    qa_flags = np.asarray(qa_attributes.get("flag_values", ())).ravel()
    qa_names = _names(qa_attributes)
    if len(qa_flags) != len(qa_names): raise ValueError("invalid CSM quality flag metadata")
    accepted = [v for v, name in zip(qa_flags, qa_names) if str(name).lower() in good_names]
    if not accepted: raise ValueError("CSM quality flag meanings do not declare good data")
    good = np.isin(qa, accepted) & np.isfinite(categories) & np.isfinite(qa)
    for attrs, values in ((attributes, categories), (qa_attributes, qa)):
        for key in ("_FillValue", "missing_value"):
            if key in attrs: good &= ~np.isin(values, np.asarray(attrs[key]).ravel())
    clear = {"clear", "clear_sky", "cloud_free", "confident_clear", "confidently_clear", "confidently_clear_sky",
             "probably_clear", "probably_clear_sky", "likely_clear", "clear_or_probably_clear"}
    cloudy = {"cloud", "cloudy", "confident_cloudy", "confidently_cloudy", "probably_cloudy", "probably_cloud",
              "likely_cloudy", "cloudy_or_probably_cloudy"}
    flags, names = np.asarray(attributes.get("flag_values", ())).ravel(), _names(attributes)
    if not len(flags) or len(flags) != len(names): raise ValueError("explicit CSM flag meanings required")
    result, declared = np.full(categories.shape, 255, np.uint8), set()
    for value, name in zip(flags, names):
        name = str(name).lower().replace("-", "_")
        if name in clear: result[good & (categories == value)] = 1; declared.add(1)
        elif name in cloudy: result[good & (categories == value)] = 0; declared.add(0)
    if declared != {0, 1}: raise ValueError("both clear/cloudy CSM mappings required")
    return result


def read_scan(scan, arrays, budget, *, b2=False):
    """Read a cropped GOES observation and return values, validity and provenance.

    Frozen latitude/longitude maps define the target grid. B2 radiance is
    calibrated with kappa0, checked for native nominal 500 m sampling and
    resampled to the fine grid; frozen clear-score classification happens later
    in ``dataset.processed_b2``. CSM is metadata-decoded and sampled at every
    fourth target pixel without interpolating binary categories.

    Range-backed reads share ``budget``. Provenance records source URL, scan
    midpoint/end, creation time, ETag and response bytes consumed by this read.
    B2 invalid values are NaN; CSM invalid values use 255, with a separate mask.
    """
    import xarray as xr
    before = budget.response_bytes
    with RangeReader(scan.url, budget) as reader:
        with xr.open_dataset(reader, engine="h5netcdf", mask_and_scale=True, decode_times=False) as ds:
            projection = dict(ds["goes_imager_projection"].attrs)
            latitude, longitude = arrays["latitude"], arrays["longitude"]
            if not b2: latitude, longitude = latitude[::4, ::4], longitude[::4, ::4]
            tx, ty = scan_coordinates(projection, latitude, longitude)
            ys, xs = _window(np.asarray(ds.y), ty), _window(np.asarray(ds.x), tx)
            subset = ds.isel(y=ys, x=xs)
            x, y = np.asarray(subset.x, np.float64), np.asarray(subset.y, np.float64)
            qa = np.asarray(subset["DQF"].load(), np.float64)
            if b2:
                if int(np.asarray(ds["band_id"]).ravel()[0]) != 2: raise ValueError("expected ABI band 2")
                height = float(projection["perspective_point_height"])
                if not all(350 <= abs(float(np.diff(ds[axis]).mean())) * height <= 600 for axis in ("x", "y")):
                    raise ValueError("B2 source must have native nominal 500 m sampling")
                values = np.asarray(subset["Rad"].load(), np.float64) * float(np.asarray(ds["kappa0"]).item())
                valid = np.isfinite(values) & np.isfinite(qa) & (qa == 0) & (values >= 0)
                result, valid = resample_b2(values, valid, x, y, tx, ty)
            else:
                field = "ACM" if "ACM" in subset else "BCM"
                values = decode_csm(np.asarray(subset[field].load(), np.float64), qa,
                                    subset[field].attrs, subset["DQF"].attrs)
                result = sample_native(values, x, y, tx, ty)
                valid = result != 255
        etag = reader.etag
    proof = {"url": scan.url, "time": scan.midpoint.isoformat(), "scan_end": scan.scan_end.isoformat(),
                 "created": scan.created.isoformat(), "etag": etag, "response_bytes": budget.response_bytes - before}
    return result, valid, proof


# Source-grid registration preserves native values and validity.

def scan_coordinates(projection, latitude, longitude):
    """Transform geographic degrees to ABI fixed-grid scan angles in radians.

    The CF projection defines the geostationary CRS. Projected meter coordinates
    are divided by satellite perspective height to match NetCDF x/y axes.
    Latitude/longitude arrays retain their input shape.
    """
    from pyproj import CRS, Transformer
    transformer = Transformer.from_crs("EPSG:4326", CRS.from_cf(projection), always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    height = float(projection["perspective_point_height"])
    return np.asarray(x) / height, np.asarray(y) / height


def resample_b2(values, valid, source_x, source_y, target_x, target_y):
    """Map reflectance using the original interpolation and encoding contract.

    Source values/masks are [y, x] on ascending or descending coordinate axes;
    target coordinate arrays share the desired output shape. Bilinear values
    are normalized by interpolated valid support, but the nearest source pixel
    must also be valid. Outside/unsupported targets remain NaN with false masks.

    Valid values undergo the training preparation's uint16 round trip at 1e-4
    resolution. Out-of-range encodings raise rather than clip. Returns float32
    reflectance and a boolean validity mask, without modifying source arrays.
    """
    from scipy.interpolate import RegularGridInterpolator
    x, y = np.asarray(source_x), np.asarray(source_y)
    if x[0] > x[-1]:
        x, values, valid = x[::-1], values[:, ::-1], valid[:, ::-1]
    if y[0] > y[-1]:
        y, values, valid = y[::-1], values[::-1], valid[::-1]
    points = np.column_stack((target_y.ravel(), target_x.ravel()))
    def interpolate(data, method="linear"):
        """Sample sorted source axes, assigning zero support outside the grid."""
        return RegularGridInterpolator((y, x), data, method=method,
                    bounds_error=False, fill_value=0)(points)
    numerator = interpolate(np.where(valid, values, 0.0))
    denominator = interpolate(valid.astype(np.float64))
    good = (interpolate(valid.astype(np.float64), "nearest") > .5) & (denominator > 1e-6)
    result = np.full(points.shape[0], np.nan, np.float64)
    result[good] = numerator[good] / denominator[good]
    result = result.reshape(target_x.shape).astype(np.float32)
    good = good.reshape(target_x.shape)
    scaled = np.rint(result[good].astype(np.float64) / .0001)
    if np.any((scaled < 0) | (scaled >= 65535)):
        raise ValueError("B2 reflectance outside original encoding; clipping is forbidden")
    result[good] = scaled.astype(np.uint16).astype(np.float32) * .0001
    return result, good


def sample_native(values, source_x, source_y, target_x, target_y):
    """Sample binary CSM by nearest center, preserving category values.

    Source values are indexed [y, x]; axes may ascend or descend. Each axis's
    median spacing supplies half-cell outer bounds, preventing extrapolation
    beyond the native grid. Ties prefer the lower coordinate. Nonfinite or
    outside target positions return uint8 invalid sentinel 255.
    """
    def nearest(axis, positions):
        """Return original-axis nearest indices and a finite half-cell bounds mask."""
        order = np.argsort(axis)
        a = np.asarray(axis, float)[order]
        right = np.clip(np.searchsorted(a, positions), 0, len(a) - 1)
        left = np.maximum(0, right - 1)
        chosen = np.where(abs(positions - a[left]) <= abs(positions - a[right]), left, right)
        delta = np.median(np.diff(a))
        inside = np.isfinite(positions) & (positions >= a[0] - delta / 2) & (positions <= a[-1] + delta / 2)
        return order[chosen], inside
    cols, cx = nearest(source_x, target_x)
    rows, cy = nearest(source_y, target_y)
    return np.where(cx & cy, values[rows, cols], 255).astype(np.uint8)


# NBM deterministic weather fields and the isolated ecCodes worker.

GRID_KEYS = ("Nx", "Ny", "latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees",
             "LaDInDegrees", "LoVInDegrees", "Latin1InDegrees", "Latin2InDegrees",
             "DxInMetres", "DyInMetres", "radius", "iScansNegatively", "jScansPositively",
             "jPointsAreConsecutive", "alternativeRowScanning", "shapeOfTheEarth", "gridType")


def select_index(text, size, cycle, lead):
    """Return the inclusive byte range for one deterministic surface TCDC field.

    NOAA's colon-delimited index must have offsets starting at zero and exactly
    one matching cycle, lead and instantaneous deterministic field. The next
    distinct message offset, or full object size, determines the range end.
    Ambiguous selections and ranges outside the object raise ``ValueError``.
    """
    records = [line.split(":") for line in text.splitlines() if line.strip()]
    offsets = sorted({int(record[1]) for record in records})
    if not offsets or offsets[0] != 0: raise ValueError("invalid NBM index offsets")
    hits = [r for r in records if r[2:5] == [f"d={cycle:%Y%m%d%H}", "TCDC", "surface"]
            and r[5] == f"{lead} hour fcst" and not any(r[6:])]
    if len(hits) != 1: raise ValueError("expected exactly one deterministic surface TCDC index entry")
    start = int(hits[0][1])
    end = next((n - 1 for n in offsets if n > start), size - 1)
    if not 0 <= start <= end < size: raise ValueError("NBM index range exceeds source object")
    return start, end


def download_field(cycle, lead, cache, budget):
    """Return one selected NBM GRIB message path and its integrity receipt.

    ``cache`` should be cycle-specific: filenames encode only forecast lead.
    Existing payload/receipt pairs are reused only when source URL and SHA-256
    match. Partial or mismatched entries raise and remain available for review.

    New reads use the index-selected range, bind it to the object's ETag and
    require exactly one complete GRIB message. Payloads are renamed from a
    temporary file before writing receipts; failures may leave a partial pair.
    The working tree under ``cache.parent`` is limited to 256 MiB when admitting
    a new payload. Field identity is checked later against decoded metadata.
    """
    url = (f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{cycle:%Y%m%d}/{cycle:%H}/core/"
           f"blend.t{cycle:%H}z.core.f{lead:03d}.co.grib2")
    cache.mkdir(parents=True, exist_ok=True)
    target, receipt = cache / f"f{lead:03d}.grib2", cache / f"f{lead:03d}.json"
    if target.exists() or receipt.exists():
        if not (target.is_file() and receipt.is_file()):
            raise ValueError("incomplete NBM cache entry; preserve it and choose a fresh cache directory")
        proof = json.loads(receipt.read_text())
        if proof.get("url") != url or hashlib.sha256(target.read_bytes()).hexdigest() != proof["sha256"]:
            raise ValueError("cached NBM field differs; preserve it for review")
        return target, proof
    _, headers, _ = budget.request(url, method="HEAD")
    size, etag = int(headers["content-length"]), headers.get("etag")
    if not etag: raise ValueError("NBM object has no immutable ETag")
    index, _, _ = budget.request(url + ".idx")
    start, end = select_index(index.decode("ascii"), size, cycle, lead)
    payload, response, status = budget.request(url,
        headers={"Range": f"bytes={start}-{end}", "If-Match": etag}, maximum=16 * 1024**2)
    if (status != 206 or response.get("content-range") != f"bytes {start}-{end}/{size}"
            or response.get("etag") != etag or len(payload) != end - start + 1):
        raise ValueError("NBM range or ETag differs from selected source")
    if payload[:4] != b"GRIB" or payload[-4:] != b"7777" or int.from_bytes(payload[8:16], "big") != len(payload):
        raise ValueError("truncated or compound NBM GRIB message")
    proof = {"url": url, "etag": etag, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                 "index_sha256": hashlib.sha256(index).hexdigest(), "range": [start, end]}
    if sum(p.stat().st_size for p in cache.parent.rglob("*") if p.is_file()) + len(payload) > 256 * 1024**2:
        raise RuntimeError("NBM cache would exceed 256 MiB; choose a fresh cache directory")
    temporary = target.with_suffix(".part")
    temporary.write_bytes(payload); temporary.replace(target)
    receipt.write_text(json.dumps(proof, indent=2) + "\n")
    return target, proof


def _metadata(ec, handle, cycle, lead):
    """Validate decoded field identity/time and return supported Lambert geometry.

    Requires instantaneous deterministic surface total cloud cover in percent
    at the exact cycle/lead. Grid scanning and spherical Earth checks constrain
    the row layout assumed by decoding and spatial registration.
    """
    get = lambda key: ec.codes_get(handle, key)
    valid = cycle + timedelta(hours=lead)
    if ([int(get(k)) for k in ("discipline", "parameterCategory", "parameterNumber")] != [0, 6, 1]
            or int(get("productDefinitionTemplateNumber")) != 0 or get("typeOfLevel") != "surface"
            or get("units") != "%" or get("stepType") != "instant"
            or int(get("dataDate")) != int(cycle.strftime("%Y%m%d")) or int(get("dataTime")) != cycle.hour * 100
            or int(get("validityDate")) != int(valid.strftime("%Y%m%d")) or int(get("validityTime")) != valid.hour * 100
            or int(get("forecastTime")) != lead):
        raise ValueError("incorrect NBM field, statistic, cycle or valid time")
    geometry = {k: get(k) for k in GRID_KEYS}
    if (geometry["gridType"] != "lambert" or geometry["iScansNegatively"] or not geometry["jScansPositively"]
            or geometry["jPointsAreConsecutive"] or geometry["shapeOfTheEarth"] not in (1, 6)
            or not 6370000 <= geometry["radius"] <= 6380000):
        raise ValueError("unsupported native NBM geometry")
    return geometry


def decode_job(job):
    """Probe geometry or write cropped percent-valued fields in an ecCodes worker.

    The JSON-compatible job supplies cycle, ordered message paths, first lead,
    mode and output path. Probe mode validates the first field and returns its
    geometry. Decode mode requires every field to match ``job['grid']``, applies
    bitmap/missing/range masks, reverses alternating scan rows when needed and
    writes stacked ``values``/``valid`` arrays to NPZ, returning ``None``.

    Run this in a separate process without importing pyproj, whose binary
    dependencies can conflict with ecCodes. Every created GRIB handle is released.
    """
    # Do not import pyproj in this process: its binary libraries can conflict with ecCodes.
    import eccodes as ec

    from .dataset import utc
    
    cycle = utc(job["cycle"])
    values, masks = [], []
    for lead, path in enumerate(job["paths"], job["first_lead"]):
        handle = ec.codes_new_from_message(Path(path).read_bytes())
        try:
            geometry = _metadata(ec, handle, cycle, lead)
            if job["mode"] == "probe": return {"geometry": geometry}
            if geometry != job["grid"]["geometry"]: raise ValueError("NBM grid changed between forecast hours")
            data = np.asarray(ec.codes_get_values(handle), np.float32).reshape(geometry["Ny"], geometry["Nx"])
            valid = np.isfinite(data) & (data != float(ec.codes_get_double(handle, "missingValue")))
            if ec.codes_get(handle, "bitmapPresent"):
                valid &= np.asarray(ec.codes_get_array(handle, "bitmap"), bool).reshape(data.shape)
            valid &= (data >= 0) & (data <= 100)
            if geometry["alternativeRowScanning"]:
                data[1::2], valid[1::2] = data[1::2, ::-1], valid[1::2, ::-1]
            crop = tuple(slice(*pair) for pair in job["grid"]["roi"])
            values.append(np.where(valid[crop], data[crop], np.nan))
            masks.append(valid[crop].copy())
        finally:
            ec.codes_release(handle)
    np.savez_compressed(job["output"], values=np.stack(values), valid=np.stack(masks))
    return None


def native_grid(geometry):
    """Build a padded NBM crop and conditioning-grid coordinates in its native CRS.

    Uses the GRIB's spherical Lambert parameters, not a default Earth ellipsoid.
    The fixed 600 x 600 UTM input determines a crop padded by three source
    cells. Return the geometry/ROI descriptor, cropped x/y axes and transformed
    target coordinates sampled every fourth pixel (150 x 150).
    """
    from pyproj import CRS, Transformer
    g = geometry
    crs = CRS.from_proj4(f"+proj=lcc +lat_1={g['Latin1InDegrees']} +lat_2={g['Latin2InDegrees']} "
        f"+lat_0={g['LaDInDegrees']} +lon_0={g['LoVInDegrees']} +R={g['radius']} +units=m +no_defs")
    geographic = CRS.from_proj4(f"+proj=longlat +R={g['radius']} +no_defs")
    x0, y0 = Transformer.from_crs(geographic, crs, always_xy=True).transform(
        g["longitudeOfFirstGridPointInDegrees"], g["latitudeOfFirstGridPointInDegrees"])
    xx, yy = np.meshgrid(400250 + np.arange(600) * 500, 4030250 + np.arange(600) * 500)
    tx, ty = Transformer.from_crs(32610, crs, always_xy=True).transform(xx, yy)
    ix, iy = (tx - x0) / g["DxInMetres"], (ty - y0) / g["DyInMetres"]
    cols = [max(0, int(np.floor(ix.min())) - 3), min(g["Nx"], int(np.ceil(ix.max())) + 4)]
    rows = [max(0, int(np.floor(iy.min())) - 3), min(g["Ny"], int(np.ceil(iy.max())) + 4)]
    x = x0 + np.arange(*cols) * g["DxInMetres"]
    y = y0 + np.arange(*rows) * g["DyInMetres"]
    return {"geometry": g, "roi": [rows, cols]}, x, y, tx[::4, ::4], ty[::4, ::4]


def fetch_nbm(issue, cache, budget):
    """Return eleven float32 cloud-cover maps [11, 150, 150] and provenance.

    ``issue`` is a UTC hour boundary. Try cycles two, three, then four hours
    earlier, selecting leads that cover issue through +10h from one cycle.
    Only HTTP 404/503 triggers an older-cycle attempt; invalid geometry, missing
    coverage or other errors propagate rather than silently changing inputs.

    Messages are cached with receipts and decoded in an isolated subprocess.
    Percent cloud cover becomes a [0, 1] fraction before bilinear interpolation;
    all conditioning locations must have complete valid interpolation support.
    Files remain under ``cache``; all attempts share the same download/deadline
    budget. Provenance records cycle, leads, field receipts and lag policy.
    """
    from scipy.interpolate import RegularGridInterpolator
    errors = []
    for lag in (2, 3, 4):
        cycle = issue - timedelta(hours=lag)
        directory = cache / cycle.strftime("%Y%m%dT%H%M%SZ")
        try:
            selected = [download_field(cycle, lead, directory, budget) for lead in range(lag, lag + 11)]
        except HTTPError as error:
            if error.code not in (404, 503): raise
            errors.append(f"{cycle.isoformat()}: HTTP {error.code}"); continue
        job = {"mode": "probe", "paths": [str(p) for p, _ in selected], "cycle": cycle.isoformat(),
                   "first_lead": lag, "output": str(directory / "decoded.npz")}
        def run_worker():
            """Run the current job within the remaining deadline and parse probe JSON."""
            path = directory / "decode-job.json"  # noqa: B023
            path.write_text(json.dumps(job) + "\n")  # noqa: B023
            try:
                result = subprocess.run([sys.executable, "-m", __name__, "--job", str(path)],
                    capture_output=True, text=True, check=True, timeout=budget.remaining())
            except subprocess.CalledProcessError as error:
                raise RuntimeError("NBM decoder failed: " + error.stderr[-2000:]) from error
            return json.loads(result.stdout) if result.stdout.strip() else None
        probe = run_worker()

        if not isinstance(probe, dict):
            raise ValueError("NBM decoder returned no probe metadata")
        grid, x, y, tx, ty = native_grid(probe["geometry"])
        job.update(mode="decode", grid=grid)
        run_worker()
        points = np.column_stack((ty.ravel(), tx.ravel()))
        fields = []
        with np.load(job["output"], allow_pickle=False) as native:
            for values, valid in zip(native["values"], native["valid"]):
                data = np.where(valid, values.astype(np.float32) / 100, 0)
                mapped = RegularGridInterpolator((y, x), data, bounds_error=False, fill_value=0)(points).reshape(150, 150)
                coverage = RegularGridInterpolator((y, x), valid.astype(float), bounds_error=False, fill_value=0)(points)
                if not (coverage >= 1 - 1e-7).all(): raise ValueError("incomplete NBM conditioning grid")
                fields.append(mapped.astype(np.float32))
        return np.stack(fields), {"cycle": cycle.isoformat(), "leads": list(range(lag, lag + 11)),
                                     "fields": [proof for _, proof in selected], "availability_policy": "minimum_2h_cycle_lag"}
    raise RuntimeError("no complete current NBM cycle with 2–4 h lag: " + "; ".join(errors))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    result = decode_job(json.loads(Path(args.job).read_text()))
    if result is not None: print(json.dumps(result))
