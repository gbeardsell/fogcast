"""Source transport, causal selection, quality flags and spatial decoding."""
from datetime import timedelta
import numpy as np
import pytest
from fogcast.dataset import utc
from fogcast.sources import GoesCatalog, RangeReader, Scan, decode_csm, parse_scan, resample_b2, select_index

def test_b2_interpolation_preserves_mask_and_original_quantization():
    pytest.importorskip('scipy')
    values = np.array([[0.111111, 0.222222], [0.333333, 0.444444]])
    valid = np.ones((2, 2), bool)
    result, mask = resample_b2(values, valid, np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([[0.5]]), np.array([[0.5]]))
    expected = np.float32(np.uint16(np.rint(np.float32(values.mean()).astype(np.float64) / 0.0001))) * 0.0001
    assert mask.item() and result.item() == expected
    valid[:] = False
    result, mask = resample_b2(values, valid, np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([[0.5]]), np.array([[0.5]]))
    assert not mask.item() and np.isnan(result.item())

class FakeRangeBudget:

    def __init__(self, data, changed=False):
        self.data, self.changed, self.gets = (data, changed, 0)

    def request(self, url, *, method='GET', headers=None, maximum=None):
        if method == 'HEAD':
            return (b'', {'content-length': str(len(self.data)), 'etag': '"fixed"'}, 200)
        self.gets += 1
        a, b = map(int, headers['Range'].removeprefix('bytes=').split('-'))
        return (self.data[a:b + 1], {'etag': '"changed"' if self.changed else '"fixed"', 'content-range': f'bytes {a}-{b}/{len(self.data)}'}, 206)

def test_http_range_reader_cache_seek_eviction_and_source_identity():
    budget = FakeRangeBudget(bytes(range(256)) * 4)
    with RangeReader('https://example.invalid/fixture', budget, block_size=256, cache_blocks=2) as reader:
        assert reader.read(10) == budget.data[:10]
        reader.seek(0)
        assert reader.read(10) == budget.data[:10] and budget.gets == 1
        reader.seek(600)
        assert reader.read(20) == budget.data[600:620]
        reader.seek(900)
        reader.read(20)
        assert len(reader.blocks) == 2
        reader.seek(-5, 2)
        assert reader.read() == budget.data[-5:]
    with pytest.raises(ValueError, match='ETag'):
        RangeReader('https://example.invalid/fixture', FakeRangeBudget(b'test', True)).read(1)

def test_scan_selection_rejects_future_creation_old_scans_and_wrong_band():
    issue = utc('2026-06-01T20:00:00Z')

    def scan(minutes, created, band):
        midpoint = issue - timedelta(minutes=minutes)
        return Scan('fixture.nc', midpoint - timedelta(seconds=30), midpoint + timedelta(seconds=30), created, band)
    expected = scan(4, issue - timedelta(minutes=2), 2)
    records = [expected, scan(1, issue + timedelta(seconds=1), 2), scan(21, issue - timedelta(minutes=20), 2), scan(2, issue - timedelta(minutes=1), 3)]
    catalog = GoesCatalog(None)
    catalog.hour = lambda *_: records
    assert catalog.select(issue, 'ABI-L1b-RadC', 2) == expected
    parsed = parse_scan('ABI-L1b-RadC/2026/152/19/OR_ABI-L1b-RadC-M6C02_G18_s20261521955000_e20261521959500_c20261521959599.nc')
    assert parsed.band == 2 and parsed.scan_end <= issue and (parsed.created <= issue)

def test_csm_uses_declared_class_and_quality_meanings():
    attrs = dict(flag_values=[7, 9], flag_meanings='cloudy clear')
    qa_attrs = dict(flag_values=[4, 5], flag_meanings='good_quality bad_quality')
    decoded = decode_csm(np.array([[7.0, 9.0, 9.0, np.nan]]), np.array([[4.0, 4.0, 5.0, 4.0]]), attrs, qa_attrs)
    np.testing.assert_array_equal(decoded, [[0, 1, 255, 255]])
    with pytest.raises(ValueError, match='quality'):
        decode_csm(np.zeros((1, 1)), np.zeros((1, 1)), attrs, {})

def test_nbm_selects_one_surface_deterministic_cloud_field_and_exact_range():
    cycle = utc('2026-06-01T18:00:00Z')
    index = '1:0:d=2026060118:TMP:2 m above ground:2 hour fcst:\n2:100:d=2026060118:TCDC:surface:2 hour fcst:\n3:200:d=2026060118:TCDC:surface:2 hour fcst:prob > 50:\n'
    assert select_index(index, 300, cycle, 2) == (100, 199)
    with pytest.raises(ValueError, match='exactly one'):
        select_index(index, 300, cycle, 3)
    with pytest.raises(ValueError, match='exceeds'):
        select_index(index, 150, cycle, 2)

def test_incomplete_nbm_cache_is_preserved_before_any_network_request(tmp_path):
    from fogcast.residual_unet_fogcast.sources import download_field
    target = tmp_path / 'f002.grib2'
    target.write_bytes(b'preserve the unverified original')
    with pytest.raises(ValueError, match='incomplete'):
        download_field(utc('2026-06-01T18:00:00Z'), 2, tmp_path, None)
    assert target.read_bytes() == b'preserve the unverified original'
