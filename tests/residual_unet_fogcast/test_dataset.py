import copy
from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from fogcast import ArrayForecastDataset, forecast_collate, forecast_loss
from fogcast.residual_unet_fogcast.dataset import CONDITIONING_CHANNELS, FINE_CHANNELS, NIGHT_SOLAR_THRESHOLD, seasonal_fields, SolarGeometry, processed_b2, utc

@pytest.fixture
def sample():
    shape = (8, 8)
    return dict(issue_time='2024-03-01T00:00:00Z', land_fraction=np.full(shape, 0.3, np.float32), elevation_normalized=np.full(shape, -1.25, np.float32), current_solar_cosine=np.full(shape, 0.7, np.float32), current_b2_score=np.full(shape, 0.9, np.float32), current_b2_valid=np.ones(shape, bool), nbm_cloud_cover=np.broadcast_to(np.arange(11, dtype=np.float32)[:, None, None] / 10, (11, 2, 2)).copy(), target_solar_cosine=np.full((10, *shape), -0.2, np.float32), current_csm=np.full((2, 2), 0.8, np.float32), current_csm_valid=np.ones((2, 2), bool), target_fine=np.zeros((10, *shape), np.float32), valid_fine=np.zeros((10, *shape), bool))

def native(coverage=1):
    return dict(fine_shape=(8, 8), row_indices=np.array([0, 0, 1, 1], np.int64), col_indices=np.array([0, 1, 2, 3], np.int64), weights=np.array([0.25, 0.75, 0.6, 0.4], np.float32), coverage=np.full(2, coverage, np.float32), native_area=np.ones(2, np.float32), covered_area=np.full(2, coverage, np.float32))

def add_coarse(sample):
    sample['target_coarse'] = np.tile(np.array([0, 1], np.float32), (10, 1))
    sample['valid_coarse'] = np.ones((10, 2), bool)

def test_all_channel_positions_fills_shapes_and_source_immutability(sample):
    sample['current_b2_valid'][0, 0] = False
    sample['current_b2_score'][0, 0] = np.nan
    sample['current_csm_valid'][0, 0] = False
    sample['current_csm'][0, 0] = 255
    sample['target_solar_cosine'] = np.linspace(-1, 1, 640, dtype=np.float32).reshape(10, 8, 8)
    before = copy.deepcopy(sample)
    item = ArrayForecastDataset([sample])[0]
    fine, cond = (item['x_fine'], item['x_coarse'])
    assert fine.shape == (5, 8, 8) and cond.shape == (25, 2, 2)
    assert fine.dtype == cond.dtype == torch.float32
    assert len(FINE_CHANNELS) == 5 and len(CONDITIONING_CHANNELS) == 25
    np.testing.assert_array_equal(fine[0], np.where(sample['current_b2_valid'], sample['current_b2_score'], 0.5))
    for channel, field in enumerate(FINE_CHANNELS[1:], 1):
        np.testing.assert_array_equal(fine[channel], sample[field])
    np.testing.assert_array_equal(cond[:11], sample['nbm_cloud_cover'])
    np.testing.assert_array_equal(cond[11:21], sample['target_solar_cosine'][:, ::4, ::4])
    sine, cosine = seasonal_fields(sample['issue_time'])
    torch.testing.assert_close(cond[21], torch.full((2, 2), sine))
    torch.testing.assert_close(cond[22], torch.full((2, 2), cosine))
    np.testing.assert_array_equal(cond[23], np.where(sample['current_csm_valid'], sample['current_csm'], 0.5))
    np.testing.assert_array_equal(cond[24], sample['current_csm_valid'])
    assert torch.isfinite(fine).all() and torch.isfinite(cond).all()
    for key in before:
        if isinstance(before[key], np.ndarray):
            np.testing.assert_array_equal(sample[key], before[key])

def test_missing_satellites_and_targets_make_inference_sample(sample):
    for key in ('current_b2_score', 'current_b2_valid', 'current_csm', 'current_csm_valid', 'target_fine', 'valid_fine'):
        sample.pop(key)
    item = ArrayForecastDataset([sample])[0]
    assert (item['x_fine'][0] == 0.5).all() and (not item['x_fine'][1].any())
    assert (item['x_coarse'][23] == 0.5).all() and (not item['x_coarse'][24].any())
    assert not item['valid_fine'].any() and item['native_geometry'] is None
    assert item['target_coarse'].shape == item['valid_coarse'].shape == (10, 0)

def test_one_source_per_target_including_twilight_boundary_and_daytime_outage(sample):
    add_coarse(sample)
    sample['target_solar_cosine'][1, 0, 0] = NIGHT_SOLAR_THRESHOLD
    sample['target_solar_cosine'][2, 0, 0] = np.nextafter(NIGHT_SOLAR_THRESHOLD, -np.inf, dtype=np.float32)
    sample['target_solar_cosine'][3, 0, 0] = 0.7
    sample['valid_fine'][1, 0, 0] = True
    sample['valid_fine'][3, 0, 0] = True
    sample['target_fine'][3, 0, 0] = 1
    sample['target_solar_cosine'][4] = 0.7
    sample['valid_fine'][5, 0, 0] = True
    sample['valid_coarse'][5, 1] = False
    item = ArrayForecastDataset([sample], native_geometry=native(0.99))[0]
    assert item['coarse_supervision'][:6].tolist() == [True, False, True, False, False, True]
    assert item['valid_coarse'][:6].tolist() == [[True, True], [False, False], [True, True], [False, False], [False, False], [True, False]]
    assert item['valid_fine'][:6].sum((1, 2)).tolist() == [0, 1, 0, 1, 0, 0]
    assert not (item['valid_fine'].any((1, 2)) & item['valid_coarse'].any(1)).any()

def test_geometry_is_filtered_once_and_labels_keep_fixed_rows_with_missing_qa(sample):
    add_coarse(sample)
    geometry = native()
    geometry['coverage'][0] = geometry['covered_area'][0] = 0.989
    other = copy.deepcopy(sample)
    other['valid_coarse'][0, 1] = False
    dataset = ArrayForecastDataset([sample, other], native_geometry=geometry)
    first, second = (dataset[0], dataset[1])
    assert first['target_coarse'].shape == second['target_coarse'].shape == (10, 1)
    assert first['native_geometry'] is second['native_geometry']
    assert first['native_geometry']['row_indices'].tolist() == [0, 0]
    assert first['native_geometry']['col_indices'].tolist() == [2, 3]
    assert first['target_coarse'][0].item() == 1
    assert not second['valid_coarse'][0].item()
    assert second['target_coarse'][0].item() == 0
    assert geometry['row_indices'].tolist() == [0, 0, 1, 1]

def test_no_covered_native_rows_produces_empty_support_and_skips_update(sample):
    add_coarse(sample)
    dataset = ArrayForecastDataset([sample], native_geometry=native(0.98))
    batch = forecast_collate([dataset[0]])
    assert batch['valid_coarse'].shape == (1, 10, 0)
    assert batch['native_geometry']['weights'].numel() == 0
    result = forecast_loss(torch.zeros(1, 10, 8, 8, requires_grad=True), batch)
    assert result.skip_update and result.loss.item() == 0
    result.loss.backward()

def test_soft_targets_keep_binary_truth_and_zero_invalid_targets(sample):
    sample['target_solar_cosine'][0] = 0.7
    sample['valid_fine'][0, 0, 0] = True
    sample['target_fine'].fill(np.nan)
    sample['target_fine'][0, 0, 0] = 0.8
    sample['target_fine_binary'] = np.ones((10, 8, 8), np.float32)
    item = ArrayForecastDataset([sample], fine_target_mode='soft')[0]
    assert item['target_fine'][0, 0, 0] == pytest.approx(0.8)
    assert item['target_fine_binary'][0, 0, 0] == 1
    assert torch.count_nonzero(item['target_fine']) == 1
    assert torch.count_nonzero(item['target_fine_binary']) == 1
    sample.pop('target_fine_binary')
    with pytest.raises(ValueError, match='authoritative'):
        ArrayForecastDataset([sample], fine_target_mode='soft')[0]
    with pytest.raises(ValueError, match='binary'):
        ArrayForecastDataset([sample])[0]

def test_dataloader_stacks_coarse_arrays_and_shares_geometry(sample):
    add_coarse(sample)
    other = copy.deepcopy(sample)
    other['valid_coarse'][0] = False
    dataset = ArrayForecastDataset([sample, other], native_geometry=native())
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=forecast_collate)))
    assert batch['x_fine'].shape == (2, 5, 8, 8)
    assert batch['target_coarse'].shape == (2, 10, 2)
    assert batch['native_geometry'] is dataset.native_geometry
    assert batch['valid_coarse'][:, 0].sum(1).tolist() == [2, 0]
    assert 'batch_index' not in dataset.native_geometry
    with pytest.raises(ValueError, match='share one'):
        forecast_collate([dataset[0], ArrayForecastDataset([sample], native_geometry=native())[0]])
    soft = copy.deepcopy(sample)
    soft['target_fine_binary'] = soft['target_fine'].copy()
    with pytest.raises(ValueError, match='mixed'):
        forecast_collate([dataset[0], ArrayForecastDataset([soft], native_geometry=native(), fine_target_mode='soft')[0]])
    with pytest.raises(ValueError, match='empty'):
        forecast_collate([])

@pytest.mark.parametrize('field,value', [('nbm_cloud_cover', np.nan), ('nbm_cloud_cover', 50), ('current_b2_score', np.inf), ('current_csm', -1), ('land_fraction', 1.1), ('elevation_normalized', np.nan), ('current_solar_cosine', -1.1), ('target_solar_cosine', 1.1)])
def test_invalid_valid_values_rejected(sample, field, value):
    sample[field].flat[0] = value
    with pytest.raises(ValueError, match='finite'):
        ArrayForecastDataset([sample])[0]

@pytest.mark.parametrize('change', ['weights', 'indices', 'areas', 'float_indices', 'label', 'grid'])
def test_bad_native_geometry_and_labels_rejected(sample, change):
    record = native()
    if change == 'weights':
        record['weights'][0] = 0.5
    if change == 'indices':
        record['col_indices'][0] = 64
    if change == 'areas':
        record['covered_area'][0] = 0.5
    if change == 'float_indices':
        record['row_indices'] = record['row_indices'].astype(float)
    if change == 'grid':
        record['fine_shape'] = (4, 4)
    add_coarse(sample)
    if change == 'label':
        sample['target_coarse'][0, 0] = 0.5
    with pytest.raises(ValueError):
        ArrayForecastDataset([sample], native_geometry=record)[0]

def test_coarse_labels_require_paired_masks_geometry_and_new_array_contract(sample):
    add_coarse(sample)
    with pytest.raises(ValueError, match='shared native_geometry'):
        ArrayForecastDataset([sample])[0]
    sample.pop('valid_coarse')
    with pytest.raises(ValueError, match='provide both'):
        ArrayForecastDataset([sample], native_geometry=native())[0]
    sample.pop('target_coarse')
    sample['native_csm'] = []
    with pytest.raises(ValueError, match='replaced'):
        ArrayForecastDataset([sample])[0]

def test_shape_mask_and_paired_fields_validated(sample):
    sample['nbm_cloud_cover'] = np.zeros((10, 2, 2), np.float32)
    with pytest.raises(ValueError, match='shape'):
        ArrayForecastDataset([sample])[0]
    sample['nbm_cloud_cover'] = np.zeros((11, 2, 2), np.float32)
    sample['current_b2_valid'] = np.ones((8, 8), np.uint8)
    with pytest.raises(ValueError, match='boolean'):
        ArrayForecastDataset([sample])[0]
    sample.pop('current_b2_valid')
    with pytest.raises(ValueError, match='both'):
        ArrayForecastDataset([sample])[0]

@pytest.mark.parametrize('year,days', [(2024, 366), (2026, 365), (2100, 365), (2000, 366)])
def test_seasonal_leap_year_and_utc_conversion(year, days):
    stamp = datetime(year, 3, 1, tzinfo=timezone.utc)
    phase = 2 * np.pi * (stamp.timetuple().tm_yday - 1) / days
    np.testing.assert_allclose(seasonal_fields(stamp), (np.sin(phase), np.cos(phase)))
    assert seasonal_fields('2026-01-01T00:30:00+01:00') == seasonal_fields('2025-12-31T23:30:00Z')
    with pytest.raises(ValueError, match='timezone'):
        seasonal_fields('2026-01-01T00:00:00')

def test_solar_cache_is_bounded_read_only_and_preserves_signed_night():
    solar = SolarGeometry([[37.7749]], [[-122.4194]])
    time = utc('2026-06-01T20:00:00Z')
    first = solar.solar_geometry(time)[0]
    assert first is solar.solar_geometry(time)[0]
    assert solar._solar_geometry.cache_info().hits == 1
    assert first[0, 0] > 0.8
    assert solar.solar_geometry(time + timedelta(hours=12))[0][0, 0] < 0
    with pytest.raises(ValueError):
        first[0, 0] = 0
    for h in range(70):
        solar.solar_geometry(time + timedelta(hours=h))[0]
    assert solar._solar_geometry.cache_info().currsize == 64

def test_reference_classifier_equality_clear_scores_and_night_mask():
    solar = SolarGeometry([[37.7749]], [[-122.4194]])
    arrays = dict(reference_coefficients=np.zeros((1, 1, 11)), offsets=np.zeros(1), classifier_coefficients=np.zeros(4), classifier_k=np.array(2.0))
    arrays['reference_coefficients'][0, 0, 0] = 0.1
    time = utc('2026-06-01T20:00:00Z')
    clear, valid, binary = processed_b2([[0.1]], [[True]], time, arrays, solar)
    assert valid.item() and clear.item() == 0.5 and (binary.item() == 1)
    clear, valid, binary = processed_b2([[0.2]], [[True]], time, arrays, solar)
    residual = float(np.float32(0.2) - np.float32(0.1))
    assert clear.item() == pytest.approx(1 / (1 + np.exp(2 * residual)))
    assert binary.item() == 0
    clear, valid, binary = processed_b2([[0.2]], [[True]], time + timedelta(hours=12), arrays, solar)
    assert not valid.item() and clear.item() == 0.5 and (binary.item() == 255)
