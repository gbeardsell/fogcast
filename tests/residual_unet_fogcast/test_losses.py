import math
import pytest
import torch
from fogcast.residual_unet_fogcast import forecast_loss, stable_probability_pool

def native(cols=(0, 1), weights=(0.25, 0.75), shape=(1, 2)):
    return {'fine_shape': shape, 'row_indices': torch.zeros(len(cols), dtype=torch.long), 'col_indices': torch.tensor(cols), 'weights': torch.tensor(weights)}

def batch(z, fine=None, target=None, coarse_target=None, coarse_valid=None, geometry=None, coarse_mode=None):
    has_coarse = coarse_target is not None
    if coarse_target is None:
        coarse_target = z.new_empty((*z.shape[:2], 0))
    else:
        coarse_target = torch.broadcast_to(torch.as_tensor(coarse_target, dtype=z.dtype), (*z.shape[:2], 1)).clone()
    if coarse_valid is None:
        coarse_valid = torch.ones_like(coarse_target, dtype=torch.bool)
    if coarse_mode is None:
        coarse_mode = torch.full(z.shape[:2], has_coarse, dtype=torch.bool)
    return {'target_fine': torch.zeros_like(z) if target is None else target, 'valid_fine': torch.zeros_like(z, dtype=torch.bool) if fine is None else fine, 'coarse_supervision': coarse_mode, 'target_coarse': coarse_target, 'valid_coarse': coarse_valid, 'native_geometry': (native() if geometry is None else geometry) if has_coarse else None}

def test_pools_probabilities_not_logits_and_exact_gradient():
    z = torch.tensor([[-2.0, 3.0]], requires_grad=True)
    rows = torch.tensor([0, 0])
    cols = torch.tensor([0, 1])
    weights = torch.tensor([0.25, 0.75])
    log_p, log_q = stable_probability_pool(z, rows, cols, weights, 1)
    expected = 0.25 * z[0, 0].sigmoid() + 0.75 * z[0, 1].sigmoid()
    torch.testing.assert_close(log_p.exp()[0], expected)
    torch.testing.assert_close(log_q.exp()[0], 1 - expected)
    actual_grad = torch.autograd.grad(-log_p.sum(), z, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(-expected.log(), z)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert not torch.isclose(log_p.exp()[0], (0.25 * z[0, 0] + 0.75 * z[0, 1]).sigmoid())

def test_shared_geometry_batched_pooling_matches_each_map_and_gradients():
    z = torch.linspace(-8, 8, 12).reshape(2, 3, 1, 2).requires_grad_()
    geometry = native()
    args = (geometry['row_indices'], geometry['col_indices'], geometry['weights'], 1)
    log_p, log_q = stable_probability_pool(z, *args)
    expected = 0.25 * z[..., 0, 0].sigmoid() + 0.75 * z[..., 0, 1].sigmoid()
    torch.testing.assert_close(log_p.exp()[..., 0], expected)
    torch.testing.assert_close(log_q.exp()[..., 0], 1 - expected)
    actual_grad = torch.autograd.grad(-log_p.sum(), z, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(-expected.log().sum(), z)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    for b in range(2):
        for h in range(3):
            single = stable_probability_pool(z[b, h], *args)
            torch.testing.assert_close(log_p[b, h], single[0])
            torch.testing.assert_close(log_q[b, h], single[1])

@pytest.mark.parametrize('value,target', [(10000.0, 0.0), (-10000.0, 1.0), (10000.0, 1.0), (-10000.0, 0.0)])
def test_extreme_finite_logits_preserve_loss_and_gradients(value, target):
    z = torch.full((1, 1, 1, 2), value, requires_grad=True)
    result = forecast_loss(z, batch(z, coarse_target=target))
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert torch.isfinite(z.grad).all()
    if (value > 0) != bool(target):
        assert result.loss.item() == pytest.approx(2500.0, abs=0.01)
        assert z.grad.abs().sum() > 0.2

def test_masked_nan_fine_targets_zero_gradient_and_empty_skip():
    z = torch.zeros(1, 2, 1, 2, requires_grad=True)
    target = torch.tensor([[[[1.0, float('nan')]], [[float('nan'), float('nan')]]]])
    fine = torch.tensor([[[[True, False]], [[False, False]]]])
    result = forecast_loss(z, batch(z, fine=fine, target=target))
    assert result.statistics.fine_count.tolist() == [1, 0]
    assert result.loss.item() == pytest.approx(math.log(2))
    result.loss.backward()
    assert z.grad[0, 0, 0, 1] == 0
    assert z.grad[0, 1].abs().sum() == 0
    empty = forecast_loss(z.detach().requires_grad_(), batch(z))
    assert empty.skip_update
    assert empty.loss.item() == 0

def test_coarse_only_loss_keeps_alpha_and_partition_aggregation():
    z = torch.tensor([[[[0.0, 0.0]]], [[[2.0, 2.0]]]], requires_grad=True)
    result = forecast_loss(z, batch(z, coarse_target=1.0), coarse_weight=0.25)
    expected = 0.25 * (torch.nn.functional.softplus(-z[0, 0, 0, 0]) + torch.nn.functional.softplus(-z[1, 0, 0, 0])) / 2
    torch.testing.assert_close(result.loss, expected)
    assert result.statistics.coarse_count.tolist() == [2]
    pieces = [forecast_loss(z[i:i + 1], batch(z[i:i + 1], coarse_target=1.0)).statistics for i in range(2)]
    combined, _ = (pieces[0] + pieces[1]).objective()
    torch.testing.assert_close(combined, result.loss)
    assert forecast_loss(z, batch(z, coarse_target=1.0), coarse_weight=0).skip_update

def test_same_lead_can_be_fine_and_coarse_in_different_samples_with_separate_denominators():
    z = torch.tensor([[[[0.0, 2.0]]], [[[1.0, 3.0]]]], requires_grad=True)
    fine = torch.zeros_like(z, dtype=torch.bool)
    fine[0] = True
    payload = batch(z, fine=fine, coarse_target=1.0, coarse_valid=torch.tensor([[[False]], [[True]]]), coarse_mode=torch.tensor([[False], [True]]))
    result = forecast_loss(z, payload)
    p = 0.25 * z[1, 0, 0, 0].sigmoid() + 0.75 * z[1, 0, 0, 1].sigmoid()
    expected = torch.nn.functional.softplus(z[0]).mean() - 0.25 * p.log()
    torch.testing.assert_close(result.loss, expected)
    assert result.statistics.fine_count.tolist() == [2]
    assert result.statistics.coarse_count.tolist() == [1]
    result.loss.backward()
    assert torch.isfinite(z.grad).all()

def test_source_exclusivity_is_checked_even_for_hand_built_batches():
    z = torch.zeros(1, 1, 1, 2)
    with pytest.raises(ValueError, match='selected supervision'):
        forecast_loss(z, batch(z, fine=torch.ones_like(z, dtype=torch.bool), coarse_target=1.0))
    with pytest.raises(ValueError, match='selected supervision'):
        forecast_loss(z, batch(z, coarse_target=1.0, coarse_mode=torch.tensor([[False]])))

def test_fixed_footprints_with_unequal_qa_counts_match_direct_bce_and_brier():
    z = torch.linspace(-2, 3, 16).reshape(2, 2, 1, 4).requires_grad_()
    y = torch.tensor([[[1.0, float('nan')], [float('nan'), float('nan')]], [[0.0, 1.0], [float('nan'), 0.0]]])
    valid = torch.isfinite(y)
    payload = batch(z)
    payload.update(target_coarse=y, valid_coarse=valid, coarse_supervision=torch.ones(2, 2, dtype=torch.bool), native_geometry=dict(fine_shape=(1, 4), row_indices=torch.tensor([0, 0, 1, 1]), col_indices=torch.arange(4), weights=torch.tensor([0.25, 0.75, 0.6, 0.4])))
    result = forecast_loss(z, payload)
    p = torch.stack((0.25 * z[..., 0, 0].sigmoid() + 0.75 * z[..., 0, 1].sigmoid(), 0.6 * z[..., 0, 2].sigmoid() + 0.4 * z[..., 0, 3].sigmoid()), -1)
    safe_y = torch.where(valid, y, 0)
    losses = torch.nn.functional.binary_cross_entropy(p, safe_y, reduction='none')
    expected_sums = torch.where(valid, losses, 0).sum((0, 2))
    expected = 0.25 * (expected_sums / torch.tensor([3, 1])).mean()
    torch.testing.assert_close(result.loss, expected)
    assert result.statistics.coarse_count.tolist() == [3, 1]
    torch.testing.assert_close(result.statistics.coarse_brier_sum, torch.where(valid, (p - safe_y).square(), 0).sum((0, 2), dtype=torch.float64))
    actual_grad = torch.autograd.grad(result.loss, z, retain_graph=True)[0]
    torch.testing.assert_close(actual_grad, torch.autograd.grad(expected, z)[0])
    assert actual_grad[0, 0, 0, 2:].abs().sum() == 0

def test_equal_horizons_not_raw_pixel_counts():
    z = torch.tensor([[[[0.0, 0.0]], [[2.0, 2.0]]]])
    fine = torch.tensor([[[[True, False]], [[True, True]]]])
    result = forecast_loss(z, batch(z, fine=fine))
    expected = (torch.nn.functional.softplus(z[0, 0, 0, 0]) + torch.nn.functional.softplus(z[0, 1, 0, 0])) / 2
    torch.testing.assert_close(result.loss, expected)

def test_bad_weights_geometry_and_nonbinary_labels_fail():
    z = torch.zeros(1, 1, 1, 2)
    with pytest.raises(ValueError, match='normalized'):
        forecast_loss(z, batch(z, coarse_target=1.0, geometry=native(weights=(0.2, 0.2))))
    with pytest.raises(ValueError, match='matching the fine grid'):
        forecast_loss(z, batch(z, coarse_target=1.0, geometry=native(shape=(2, 1))))
    with pytest.raises(ValueError, match='binary'):
        forecast_loss(z, batch(z, coarse_target=0.5))

def test_soft_bce_uses_continuous_target_and_brier_uses_binary_truth():
    z = torch.tensor([[[[1.0, -1.0]]]], requires_grad=True)
    target = torch.tensor([[[[0.8, 0.2]]]])
    labels = torch.tensor([[[[1.0, 0.0]]]])
    payload = batch(z, fine=torch.ones_like(z, dtype=torch.bool), target=target)
    payload.update(fine_target_mode='soft', target_fine_binary=labels)
    result = forecast_loss(z, payload)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(z, target)
    torch.testing.assert_close(result.loss, expected)
    expected_brier = (z.detach().sigmoid() - labels).square().mean().item()
    assert result.statistics.report()['fine_brier'] == pytest.approx([expected_brier])
    result.loss.backward()
    torch.testing.assert_close(z.grad, (z.detach().sigmoid() - target) / 2)
    payload.pop('target_fine_binary')
    with pytest.raises(ValueError, match='authoritative'):
        forecast_loss(z, payload)

def test_partition_sums_counts_and_unsupported_metrics():
    z = torch.tensor([[[[0.0, 2.0]], [[0.0, 0.0]]], [[[1.0, -1.0]], [[0.0, 0.0]]]])
    fine = torch.zeros_like(z, dtype=torch.bool)
    fine[0, 0, 0, 0] = True
    fine[1, 0] = True
    whole = forecast_loss(z, batch(z, fine=fine)).statistics
    pieces = [forecast_loss(z[b:b + 1], batch(z[b:b + 1], fine=fine[b:b + 1])).statistics.detached() for b in range(2)]
    combined = pieces[0] + pieces[1]
    torch.testing.assert_close(combined.fine_sum, whole.fine_sum)
    torch.testing.assert_close(combined.fine_brier_sum, whole.fine_brier_sum)
    assert combined.report()['fine_count'] == [3, 0]
    assert combined.report()['fine_brier'][1] is None
    assert combined.report()['coarse_brier'] == [None, None]
    assert combined.report()['fine_bce'][0] == pytest.approx(whole.report()['fine_bce'][0])

def test_missing_coarse_qa_does_not_change_geometry_or_leak_nan_gradients():
    z = torch.zeros(2, 1, 1, 2, requires_grad=True)
    payload = batch(z, coarse_target=torch.tensor([[[1.0]], [[float('nan')]]]), coarse_valid=torch.tensor([[[True]], [[False]]]))
    result = forecast_loss(z, payload)
    assert result.statistics.coarse_count.tolist() == [1]
    assert result.statistics.report()['coarse_brier'] == pytest.approx([0.25])
    result.loss.backward()
    assert z.grad[1].abs().sum() == 0
    assert torch.isfinite(z.grad).all()

def test_declared_valid_nan_is_rejected():
    z = torch.zeros(1, 1, 1, 2)
    with pytest.raises(ValueError, match='finite'):
        forecast_loss(z, batch(z, fine=torch.ones_like(z, dtype=torch.bool), target=torch.full_like(z, float('nan'))))

def test_probability_pool_and_bce_remain_float32_inside_autocast():
    z = torch.tensor([[[[-2.0, 3.0]]]], requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = forecast_loss(z, batch(z, coarse_target=1.0))
    expected = forecast_loss(z, batch(z, coarse_target=1.0))
    assert actual.loss.dtype == torch.float32
    torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
    actual.loss.backward()
    assert torch.isfinite(z.grad).all()
