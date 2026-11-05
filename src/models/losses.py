"""Masked fine BCE and stable area-pooled native-footprint BCE.

All probability pooling and losses use float32, including under model autocast.
Statistics accumulate sums and counts so a partition is never scored by taking
an unweighted average of batch means. Unsupported metrics are reported as None.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class HorizonStatistics:
    """Additive BCE/Brier numerators and support counts, indexed by horizon.
    
        Fine support counts pixels; coarse support counts native CSM footprints.
        Aggregate these statistics across batches before calling ``report`` to get
        partition metrics weighted by valid support. Only BCE sums carry gradients;
        Brier sums are evaluation statistics against binary clear-sky labels.
        """

    # Each tensor has one entry per forecast horizon. Keep raw syms/counts so
    # adding batches weights every valid pixel or footprint equally.
    fine_sum: Tensor
    fine_count: Tensor
    coarse_sum: Tensor
    coarse_count: Tensor
    fine_brier_sum: Tensor
    coarse_brier_sum: Tensor

    def detached(self) -> HorizonStatistics:
        """Return statistics sharing tensor storage but detached from autograd.
        
        Use this before accumulating epoch metrics to avoid retaining each
        batch's computation graph. This is not an independent tensor copy.
        """
        # Retain reporting values without retaining the training graph.
        return HorizonStatistics(*(getattr(self, name).detach() for name in self.__dataclass_fields__))

    def __add__(self, other: HorizonStatistics) -> HorizonStatistics:
        """Combine matching horizons without averaging or detaching either operand."""
        return HorizonStatistics(*(getattr(self, name) + getattr(other, name)
                                   for name in self.__dataclass_fields__))

    def objective(self, coarse_weight: float = 0.25) -> tuple[Tensor, Tensor]:
        """Return the equally weighted mean over active horizons and their mask.
        
        Each horizon contributes mean fine BCE plus 'coarse_weight' times 
        mean footprint BCE. Missing sources contribute zero; horizons with no 
        weighted supervision are excluded.  The weight must be finite and
        nonnegative; zero disables coarse-only horizons. """

        if not math.isfinite(coarse_weight) or coarse_weight<0:
            raise ValueError("coarse_weight must be finite and nonnegative")
        # Normalize fine pixels and coarse footprints separately, then give each
        # supervised horizon equal weight. A missing source contributes zero;
        # coarse-only horizons are inactive when coarse_weight is zero.
        active = (self.fine_count > 0) | ((self.coarse_count > 0) & (coarse_weight > 0))
        terms = (self.fine_sum / self.fine_count.clamp_min(1)
                 + coarse_weight * self.coarse_sum / self.coarse_count.clamp_min(1))
        return (terms * active).sum() / active.sum().clamp_min(1), active

    def report(self, coarse_weight: float=0.25) -> dict[str, Any]:
        """Export raw statistics, per-horizon means and the objective as Python values."""
        loss, active = self.objective(coarse_weight)
        result = {name: getattr(self, name).detach().cpu().tolist()
                  for name in self.__dataclass_fields__}

        # Create a dictionary containing every dataclass field’s raw statistics.
        result.update(loss=float(loss.detach().cpu()), active_horizons=active.cpu().tolist())
        for name, sums, counts in (
            ("fine_bce", self.fine_sum, self.fine_count),
            ("coarse_bce", self.coarse_sum, self.coarse_count),
            ("fine_brier", self.fine_brier_sum, self.fine_count),
            ("coarse_brier", self.coarse_brier_sum, self.coarse_count),
        ):
            # No support means an undefined metric, rather than a perfect zero.
            result[name] = [s / n if n else None for s, n in
                            zip(sums.detach().cpu().tolist(), counts.detach().cpu().tolist())]
        return result

@dataclass
class ForecastLossResult:
    """Differentiable objective, additive metrics and optimizer-step eligibility.
    
    ''skip_update'' is true when no horizon has weighted supervision. Callers 
    should then skip the optimizer step, since even zero gradients can otherwise
    change parameters through momentum (memory of past gradients) or weight decay."""
    loss: Tensor
    statistics: HorizonStatistics
    skip_update: bool

def _segment_logsumexp(values: Tensor, rows: Tensor, nrows: int) -> Tensor:
    """Reduce log-values by row while preserving gradients through every overlap.

    ``rows`` assigns each one-dimensional value to a row in ``[0, nrows)``.
    Callers supply at least one finite contribution per row. A detached row
    maximum is only a numerical shift, so tied maxima do not alter gradients.

    values: [number_of_selected_maps, number_of_overlap_entries]
    rows: [number_of_overlap_entries]
    output: [number_of_selected_maps, nrows]
    """
    # Compute log(sum(exp(values))) separately for each native footprint.
    # Subtracting its maximum prevents overflow in the exponential.
    # Detach the stabilizing maximum to preserve  logsumexp's exact derivative.
    maxima = values.new_full((*values.shape[:-1], nrows), -torch.inf) # [:-1] means keep everything except the last dimension. The * unpacks the remaining shape size from (3,) to 3.

    # the trailing _ in scatter_reduce_ means the operation modifies the tensor in place. 
    # This is a common Pytorch naming convention.
    indices = rows.expand_as(values)
    maxima.scatter_reduce_(-1, indices, values.detach(), reduce="amax", include_self=True)
    sums = torch.zeros_like(maxima).scatter_add(-1, indices,
        torch.exp(values - maxima.gather(-1, indices)))
    return maxima + torch.log(sums)


def stable_probability_pool(logits: Tensor, row_indices: Tensor, col_indices: Tensor,
                            weights: Tensor, nrows:int) -> tuple[Tensor, Tensor]:
    """Return log clear/cloudy probabilities averaged over native footprints.

    Stable: it uses logarithms to keep calculations reliable when probabilities 
    are extremely close to zero or one.

    ``logits`` has shape ``[..., H, W]`` (or one flat map). The shared
    equally sized Coordinate-format (COO) arrays identify footprint rows, flattened pixel columns,
    and positive overlap_area / covered_footprint_area weights. Indices must be
    int64, in bounds, and weights must sum to one per row within tolerance.
    All tensors must already be on compatible devices.

    The two returned tensors have shape ``[..., nrows]`` and represent
    ``log(sum(w * sigmoid(z)))`` and ``log(sum(w * sigmoid(-z)))`` for each map.
    Float32 log-space reduction keeps BCE stable for extreme logits while preserving
    gradients to fine pixels. Averaging logits would change the model. Zero
    rows return two empty tensors without validating the unused geometry.

    logits: [M, H, W]
    row_indices: [E]
    col_indices: [E]
    weights: [E]

    nrows is the number of coarse footprints
    M is the number of forecast maps.
    E is the total number of overlaps.
    """

    if nrows == 0:
        shape = (*logits.shape[:-2], 0)
        return logits.new_empty(shape), logits.new_empty(shape)

    # COO geometry: rows identify native footprints; cols identify flattened
    # fine-grid pixels. Multiple entries describe each footprint's overlaps.
    flat = (logits.flatten(start_dim=-2) if logits.ndim >= 2 else logits).float() # e.g., from [M,H,W] to [M,H*W]

    if row_indices.dtype != torch.int64 or col_indices.dtype != torch.int64:
        raise ValueError("pooling indices must be int64")
    
    rows, cols, w = row_indices, col_indices, weights.float()
    if (rows.ndim != 1 or cols.shape != rows.shape or w.shape != rows.shape
            or not bool(torch.isfinite(w).all()) or bool((w <= 0).any())):
        raise ValueError("pooling requires equal one-dimensional positive finite COO arrays")
    
    if (bool((rows < 0).any()) or bool((rows >= nrows).any())
            or bool((cols < 0).any()) or bool((cols >= flat.shape[-1]).any())):
        raise ValueError("pooling COO index outside geometry")
    
    totals = w.new_zeros(nrows).index_add(0, rows, w)
    if not bool(torch.allclose(totals, torch.ones_like(totals), atol=2e-5, rtol=2e-5)):
        raise ValueError("every pooled row must have positive normalized weights")
    
    log_w = torch.log(w)

    # log(sigmoid(z)) and log(sigmoid(-z)) remain stable for extreme logits.
    # Pool both classes directly in log space, avoiding log(1 - rounded_p).
    return (_segment_logsumexp(log_w + F.logsigmoid(flat[..., cols]), rows, nrows),
            _segment_logsumexp(log_w + F.logsigmoid(-flat[..., cols]), rows, nrows))


def _forecast_loss(logits: Tensor, batch: Mapping[str, Any], coarse_weight: float
                   ) -> ForecastLossResult:
    """Accumulate fine pixels and fixed native rows under per-time source selection.

    The dataset validates and compacts geometry once. Here source masks must be
    exclusive for each sample/target time; pooling handles all selected coarse
    maps together. The caller has disabled autocast.
    
    logits is a tensor: [B, horizon, H, W] -> [B, 10, 400, 400]
    batch is a dictionary with:
    batch["target_fine"]: floats, [B, 10, 400, 400]
    batch["valid_fine"]: bools, [B, 10, 400, 400]
    batch["coarse_supervision"]: bools, [B, 10] (TRUE selects coarse supervision; FALSE selects fine supervision)
    batch["target_coarse"]:	floats, [B, 10, J]
    batch["valid_coarse"]:	bools, [B, 10, J]
    batch["fine_target_mode"]: "hard" or "soft"; defaults to "hard"
    (soft mode only) batch["target_fine_binary"]: binary evaluation labels, [B, 10, 400, 400]
    batch["native_geometry"]: shared dictionary containing:
        "fine_shape": (H, W)
        "row_indices": int64 tensor, [E]
        "col_indices": int64 tensor, [E]
        "weights": floating tensor, [E]

    E is the number of retained fine-pixel/footprint overlap entries.
    Geometry has no batch or horizon dimension because it is shared.
    J is the number of retained native CSM footprints
    """

    if logits.ndim != 4 or not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("forecast logits must be finite floating point with shape [B,horizon,H,W]")
    
    z = logits.float() # Ensures z is float32
    target = batch["target_fine"].float()
    fine = batch["valid_fine"]
    coarse_mode = batch["coarse_supervision"]

    if target.shape != z.shape or fine.shape != z.shape or coarse_mode.shape != z.shape[:2]:
        raise ValueError("fine target/mask and per-time supervision shapes must match logits")
    if fine.dtype != torch.bool or coarse_mode.dtype != torch.bool:
        raise ValueError("supervision masks must be boolean")

    y, coarse = batch["target_coarse"].float(), batch["valid_coarse"]
    if y.ndim != 3 or y.shape[:2] != z.shape[:2] or coarse.shape != y.shape or coarse.dtype != torch.bool:
        raise ValueError("coarse targets and boolean validity must have shape [B,horizon,J]")
    if bool((fine & coarse_mode[..., None, None]).any()) or bool((coarse & ~coarse_mode[..., None]).any()):
        raise ValueError("fine and coarse validity must follow the selected supervision source")
    if not bool(torch.isfinite(y[coarse]).all()) or bool(((y[coarse] != 0) & (y[coarse] != 1)).any()):
        raise ValueError("native CSM labels must be finite binary clear classes")
        
    selected = target[fine]
    if not bool(torch.isfinite(selected).all()) or bool(((selected < 0) | (selected > 1)).any()):
        raise ValueError("valid fine targets must be finite and in [0,1]")
    mode = batch.get("fine_target_mode", "hard")
    if mode not in ("hard", "soft"):
        raise ValueError("fine_target_mode must be hard or soft")
    if mode == "hard" and bool(((selected != 0) & (selected != 1)).any()):
        raise ValueError("hard fine targets must be binary clear classes")

    # Soft labels may train BCE, but Brier evaluation always uses the supplied
    # binary truth so its meaning stays consistent across target modes.
    binary = batch.get("target_fine_binary") if mode == "soft" else target
    if binary is None or binary.shape != target.shape:
        raise ValueError("soft targets require matching authoritative binary evaluation labels")
    
    truth = binary[fine]
    if not bool(torch.isfinite(truth).all()) or bool(((truth != 0) & (truth != 1)).any()):
        raise ValueError("binary evaluation labels must be finite 0/1 on valid pixels")

    # Replace invalid targets before BCE: masking a NaN loss afterwards would
    #  still risk NaN gradients. Reduce batch/spatial axes, retaining horizons.
    safe_target = torch.where(fine, target, torch.zeros_like(target))
    fine_losses = F.binary_cross_entropy_with_logits(z, safe_target, reduction="none")
    fine_sum = torch.where(fine, fine_losses, torch.zeros_like(fine_losses)).sum((0, 2, 3))
    fine_count = fine.sum((0, 2, 3)) # Reduce across all dimensions except horizons

    # Brier scores use continuous probabilities, with no thresholding or gradient.
    with torch.no_grad():
        safe_binary = torch.where(fine, binary.float(), 0)
        errors = (z.sigmoid() - safe_binary).square()
        fine_brier_sum = torch.where(fine, errors, 0).sum((0,2,3), dtype=torch.float64)
    horizons = z.shape[1]

    # A graph-connected zero keeps backward() valid even when there is no supervision.
    coarse_sum = (z * 0).sum((0,2,3)) # shape: [horizons]
    coarse_count = coarse.sum((0,2)) # shape: [horizons]
    coarse_brier_sum = torch.zeros(horizons, device=z.device, dtype=torch.float64) # shape: [horizons]

    # Find all maps (sample and horizon) that have at least one valid coarse footprint for supervision
    b, k = coarse.any(-1).nonzero(as_tuple=True)
    if b.numel():
        geometry = batch["native_geometry"]
        if geometry is None or tuple(geometry["fine_shape"]) != tuple(z.shape[-2:]):
            raise ValueError("coarse supervision requires native geometry matching the fine grid")
        
        log_p, log_q = stable_probability_pool(z[b, k], geometry["row_indices"],
            geometry["col_indices"], geometry["weights"], y.shape[-1])
        valid = coarse[b, k]
        # Each valid native footprint contributes once; missing QA never changes
        # the fixed operator or its original normalized area weights.
        safe_y = torch.where(valid, y[b, k], 0)
        losses = torch.where(safe_y == 1, -log_p, -log_q)
        coarse_sum = coarse_sum.index_add(0, k, torch.where(valid, losses, 0).sum(-1))
        with torch.no_grad():
            errors = (log_p.exp() - safe_y).square()
            coarse_brier_sum.index_add_(0, k, torch.where(valid, errors, 0).sum(-1, dtype=torch.float64))

    statistics = HorizonStatistics(fine_sum, fine_count, coarse_sum,
        coarse_count, fine_brier_sum, coarse_brier_sum)
    loss, active = statistics.objective(coarse_weight)

    # Callers should skip optimizer steps when the batch has no active horizon.
    return ForecastLossResult(loss, statistics, not bool(active.any()))
    

def forecast_loss(logits: Tensor, batch: Mapping[str, Any], *, coarse_weight: float = 0.25
                   ) -> ForecastLossResult:
    
    """Score clear-sky logits using fine labels and native-footprint supervision.

    Args:
        logits: Finite floating tensor ``[B, horizon, H, W]``; positive values
            indicate clear sky. Loss arithmetic uses float32 under autocast.
        batch: Collated dataset output on the logits' device. ``target_fine``,
            ``valid_fine`` match the logits' shape. ``coarse_supervision`` is
            boolean ``[B,horizon]`` and selects one source per sample/time. Hard
            targets are binary; soft targets lie in [0, 1] and require separate
            ``target_fine_binary`` truth for Brier evaluation.
            ``target_coarse`` / ``valid_coarse`` are ``[B,horizon,J]``.
            ``native_geometry`` is the shared, coverage-filtered overlap operator.
        coarse_weight: Finite, nonnegative multiplier for native BCE, default
            0.25. Fine and coarse means are normalized separately per horizon.

    Returns:
        A ``ForecastLossResult`` whose loss averages active horizons equally.
        Statistics retain separate BCE/Brier sums and support counts for full
        partition aggregation. Brier uses unthresholded probabilities and binary
        truth. With no weighted supervision, loss is a differentiable zero and
        ``skip_update`` is true. TV regularization has baseline weight zero and
        is omitted.

    Raises:
        ValueError: Logits, labels, masks or native geometry violate the contract.
    """

    # Keep pooling and BCE in float32 even when the model uses mixed precision, 
    # by overriding any outer autocast context.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return _forecast_loss(logits, batch, coarse_weight)


    
    






        




    

