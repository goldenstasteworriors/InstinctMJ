"""Additive binary-calibration statistics; empty/absent classes stay explicit."""

import torch


@torch.no_grad()
def binary_calibration(probability, target, valid, bins=10):
    """Mask whole environments before flattening bodies; sums can be pooled safely.

    Precision/recall are None when their denominator is zero. Returning counts
    and reliability-bin sums prevents averages of per-window ratios from hiding
    rare torso failures or unequal numbers of surviving trajectories.
    """
    if bins < 1 or valid.ndim != 1 or probability.shape != target.shape or probability.shape[0] != valid.numel():
        raise ValueError("Invalid calibration shape or bin count")
    probability, target = probability[valid.bool()].flatten(), target[valid.bool()].flatten().float()
    if not torch.isfinite(probability).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite calibration input")
    if ((probability < 0) | (probability > 1)).any() or ((target != 0) & (target != 1)).any():
        raise ValueError("Expected probabilities in [0,1] and binary labels")
    truth, prediction = target.bool(), probability >= 0.5
    count = probability.numel()
    tp, fp = int((prediction & truth).sum()), int((prediction & ~truth).sum())
    fn, tn = int((~prediction & truth).sum()), int((~prediction & ~truth).sum())
    index = (probability * bins).long().clamp(max=bins - 1)
    bin_count = torch.bincount(index, minlength=bins)
    bin_prob = torch.zeros(bins, device=probability.device).scatter_add_(0, index, probability.float())
    bin_positive = torch.zeros_like(bin_prob).scatter_add_(0, index, target.float())
    brier_sum = float((probability - target).square().sum())
    eps = torch.finfo(torch.float32).eps
    clipped = probability.float().clamp(eps, 1 - eps)
    logloss_sum = float(-(target * clipped.log() + (1 - target) * torch.log1p(-clipped)).sum())
    return dict(count=count, positive_count=tp + fn, predicted_positive_count=tp + fp,
                tp=tp, fp=fp, fn=fn, tn=tn,
                brier_sum=brier_sum, brier=brier_sum / count if count else None,
                logloss_sum=logloss_sum, logloss=logloss_sum / count if count else None,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                bin_count=bin_count.tolist(), bin_probability_sum=bin_prob.tolist(),
                bin_positive_count=bin_positive.tolist())
