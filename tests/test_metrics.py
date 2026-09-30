import numpy as np
import pytest
import torch

from deepgaze_pytorch.data import FixationMaskTransform
from deepgaze_pytorch.metrics import _general_auc, auc, log_likelihood, nss


H, W = 12, 16
# per image: (xs, ys); the first image has two fixations on the same pixel (5, 7)
FIXATIONS = [([1, 5, 5, 9], [2, 7, 7, 3]), ([0, 15], [11, 0])]


def _log_density(seed):
    g = torch.Generator().manual_seed(seed)
    logits = 3 * torch.randn(H, W, generator=g)
    return logits - logits.logsumexp(dim=(0, 1))


def _mask(xs, ys, sparse):
    item = {'image': np.zeros((3, H, W)), 'x': np.array(xs), 'y': np.array(ys)}
    return FixationMaskTransform(sparse=sparse)(item)['fixation_mask']


def _batch(sparse=False):
    log_density = torch.stack([_log_density(0), _log_density(1)])
    masks = torch.stack([_mask(xs, ys, sparse) for xs, ys in FIXATIONS])
    weights = torch.ones(len(FIXATIONS))
    return log_density, masks, weights


def _reference_ll(log_density, xs, ys):
    values = log_density.numpy().astype(np.float64)
    return (np.mean([values[y, x] for x, y in zip(xs, ys)]) + np.log(H * W)) / np.log(2)


def _reference_nss(log_density, xs, ys):
    density = np.exp(log_density.numpy().astype(np.float64))
    saliency_map = (density - density.mean()) / density.std(ddof=1)
    return np.mean([saliency_map[y, x] for x, y in zip(xs, ys)])


def _reference_auc(log_density, xs, ys):
    values = log_density.numpy().astype(np.float64)
    negatives = values.flatten()
    # AUC over a set of positives is the mean of the single-positive AUCs
    return np.mean([_general_auc(np.array([values[y, x]]), negatives) for x, y in zip(xs, ys)])


@pytest.mark.parametrize('metric, reference', [
    (log_likelihood, _reference_ll),
    (nss, _reference_nss),
    (auc, _reference_auc),
])
def test_metric_matches_reference(metric, reference):
    log_density, masks, weights = _batch()
    expected = np.mean([reference(log_density[i], xs, ys) for i, (xs, ys) in enumerate(FIXATIONS)])
    assert metric(log_density, masks, weights=weights).item() == pytest.approx(expected, rel=1e-5)


def test_nss_uses_std_and_mean_in_the_right_order():
    # regression: torch.std_mean returns (std, mean); swapping them gave (p - std) / mean
    log_density, masks, weights = _batch()
    density = torch.exp(log_density)
    std, mean = torch.std_mean(density, dim=(-1, -2), keepdim=True)
    swapped = torch.mean(torch.sum((density - std) / mean * masks, dim=(-1, -2)) / masks.sum(dim=(-1, -2)))
    assert not np.isclose(nss(log_density, masks, weights=weights).item(), swapped.item())


def test_auc_counts_repeated_fixations():
    log_density = _log_density(0)[np.newaxis]
    weights = torch.ones(1)
    with_repeat = auc(log_density, _mask([5, 5, 9], [7, 7, 3], sparse=False)[np.newaxis], weights=weights).item()
    without_repeat = auc(log_density, _mask([5, 9], [7, 3], sparse=False)[np.newaxis], weights=weights).item()
    assert with_repeat == pytest.approx(_reference_auc(log_density[0], [5, 5, 9], [7, 7, 3]))
    assert with_repeat != pytest.approx(without_repeat)


@pytest.mark.parametrize('metric', [log_likelihood, nss, auc])
def test_sparse_masks_give_same_result_as_dense(metric):
    # FixationMaskTransform(sparse=True) is the default; metrics used to crash on it
    log_density, dense_masks, weights = _batch(sparse=False)
    _, sparse_masks, _ = _batch(sparse=True)
    assert sparse_masks.is_sparse
    assert metric(log_density, sparse_masks, weights=weights).item() == pytest.approx(
        metric(log_density, dense_masks, weights=weights).item())


@pytest.mark.parametrize('metric', [log_likelihood, nss, auc])
def test_channel_dimension_does_not_mix_images(metric):
    # DeepGaze IIE / III mixtures return (B, 1, H, W); this used to broadcast to (B, B, H, W)
    log_density, masks, weights = _batch()
    assert metric(log_density[:, np.newaxis], masks, weights=weights).item() == pytest.approx(
        metric(log_density, masks, weights=weights).item())


def test_mismatched_shapes_raise():
    log_density, masks, weights = _batch()
    with pytest.raises(ValueError):
        log_likelihood(log_density[:, :-1], masks, weights=weights)
