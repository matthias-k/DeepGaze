import numpy as np
from pysaliency.roc import general_roc
from pysaliency.numba_utils import auc_for_one_positive
import torch


def _general_auc(positives, negatives):
    if len(positives) == 1:
        return auc_for_one_positive(positives[0], negatives)
    else:
        return general_roc(positives, negatives)[0]


def _dense_mask(fixation_mask):
    if fixation_mask.is_sparse:
        return fixation_mask.to_dense()
    return fixation_mask


def _match_mask_shape(log_density, dense_mask):
    # DeepGaze IIE/III mixtures return (B, 1, H, W); without dropping the channel it would
    # broadcast against the (B, H, W) mask to (B, B, H, W) and mix up images within a batch.
    if log_density.dim() == dense_mask.dim() + 1 and log_density.shape[1] == 1:
        log_density = log_density[:, 0]
    if log_density.shape != dense_mask.shape:
        raise ValueError(
            f"log density shape {tuple(log_density.shape)} does not match "
            f"fixation mask shape {tuple(dense_mask.shape)}"
        )
    return log_density


def log_likelihood(log_density, fixation_mask, weights=None):
    #if weights is None:
    #    weights = torch.ones(log_density.shape[0])

    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()

    dense_mask = _dense_mask(fixation_mask)
    log_density = _match_mask_shape(log_density, dense_mask)
    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)
    ll = torch.mean(
        weights * torch.sum(log_density * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
    )
    return (ll + np.log(log_density.shape[-1] * log_density.shape[-2])) / np.log(2)


def nss(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()
    dense_mask = _dense_mask(fixation_mask)
    log_density = _match_mask_shape(log_density, dense_mask)

    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)

    density = torch.exp(log_density)
    # torch.std_mean returns (std, mean)
    std, mean = torch.std_mean(density, dim=(-1, -2), keepdim=True)
    saliency_map = (density - mean) / std

    nss = torch.mean(
        weights * torch.sum(saliency_map * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
    )
    return nss


def auc(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights / weights.sum()
    dense_mask = _dense_mask(fixation_mask)
    log_density = _match_mask_shape(log_density, dense_mask)

    def image_auc(log_density, fixation_counts):
        log_density = log_density.detach().cpu().numpy().astype(np.float64)
        fixation_counts = fixation_counts.detach().cpu().numpy().astype(np.int64)

        fixated = fixation_counts > 0
        # every fixation is a positive, also when several fixations fall on the same pixel
        positives = np.repeat(log_density[fixated], fixation_counts[fixated])
        negatives = log_density.flatten()

        return _general_auc(positives, negatives)

    return torch.mean(weights.cpu() * torch.tensor([
        image_auc(log_density[i], dense_mask[i]) for i in range(log_density.shape[0])
    ]))
