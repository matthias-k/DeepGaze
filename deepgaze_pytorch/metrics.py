import numpy as np
from pysaliency.roc import general_roc
from pysaliency.numba_utils import auc_for_one_positive
import torch


def _general_auc(positives, negatives):
    if len(positives) == 1:
        return auc_for_one_positive(positives[0], negatives)
    else:
        return general_roc(positives, negatives)[0]


def log_likelihood(log_density, fixation_mask, weights=None):
    #if weights is None:
    #    weights = torch.ones(log_density.shape[0])

    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()

    if isinstance(fixation_mask, torch.sparse.IntTensor):
        dense_mask = fixation_mask.to_dense()
    else:
        dense_mask = fixation_mask
    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)
    ll = torch.mean(
        weights * torch.sum(log_density * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
    )
    return (ll + np.log(log_density.shape[-1] * log_density.shape[-2])) / np.log(2)


def nss(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()
    if isinstance(fixation_mask, torch.sparse.IntTensor):
        dense_mask = fixation_mask.to_dense()
    else:
        dense_mask = fixation_mask

    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)

    density = torch.exp(log_density)
    mean, std = torch.std_mean(density, dim=(-1, -2), keepdim=True)
    saliency_map = (density - mean) / std

    nss = torch.mean(
        weights * torch.sum(saliency_map * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
    )
    return nss


def auc(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights / weights.sum()

    # TODO: This doesn't account for multiple fixations in the same location!
    def image_auc(log_density, fixation_mask):
        if isinstance(fixation_mask, torch.sparse.IntTensor):
            dense_mask = fixation_mask.to_dense()
        else:
            dense_mask = fixation_mask

        positives = torch.masked_select(log_density, dense_mask.type(torch.bool)).detach().cpu().numpy().astype(np.float64)
        negatives = log_density.flatten().detach().cpu().numpy().astype(np.float64)

        auc = _general_auc(positives, negatives)

        return torch.tensor(auc)

    return torch.mean(weights.cpu() * torch.tensor([
        image_auc(log_density[i], fixation_mask[i]) for i in range(log_density.shape[0])
    ]))
