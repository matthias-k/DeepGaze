"""Helpers for scoring and training scanpath models without full-resolution maps per fixation.

DeepGaze finalizers compute an unnormalized log density at a reduced resolution, upsample it to the
image size with nearest-neighbour interpolation and normalize it. For many fixations on the same
image, materializing a full-resolution map per fixation is expensive. The functions here compute
exactly the same normalized log density values at the fixation pixels directly from the reduced map.
"""
import math

import torch


def nearest_source_index(dst_size, src_size, device=None):
    """Source index that ``F.interpolate(mode='nearest', size=...)`` uses for every output index.

    PyTorch computes ``floor(dst_index * (float(src_size) / dst_size))`` in float32; doing the same
    here keeps the indices identical at cell boundaries.
    """
    scale = torch.tensor(src_size, dtype=torch.float32) / torch.tensor(dst_size, dtype=torch.float32)
    index = torch.floor(torch.arange(dst_size, dtype=torch.float32) * scale).long()
    return index.clamp(max=src_size - 1).to(device)


def _upsampling_log_counts(pre, image_size):
    height, width = image_size
    iy = nearest_source_index(height, pre.shape[-2], device=pre.device)
    ix = nearest_source_index(width, pre.shape[-1], device=pre.device)
    log_cy = torch.log(torch.bincount(iy, minlength=pre.shape[-2]).to(pre.dtype))
    log_cx = torch.log(torch.bincount(ix, minlength=pre.shape[-1]).to(pre.dtype))
    return iy, ix, log_cy, log_cx


def log_normalizer(pre, image_size):
    """log of the sum of exp(upsampled map) over all image pixels, per item: (B, h, w) -> (B,)."""
    _, _, log_cy, log_cx = _upsampling_log_counts(pre, image_size)
    return torch.logsumexp(pre + log_cy[:, None] + log_cx[None, :], dim=(-2, -1))


def log_density_at(pre, image_size, ys, xs):
    """Normalized full-resolution log density at integer pixel positions.

    Equivalent to upsampling ``pre`` (B, h, w) to ``image_size`` with nearest-neighbour
    interpolation, normalizing each map and reading ``[b, ys[b], xs[b]]``. A single map (B = 1) is
    read at all positions.
    """
    if pre.shape[0] not in (1, len(ys)):
        raise ValueError(f"{pre.shape[0]} maps for {len(ys)} positions")
    iy, ix, log_cy, log_cx = _upsampling_log_counts(pre, image_size)
    log_z = torch.logsumexp(pre + log_cy[:, None] + log_cx[None, :], dim=(-2, -1))
    batch = torch.arange(pre.shape[0], device=pre.device)
    return pre[batch, iy[ys], ix[xs]] - log_z


def full_log_density(pre, image_size):
    """Normalized full-resolution log density maps (B, H, W) from the reduced map."""
    iy, ix, _, _ = _upsampling_log_counts(pre, image_size)
    full = pre[:, iy][:, :, ix]
    return full - torch.logsumexp(full, dim=(-2, -1), keepdim=True)


def encode_history_dva(x_hist, y_hist, image_size, readout_shape, pixel_per_dva):
    """Scanpath features in degrees of visual angle at the centres of the readout cells.

    Returns (B, 3 * F, h, w): the x offsets of all F history fixations, then the y offsets, then
    the distances -- the channel layout ``FlexibleScanpathHistoryEncoding`` expects. Missing
    history fixations (NaN) stay NaN, which disables the corresponding convolution.
    """
    height, width = image_size
    rows, cols = readout_shape
    device = x_hist.device
    xs = (torch.arange(cols, device=device, dtype=torch.float32) + 0.5) * (width / cols)
    ys = (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) * (height / rows)
    dx = (xs[None, None, None, :] - x_hist[:, :, None, None].float()) / pixel_per_dva
    dy = (ys[None, None, :, None] - y_hist[:, :, None, None].float()) / pixel_per_dva
    dx, dy = torch.broadcast_tensors(dx, dy)
    return torch.cat((dx, dy, torch.sqrt(dx ** 2 + dy ** 2)), dim=1)


def encode_history_pixels(x_hist, y_hist, image_size, readout_shape):
    """DeepGaze III's scanpath features, computed directly at the readout resolution.

    ``modules.encode_scanpath_features`` builds (x offset, y offset, distance) maps in pixels at full
    resolution and ``DeepGazeIII.forward`` downsamples them with nearest-neighbour interpolation. That
    reads the full-resolution maps at the pixels ``nearest_source_index`` returns, so evaluating the
    maps only there gives identical values without the full-resolution tensors.
    """
    height, width = image_size
    rows, cols = readout_shape
    device = x_hist.device
    xs = nearest_source_index(cols, width, device=device).to(torch.float32)
    ys = nearest_source_index(rows, height, device=device).to(torch.float32)
    dx = xs[None, None, None, :] - x_hist[:, :, None, None].float()
    dy = ys[None, None, :, None] - y_hist[:, :, None, None].float()
    dx, dy = torch.broadcast_tensors(dx, dy)
    return torch.cat((dx, dy, torch.sqrt(dx ** 2 + dy ** 2)), dim=1)


def mixture_log_density_at(pre, image_size, ys, xs):
    """``log_density_at`` for one map or an equal-weight mixture of several maps (list)."""
    if isinstance(pre, (list, tuple)):
        stacked = torch.stack([log_density_at(p, image_size, ys, xs) for p in pre])
        return torch.logsumexp(stacked, dim=0) - math.log(len(pre))
    return log_density_at(pre, image_size, ys, xs)


def mixture_full_log_density(pre, image_size):
    """``full_log_density`` for one map or an equal-weight mixture of several maps (list)."""
    if isinstance(pre, (list, tuple)):
        stacked = torch.stack([full_log_density(p, image_size) for p in pre])
        return torch.logsumexp(stacked, dim=0) - math.log(len(pre))
    return full_log_density(pre, image_size)
