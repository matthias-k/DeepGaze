"""Image-grouped training and evaluation of scanpath models.

DeepGaze III's training loop (``training.py`` with ``data.FixationDataset``) treats every fixation as
a sample and recomputes the frozen backbone for each one. Here all fixations of an image are processed
together: the per-image part (backbone, priority map) is evaluated once and the scanpath part for all
fixation histories of the image. The loss is the image-averaged negative log-likelihood, as in
DeepGaze III's scanpath training (``average='image'``).

A model plugs in through a ``ScanpathTask`` with two methods: ``image_context(item)`` computes the
per-image part and ``pre_log_density(context, x_hist, y_hist)`` the unnormalized log density for a
chunk of histories (see ``scanpath_utils.log_density_at`` for the normalization).
"""
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .scanpath_utils import mixture_full_log_density as _full_log_density
from .scanpath_utils import mixture_log_density_at as _log_density_at


@dataclass
class ImageFixations:
    """All scored fixations on one image, with their histories (most recent first)."""
    index: int
    image_size: tuple
    xs: torch.Tensor
    ys: torch.Tensor
    x_hist: torch.Tensor
    y_hist: torch.Tensor

    def __len__(self):
        return len(self.xs)


def history_arrays(fixations, included_fixations: Sequence[int]):
    """(N, len(included_fixations)) histories; entry -k is the k-th last previous fixation or NaN."""
    lengths = np.asarray(fixations.lengths, dtype=int)
    n = len(lengths)
    x_hist = np.full((n, len(included_fixations)), np.nan, dtype=np.float64)
    y_hist = np.full((n, len(included_fixations)), np.nan, dtype=np.float64)
    rows = np.arange(n)
    for column, offset in enumerate(included_fixations):
        position = lengths + offset  # offset is negative
        valid = position >= 0
        x_hist[rows[valid], column] = fixations.x_hist[rows[valid], position[valid]]
        y_hist[rows[valid], column] = fixations.y_hist[rows[valid], position[valid]]
    return x_hist, y_hist


def group_by_image(stimuli, fixations, included_fixations: Sequence[int], with_history: bool = True) -> List[ImageFixations]:
    """Group fixations by image.

    With ``with_history`` (scanpath models) only fixations with at least one previous fixation are
    scored. Without it (spatial models) all fixations are used and the histories are empty (zero
    columns). Works in O(N log N), so it also handles datasets with tens of millions of fixations.
    """
    if with_history:
        scored = fixations[fixations.lengths > 0]
        x_hist, y_hist = history_arrays(scored, included_fixations)
    else:
        scored = fixations
        x_hist = y_hist = None
    sizes = stimuli.sizes
    xs = np.asarray(scored.x_int, dtype=np.int64)
    ys = np.asarray(scored.y_int, dtype=np.int64)
    ns = np.asarray(scored.n, dtype=np.int64)
    order = np.argsort(ns, kind='stable')
    boundaries = np.flatnonzero(np.diff(ns[order])) + 1
    items = []
    for group in np.split(order, boundaries):
        if not len(group):
            continue
        n = int(ns[group[0]])
        height, width = sizes[n]
        gx, gy = xs[group], ys[group]
        if gx.min() < 0 or gy.min() < 0 or gx.max() >= width or gy.max() >= height:
            raise ValueError(f"fixation outside image {n} ({width}x{height})")
        if with_history:
            hx = torch.from_numpy(x_hist[group]).float()
            hy = torch.from_numpy(y_hist[group]).float()
        else:
            hx = hy = torch.zeros((len(group), 0))
        items.append(ImageFixations(index=n, image_size=(int(height), int(width)),
                                    xs=torch.from_numpy(gx), ys=torch.from_numpy(gy), x_hist=hx, y_hist=hy))
    return items


def bits_relative_to_uniform(log_density_nats: torch.Tensor, image_size) -> torch.Tensor:
    return (log_density_nats + math.log(image_size[0] * image_size[1])) / math.log(2)




def run_epoch(task, items: Sequence[ImageFixations], optimizer=None, chunk_size: int = 48,
              rng: Optional[np.random.RandomState] = None, device='cuda') -> Dict[int, np.ndarray]:
    """One pass over ``items``; trains if ``optimizer`` is given.

    Returns the per-fixation log-likelihoods (bits relative to a uniform prediction) per image index.
    """
    order = np.arange(len(items)) if rng is None else rng.permutation(len(items))
    results = {}
    training = optimizer is not None
    for position in order:
        item = items[position]
        n = len(item)
        with torch.set_grad_enabled(training):
            context = task.image_context(item)
            lls = []
            if getattr(task, 'history_independent', False):
                # one prediction for all fixations of the image
                pre = task.pre_log_density(context, item.x_hist[:1].to(device), item.y_hist[:1].to(device))
                pre = [p.expand(n, -1, -1) for p in pre] if isinstance(pre, (list, tuple)) else pre.expand(n, -1, -1)
                ll = _log_density_at(pre, item.image_size, item.ys.to(device), item.xs.to(device))
                if training:
                    (-ll.mean()).backward()
                lls.append(ll.detach())
                n_chunks = 0
            else:
                n_chunks = n
            for start in range(0, n_chunks, chunk_size):
                stop = min(start + chunk_size, n)
                pre = task.pre_log_density(context, item.x_hist[start:stop].to(device), item.y_hist[start:stop].to(device))
                ll = _log_density_at(pre, item.image_size, item.ys[start:stop].to(device), item.xs[start:stop].to(device))
                if training:
                    # image-averaged loss; the per-image part of the graph is shared by all chunks
                    (-ll.sum() / n).backward(retain_graph=stop < n)
                lls.append(ll.detach())
        if training:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        results[item.index] = bits_relative_to_uniform(torch.cat(lls), item.image_size).cpu().numpy()
    return results


@torch.no_grad()
def evaluate(task, items: Sequence[ImageFixations], chunk_size: int = 16, device='cuda') -> Dict[int, Dict[str, np.ndarray]]:
    """Per-fixation LL (bits relative to uniform), AUC and NSS of the conditional predictions.

    AUC and NSS follow pysaliency: all image pixels are negatives (ties count half) and NSS
    standardizes the predicted density with its mean and population standard deviation.
    """
    results = {}
    for item in items:
        context = task.image_context(item)
        per_metric = {'LL': [], 'AUC': [], 'NSS': []}
        for start in range(0, len(item), chunk_size):
            stop = min(start + chunk_size, len(item))
            ys = item.ys[start:stop].to(device)
            xs = item.xs[start:stop].to(device)
            x_hist, y_hist = item.x_hist[start:stop].to(device), item.y_hist[start:stop].to(device)
            if hasattr(task, 'log_density_maps'):  # tasks that produce maps on the original grid themselves
                maps = task.log_density_maps(context, x_hist, y_hist, item.image_size)
            else:
                maps = _full_log_density(task.pre_log_density(context, x_hist, y_hist), item.image_size)
            batch = torch.arange(len(ys), device=device)
            values = maps[batch, ys, xs]
            per_metric['LL'].append(bits_relative_to_uniform(values, item.image_size))
            flat = maps.flatten(1)
            per_metric['AUC'].append((flat < values[:, None]).double().mean(1) + 0.5 * (flat == values[:, None]).double().mean(1))
            density = flat.double().exp()
            per_metric['NSS'].append((values.double().exp() - density.mean(1)) / density.std(1, unbiased=False))
        results[item.index] = {k: torch.cat(v).cpu().numpy() for k, v in per_metric.items()}
    return results


def summarize(lls: Dict[int, np.ndarray], baseline: Optional[Dict[int, np.ndarray]] = None) -> Dict[str, float]:
    """Per-fixation and per-image averages of LL and, with a baseline, of information gain."""
    summary = {
        'LL_fixation': float(np.mean(np.concatenate(list(lls.values())))),
        'LL_image': float(np.mean([v.mean() for v in lls.values()])),
    }
    if baseline is not None:
        gains = {n: lls[n] - baseline[n] for n in lls}
        summary['IG_fixation'] = float(np.mean(np.concatenate(list(gains.values()))))
        summary['IG_image'] = float(np.mean([v.mean() for v in gains.values()]))
    return summary


def train(task, train_items, val_items, parameters, directory: str, lr: float, val_baseline=None,
          min_lr: float = 1e-6, patience: int = 2, max_epochs: int = 100, chunk_size: int = 48,
          milestones: Optional[Sequence[int]] = None, val_task=None,
          seed: int = 0, device='cuda', log: Callable[[str], None] = print,
          state_dict_fn: Optional[Callable[[], dict]] = None, load_state_fn: Optional[Callable[[dict], None]] = None):
    """Adam; the learning rate drops by 10x at ``milestones`` (epochs, as in train_deepgaze3.ipynb) or,
    without milestones, when the validation LL (image-averaged) stops improving for ``patience``
    epochs. Training stops once the learning rate falls below ``min_lr``. Keeps the best validation
    state (``best.pth``) and resumes from ``last.pth`` if the directory already holds a run.
    ``val_task`` evaluates the validation images if they need other loaders than the training images
    (same model).
    """
    os.makedirs(directory, exist_ok=True)
    val_task = val_task or task
    state_dict_fn = state_dict_fn or task.model.state_dict
    load_state_fn = load_state_fn or (lambda state: task.model.load_state_dict(state))
    optimizer = torch.optim.Adam(parameters, lr=lr)
    if milestones is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(milestones))
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=patience)
    history, epoch, best = [], 0, -np.inf
    last_path = os.path.join(directory, 'last.pth')
    if os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        load_state_fn(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        history, epoch, best = checkpoint['history'], checkpoint['epoch'], checkpoint['best']
        log(f"resumed from epoch {epoch}")

    rng = np.random.RandomState(seed + epoch)
    while epoch < max_epochs and optimizer.param_groups[0]['lr'] >= min_lr:
        started = time.time()
        task.model.train()
        train_lls = run_epoch(task, train_items, optimizer=optimizer, chunk_size=chunk_size, rng=rng, device=device)
        trained = time.time()
        task.model.eval()
        val_lls = run_epoch(val_task, val_items, chunk_size=chunk_size, device=device)
        epoch += 1
        row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'],
               'train': summarize(train_lls), 'val': summarize(val_lls, val_baseline),
               'seconds': {'train': round(trained - started, 1), 'val': round(time.time() - trained, 1)}}
        history.append(row)
        score = row['val']['LL_image']
        if score > best:
            best = score
            torch.save(state_dict_fn(), os.path.join(directory, 'best.pth'))
        if milestones is not None:
            scheduler.step()
        else:
            scheduler.step(score)
        torch.save({'model': state_dict_fn(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'history': history, 'epoch': epoch, 'best': best}, last_path)
        with open(os.path.join(directory, 'history.json'), 'w') as f:
            json.dump(history, f, indent=1)
        log(json.dumps(row))
    return history
