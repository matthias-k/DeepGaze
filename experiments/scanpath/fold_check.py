"""Which MIT1003 images did the released models train on?

A model predicts its training images better than held-out ones, so per-fold log-likelihoods reveal
held-out folds:

- DeepGaze III component c should be worse than the other components on its test fold c and its
  validation fold c - 1 (train_deepgaze3.ipynb with pysaliency's default splits).
- DeepGaze MSDB used 9 of the 10 folds for training (Kümmerer et al. 2025, appendix I) without
  naming the held-out one. Relative to the DeepGaze III component that did not see a fold, MSDB
  should do worst on its own held-out fold.

Requires ``cache_msdb_saliency.py mit1003``. Writes runs/fold_check.json.

    python experiments/scanpath/fold_check.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root: deepgaze_pytorch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from cache_msdb_saliency import saliency_loader  # noqa: E402
from deepgaze_pytorch import DeepGazeIII  # noqa: E402
from deepgaze_pytorch.deepgaze3_msdb import DeepGazeIIIMSDB  # noqa: E402
from deepgaze_pytorch.deepgazemsdb import MSDBDataset  # noqa: E402
from deepgaze_pytorch.scanpath_tasks import DeepGazeIIITask, MSDBScanpathTask  # noqa: E402
from deepgaze_pytorch.scanpath_training import bits_relative_to_uniform, run_epoch  # noqa: E402
from deepgaze_pytorch.scanpath_utils import log_density_at  # noqa: E402


@torch.no_grad()
def component_lls(task, items, device, chunk_size=48):
    """Image-averaged LL (bits) of every mixture component: {image: array(components)}."""
    result = {}
    for item in items:
        context = task.image_context(item)
        per_component = []
        for start in range(0, len(item), chunk_size):
            stop = min(start + chunk_size, len(item))
            pres = task.pre_log_density(context, item.x_hist[start:stop].to(device), item.y_hist[start:stop].to(device))
            ys, xs = item.ys[start:stop].to(device), item.xs[start:stop].to(device)
            per_component.append(torch.stack([log_density_at(p, item.image_size, ys, xs) for p in pres]))
        lls = bits_relative_to_uniform(torch.cat(per_component, dim=1), item.image_size)
        result[item.index] = lls.mean(dim=1).cpu().numpy()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, help='only images with index < N (smoke test)')
    args = parser.parse_args()
    device = torch.device(args.device)
    stimuli, scanpaths = common.load_mit1003()
    items = [item for item in common.items_by_index(stimuli, scanpaths).values()
             if args.limit is None or item.index < args.limit]
    load_centerbias = common.centerbias_cache('mit1003', stimuli, lambda: common.mit1003_centerbias_model(stimuli, scanpaths))
    folds = common.fold_indices(len(stimuli))
    fold_of = {n: f for f, members in enumerate(folds) for n in members}

    dg3 = DeepGazeIII(pretrained=True).to(device).eval()
    dg3_lls = component_lls(DeepGazeIIITask(dg3, common.image_loader(stimuli, device), lambda n: load_centerbias(n, device)),
                            items, device)
    del dg3
    torch.cuda.empty_cache()

    msdb = DeepGazeIIIMSDB(pretrained_msdb=True, with_backbone=False).to(device).eval()  # untrained head = MSDB
    msdb_task = MSDBScanpathTask(msdb, lambda n: load_centerbias(n, device), common.MIT1003_PIXEL_PER_DVA,
                                 MSDBDataset.MIT1003, saliency_maps=saliency_loader('mit1003', device))
    with torch.no_grad():
        msdb_lls = {n: v.mean() for n, v in run_epoch(msdb_task, items, device=device).items()}

    n_folds = len(folds)
    # matrix[c, f]: mean over images of fold f of (component c minus the mean of all components)
    relative = np.zeros((n_folds, n_folds))
    for f in range(n_folds):
        members = [n for n in dg3_lls if fold_of[n] == f]
        if not members:
            relative[:, f] = np.nan
            continue
        values = np.stack([dg3_lls[n] for n in members])  # images x components
        relative[:, f] = (values - values.mean(axis=1, keepdims=True)).mean(axis=0)
    # MSDB relative to the component that held out each fold (component f on fold f)
    msdb_vs_heldout = []
    for f in range(n_folds):
        diff = np.array([msdb_lls[n] - dg3_lls[n][f] for n in dg3_lls if fold_of[n] == f])
        if len(diff) < 2:
            continue
        msdb_vs_heldout.append({'fold': f, 'mean': float(diff.mean()), 'sem': float(diff.std(ddof=1) / np.sqrt(len(diff))),
                                'images': len(diff)})
    result = {
        'dg3_component_minus_mean_by_fold': relative.round(4).tolist(),
        'dg3_worst_fold_per_component': np.nanargmin(relative, axis=1).tolist(),
        'msdb_minus_heldout_dg3_component_by_fold': msdb_vs_heldout,
    }
    common.RUNS.mkdir(parents=True, exist_ok=True)
    with open(common.RUNS / 'fold_check.json', 'w') as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
