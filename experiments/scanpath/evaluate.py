"""Evaluate scanpath models on held-out data and compare them with paired statistics.

Datasets:
  mit1003_fold<k>  test fold k of MIT1003 (original, unstretched images). Only models that did not
                   train on fold k may be evaluated here, e.g. component k of the released DeepGaze III.
  osie             OSIE (Xu et al. 2014), used by none of the models for training.

Models:
  centerbias        the dataset's center bias (IG baseline)
  msdb_spatial      DeepGaze MSDB (no history)
  dg3msdb           DeepGaze III scanpath part trained on MSDB (runs/<run>/fold<k>/best.pth)
  dg3_component<k>  one component (cross-validation fold) of the released DeepGaze III
  dg3_mixture       the released DeepGaze III (all 10 folds; not held out on any MIT1003 fold)
  densenet_<variant>  DeepGaze III retrained by train_densenet.py on ``stretched`` or ``original``
                    MIT1003 (on mit1003_fold<k> the fold-k model, on OSIE the ``--densenet-fold`` model)

DeepGaze III models always see images at their training resolution (35 pixels per degree); on OSIE
(24 pixels per degree) images are upscaled for them and their predictions mapped back to the original
pixels. Every model gets the same center-bias input and is scored on the same fixations: all fixations
except the first of each scanpath, conditioned on the true previous fixations.

    python experiments/scanpath/evaluate.py osie --models centerbias msdb_spatial dg3msdb dg3_component0
    python experiments/scanpath/evaluate.py osie --compare dg3msdb dg3_component0
    python experiments/scanpath/evaluate.py mit1003_fold0,mit1003_fold1 --compare densenet_original densenet_stretched
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from cache_msdb_saliency import saliency_loader  # noqa: E402
from train_msdb_scanpath import load_head  # noqa: E402
from deepgaze_pytorch.deepgaze3_msdb import DeepGazeIIIMSDB  # noqa: E402
from deepgaze_pytorch.deepgazemsdb import MSDBDataset  # noqa: E402
from deepgaze_pytorch.scanpath_tasks import DeepGazeIIITask, MSDBScanpathTask, RescaledTask  # noqa: E402
from deepgaze_pytorch.scanpath_training import evaluate  # noqa: E402

METRICS = ('LL', 'AUC', 'NSS')


class CenterBiasTask:
    """The center bias as a (history-independent) prediction."""

    def __init__(self, load_centerbias):
        self.load_centerbias = load_centerbias
        self.model = None

    def image_context(self, item):
        return self.load_centerbias(item.index)

    def pre_log_density(self, context, x_hist, y_hist):
        return context.expand(x_hist.shape[0], -1, -1)


def load_dataset(name):
    """-> (stimuli, items, load_centerbias(n, device), pixel_per_dva, msdb_dataset, cache name)"""
    if name.startswith('mit1003_fold'):
        fold = int(name[len('mit1003_fold'):])
        stimuli, scanpaths = common.load_mit1003()
        items = common.items_by_index(stimuli, scanpaths)
        _, _, test_idx = common.split_indices(len(stimuli), fold)
        items = [items[n] for n in test_idx if n in items]
        load_centerbias = common.centerbias_cache('mit1003', stimuli, common.mit1003_centerbias_model(stimuli, scanpaths))
        return stimuli, items, load_centerbias, common.MIT1003_PIXEL_PER_DVA, MSDBDataset.MIT1003, 'mit1003'
    if name == 'osie':
        stimuli, scanpaths = common.load_osie()
        items = list(common.items_by_index(stimuli, scanpaths).values())
        load_centerbias = common.centerbias_cache('osie', stimuli, common.osie_centerbias_model(stimuli, scanpaths))
        return stimuli, items, load_centerbias, common.OSIE_PIXEL_PER_DVA, None, 'osie'
    raise ValueError(name)


def deepgaze3_task(model, stimuli, load_centerbias, pixel_per_dva, device, components):
    """Released DeepGaze III at its training resolution (35 pixels per degree)."""
    scale = common.MIT1003_PIXEL_PER_DVA / pixel_per_dva
    load_image = common.image_loader(stimuli, device=device)
    if scale == 1:
        return DeepGazeIIITask(model, load_image, lambda n: load_centerbias(n, device), components=components)

    def scaled_size(n):
        height, width = stimuli.sizes[n]
        return round(height * scale), round(width * scale)

    def load_scaled_image(n):
        return F.interpolate(load_image(n), size=scaled_size(n), mode='bilinear', align_corners=False)

    def load_scaled_centerbias(n):
        density = F.interpolate(load_centerbias(n, device).exp()[:, None], size=scaled_size(n), mode='bilinear',
                                align_corners=False)[:, 0]
        return torch.log(density) - torch.log(density.sum())

    inner = DeepGazeIIITask(model, load_scaled_image, load_scaled_centerbias, components=components)
    height, width = stimuli.sizes[0]
    sizes = {tuple(s) for s in stimuli.sizes}
    if len(sizes) != 1:
        raise ValueError("rescaled evaluation assumes one image size per dataset")
    new_height, new_width = scaled_size(0)
    return RescaledTask(inner, new_height / height, new_width / width)


def build_task(model_name, stimuli, load_centerbias, pixel_per_dva, msdb_dataset, cache_name, device, run,
               densenet_fold):
    if model_name == 'centerbias':
        return CenterBiasTask(lambda n: load_centerbias(n, device))
    if model_name in ('msdb_spatial', 'dg3msdb'):
        model = DeepGazeIIIMSDB(pretrained_msdb=True, with_backbone=False).to(device).eval()
        if model_name == 'dg3msdb':
            load_head(model, torch.load(common.RUNS / run / 'best.pth', map_location=device))
        return MSDBScanpathTask(model, lambda n: load_centerbias(n, device), pixel_per_dva, msdb_dataset,
                                saliency_maps=saliency_loader(cache_name, device))
    match = re.fullmatch(r'densenet_(stretched|original)', model_name)
    if match:
        import train_densenet
        model = train_densenet.build_model(scanpath=True, downsample=2).to(device).eval()
        checkpoint = train_densenet.run_dir(match.group(1), f'fold{densenet_fold}', 'scanpath_full') / 'best.pth'
        train_densenet.load_trainable(model, torch.load(checkpoint, map_location=device))
        return deepgaze3_task(model, stimuli, load_centerbias, pixel_per_dva, device, None)
    match = re.fullmatch(r'dg3_component(\d+)|dg3_mixture', model_name)
    if match:
        from deepgaze_pytorch import DeepGazeIII
        model = DeepGazeIII(pretrained=True).to(device).eval()
        components = None if model_name == 'dg3_mixture' else [int(match.group(1))]
        return deepgaze3_task(model, stimuli, load_centerbias, pixel_per_dva, device, components)
    raise ValueError(model_name)


def results_path(dataset, model_name):
    return common.RUNS / 'eval' / dataset / f'{model_name}.npz'


def save_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f'{n}/{metric}': values[metric] for n, values in results.items() for metric in METRICS}
    np.savez(path, **arrays)


def load_results(path):
    data = np.load(path)
    results = {}
    for key in data.files:
        n, metric = key.split('/')
        results.setdefault(int(n), {})[metric] = data[key]
    return results


def summary(results):
    return {metric: {'per_fixation': float(np.mean(np.concatenate([r[metric] for r in results.values()]))),
                     'per_image': float(np.mean([r[metric].mean() for r in results.values()]))}
            for metric in METRICS} | {'images': len(results),
                                      'fixations': int(sum(len(r['LL']) for r in results.values()))}


def compare(a, b, n_boot=10000, seed=0):
    """Paired comparison a - b over images (fixations within an image are not independent)."""
    common_images = sorted(set(a) & set(b))
    if len(common_images) != len(a) or len(common_images) != len(b):
        raise ValueError("models were evaluated on different images")
    rng = np.random.RandomState(seed)
    out = {}
    for metric in METRICS:
        diff = np.array([a[n][metric].mean() - b[n][metric].mean() for n in common_images])
        per_fixation = float(np.mean(np.concatenate([a[n][metric] - b[n][metric] for n in common_images])))
        boot = diff[rng.randint(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
        out[metric] = {
            'mean_diff_per_image': float(diff.mean()),
            'ci95_per_image': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            'mean_diff_per_fixation': per_fixation,
            'images_better': int((diff > 0).sum()), 'images': len(diff),
            'p_ttest': float(stats.ttest_rel([a[n][metric].mean() for n in common_images],
                                             [b[n][metric].mean() for n in common_images]).pvalue),
            'p_wilcoxon': float(stats.wilcoxon(diff).pvalue),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--models', nargs='*', default=[])
    parser.add_argument('--compare', nargs=2, metavar=('A', 'B'))
    parser.add_argument('--run', default='msdb_scanpath/fold0', help='training run of dg3msdb')
    parser.add_argument('--densenet-fold', type=int, help='fold of the densenet_* models (default: the test fold, 0 on OSIE)')
    parser.add_argument('--chunk-size', type=int, default=16)
    args = parser.parse_args()

    if args.models:
        device = torch.device('cuda')
        stimuli, items, load_centerbias, ppd, msdb_dataset, cache_name = load_dataset(args.dataset)
        densenet_fold = args.densenet_fold
        if densenet_fold is None:
            densenet_fold = int(args.dataset[len('mit1003_fold'):]) if args.dataset.startswith('mit1003_fold') else 0
        for model_name in args.models:
            task = build_task(model_name, stimuli, load_centerbias, ppd, msdb_dataset, cache_name, device, args.run,
                              densenet_fold)
            results = evaluate(task, items, chunk_size=args.chunk_size, device=device)
            save_results(results_path(args.dataset, model_name), results)
            print(json.dumps({'dataset': args.dataset, 'model': model_name, **summary(results)}), flush=True)
            del task
            torch.cuda.empty_cache()

    if args.compare:
        # several datasets (e.g. the test folds of a cross-validation) are pooled
        a, b = ({n: r for dataset in args.dataset.split(',') for n, r in load_results(results_path(dataset, name)).items()}
                for name in args.compare)
        print(json.dumps({'dataset': args.dataset, 'a': args.compare[0], 'b': args.compare[1], **compare(a, b)}, indent=1))


if __name__ == '__main__':
    main()
