"""Datasets, splits and caches shared by the scanpath experiments.

Paths come from the environment: ``DEEPGAZE_DATA`` (datasets, default ``/data``), ``DEEPGAZE_CACHE``
(center-bias and priority-map caches, default ``$DEEPGAZE_DATA/cache``) and ``DEEPGAZE_RUNS``
(training runs and results, default ``$DEEPGAZE_DATA/runs``).
"""
import os
from pathlib import Path

import numpy as np
import pysaliency
import torch
from PIL import Image
from boltons.iterutils import chunked
from pysaliency.baseline_utils import CrossvalidatedBaselineModel
from tqdm import tqdm

from deepgaze_pytorch.custom_data import fit_centerbias
from deepgaze_pytorch.scanpath_training import group_by_image

DATA = Path(os.environ.get('DEEPGAZE_DATA', '/data'))
RUNS = Path(os.environ.get('DEEPGAZE_RUNS', DATA / 'runs'))
DATASETS = DATA / 'pysaliency_datasets'
CACHE = Path(os.environ.get('DEEPGAZE_CACHE', DATA / 'cache'))

MIT1003_PIXEL_PER_DVA = 35.0  # DeepGaze MSDB README / DeepGaze III training resolution
OSIE_PIXEL_PER_DVA = 24.0     # Xu et al. 2014: 800 x 600 images, 1 degree ~ 24 pixels

# MIT1003 center bias as in train_deepgaze3.ipynb (leave-one-image-out crossvalidated KDE)
MIT1003_CENTERBIAS = dict(bandwidth=10 ** -1.6667673342543432, eps=10 ** -14.884189168516073)

INCLUDED_FIXATIONS = [-1, -2, -3, -4]
CROSSVAL_FOLDS = 10


# ---- datasets ----------------------------------------------------------------------------------

def load_mit1003():
    """MIT1003 scanpaths starting with the initial central fixation, at the original image sizes."""
    return pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(
        location=str(DATASETS), replace_initial_invalid_fixations=True)


def _stretched_size(height, width):
    # train_deepgaze3.ipynb: every image becomes 1024 x 768 (landscape) or 768 x 1024 (portrait)
    return (768, 1024) if height < width else (1024, 768)


def load_mit1003_stretched():
    """MIT1003 as train_deepgaze3.ipynb prepares it: images and fixations stretched to 1024 x 768 or
    768 x 1024, whatever their aspect ratio."""
    stimuli, scanpaths = load_mit1003()
    directory = DATASETS / 'MIT1003_twosize' / 'stimuli'
    directory.mkdir(parents=True, exist_ok=True)
    filenames = []
    for filename, (height, width) in zip(stimuli.filenames, stimuli.sizes):
        target = directory / os.path.basename(filename)
        if not target.exists():
            new_height, new_width = _stretched_size(height, width)
            image = Image.open(filename).convert('RGB')
            if (new_height, new_width) != (height, width):
                image = image.resize((new_width, new_height), Image.BILINEAR)
            image.save(target)
        filenames.append(str(target))
    new_stimuli = pysaliency.FileStimuli(filenames)

    train_xs = [x.copy() for x in scanpaths.train_xs]
    train_ys = [y.copy() for y in scanpaths.train_ys]
    for i, n in enumerate(scanpaths.train_ns):
        height, width = stimuli.sizes[n]
        new_height, new_width = _stretched_size(height, width)
        train_xs[i] *= new_width / width
        train_ys[i] *= new_height / height
    attributes = {key: getattr(scanpaths, key).copy() for key in scanpaths.__attributes__
                  if key not in ['subjects', 'scanpath_index']}
    new_scanpaths = pysaliency.FixationTrains(
        train_xs=train_xs, train_ys=train_ys, train_ts=scanpaths.train_ts.copy(), train_ns=scanpaths.train_ns.copy(),
        train_subjects=scanpaths.train_subjects.copy(), attributes=attributes)
    return new_stimuli, new_scanpaths


def load_osie():
    return pysaliency.external_datasets.get_OSIE(location=str(DATASETS))


# ---- splits ------------------------------------------------------------------------------------

def crossval_folds(n_stimuli, crossval_folds=CROSSVAL_FOLDS):
    """Stimulus indices of each fold, exactly as pysaliency.filter_datasets creates them
    (random=True: shuffled with RandomState(42), then chunked)."""
    indices = list(range(n_stimuli))
    np.random.RandomState(seed=42).shuffle(indices)
    size = int(np.ceil(n_stimuli / crossval_folds))
    return [list(chunk) for chunk in chunked(indices, size=size)]


def split_indices(n_stimuli, fold_no, crossval_folds=CROSSVAL_FOLDS):
    """train / val / test stimulus indices of fold ``fold_no`` with pysaliency's defaults
    (``val_folds=1, test_folds=1``: test fold ``fold_no``, validation fold ``fold_no - 1``), as used
    by train_deepgaze3.ipynb for the released DeepGaze III."""
    folds = crossval_folds(n_stimuli, crossval_folds)
    test = fold_no
    val = (fold_no - 1) % crossval_folds
    train = [i for f in range(crossval_folds) if f not in (test, val) for i in folds[f]]
    return sorted(train), sorted(folds[val]), sorted(folds[test])


# ---- center bias -------------------------------------------------------------------------------

def centerbias_cache(name, stimuli, model):
    """Cache of the center-bias log densities (float32, one file per image) -> loader."""
    directory = CACHE / 'centerbias' / name
    directory.mkdir(parents=True, exist_ok=True)
    missing = [n for n in range(len(stimuli)) if not (directory / f'{n}.npy').exists()]
    for n in tqdm(missing, desc=f'center bias {name}'):
        np.save(directory / f'{n}.npy', model.log_density(stimuli.stimuli[n]).astype(np.float32))

    def load(n, device='cpu'):
        return torch.from_numpy(np.load(directory / f'{n}.npy'))[None].to(device)
    return load


def mit1003_centerbias_model(stimuli, scanpaths):
    return CrossvalidatedBaselineModel(stimuli, scanpaths[scanpaths.lengths > 0], caching=False, **MIT1003_CENTERBIAS)


def osie_centerbias_model(stimuli, scanpaths):
    return fit_centerbias(stimuli, scanpaths[scanpaths.lengths > 0], crossvalidated=True, verbose=True)


# ---- images and fixations ----------------------------------------------------------------------

def image_loader(stimuli, device='cpu'):
    def load(n):
        image = np.asarray(Image.open(stimuli.filenames[n]).convert('RGB'))
        return torch.from_numpy(image.transpose(2, 0, 1).copy())[None].float().to(device)
    return load


def items_by_index(stimuli, scanpaths):
    return {item.index: item for item in group_by_image(stimuli, scanpaths, INCLUDED_FIXATIONS)}
