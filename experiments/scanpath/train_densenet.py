"""DeepGaze III (DenseNet-201) trained on stretched vs. original MIT1003 images.

train_deepgaze3.ipynb stretches every MIT1003 image (and its fixations) to 1024 x 768 or 768 x 1024.
This script trains the same model on both versions of the data with an identical recipe, so that the
effect of the stretching can be measured on held-out, unstretched images.

Stages (architecture, initialization and freezing as in train_deepgaze3.ipynb; all fixations of an
image form one batch, see deepgaze_pytorch/scanpath_training.py):
  salicon                            spatial pretraining on SALICON, shared by both variants
  spatial   --variant V --fold k     spatial fine-tuning on MIT1003
  scanpath  --variant V --fold k     scanpath training with the first saliency layers frozen,
                                     then fine-tuning of everything except the backbone

Variants: ``stretched`` (as the notebook) and ``original`` (aspect ratios kept; the long side of
the MIT1003 images already is 1024 pixels). SALICON uses the notebook's epoch milestones; the MIT1003
stages lower the learning rate when the validation fold stops improving, so both variants are trained
to convergence by the same criterion.
"""
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pysaliency
import torch
import torch.nn as nn
from pysaliency.baseline_utils import BaselineModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from deepgaze_pytorch.deepgaze3 import build_saliency_network, build_scanpath_network  # noqa: E402
from deepgaze_pytorch.features.densenet import RGBDenseNet201  # noqa: E402
from deepgaze_pytorch.layers import Bias, Conv2dMultiInput, LayerNorm, LayerNormMultiInput  # noqa: E402
from deepgaze_pytorch.modules import DeepGazeIII, FeatureExtractor  # noqa: E402
from deepgaze_pytorch.scanpath_tasks import DeepGazeIIITask  # noqa: E402
from deepgaze_pytorch.scanpath_training import group_by_image, train  # noqa: E402

DENSENET_LAYERS = [
    '1.features.denseblock4.denselayer32.norm1',
    '1.features.denseblock4.denselayer32.conv1',
    '1.features.denseblock4.denselayer31.conv2',
]
FROZEN_SCOPES = ('saliency_network.layernorm0', 'saliency_network.conv0', 'saliency_network.bias0',
                 'saliency_network.layernorm1', 'saliency_network.conv1', 'saliency_network.bias1')
SALICON_CENTERBIAS = dict(bandwidth=0.0217, eps=2e-13)  # train_deepgaze3.ipynb
SALICON_MILESTONES = [15, 30, 45, 60, 75, 90, 105, 120]  # train_deepgaze3.ipynb


def build_fixation_selection_network(scanpath_features):
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNormMultiInput([1, scanpath_features])),
        ('conv0', Conv2dMultiInput([1, scanpath_features], 128, (1, 1), bias=False)),
        ('bias0', Bias(128)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
    ]))


def build_model(scanpath: bool, downsample: float):
    return DeepGazeIII(
        features=FeatureExtractor(RGBDenseNet201(), DENSENET_LAYERS),
        saliency_network=build_saliency_network(2048),
        scanpath_network=build_scanpath_network() if scanpath else None,
        fixation_selection_network=build_fixation_selection_network(16 if scanpath else 0),
        downsample=downsample, readout_factor=4, saliency_map_factor=4,
        included_fixations=common.INCLUDED_FIXATIONS if scanpath else [],
    )


def trainable_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith('features.')}


def load_trainable(model, state, allow_missing=()):
    """Load a checkpoint without backbone weights; ``allow_missing``: prefixes that may be new."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith(('features.',) + tuple(allow_missing))]
    if unexpected or missing:
        raise RuntimeError(f"checkpoint does not match: missing {missing}, unexpected {unexpected}")


def run_dir(*parts):
    return common.RUNS / 'densenet' / Path(*parts)


def best_checkpoint(directory):
    path = directory / 'best.pth'
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run the previous stage first")
    return torch.load(path, map_location='cpu')


def log(line):
    print(line, flush=True)


def stage_salicon(args, device):
    train_stimuli, train_fixations = pysaliency.get_SALICON_train(location=str(common.DATASETS))
    val_stimuli, val_fixations = pysaliency.get_SALICON_val(location=str(common.DATASETS))
    sizes = {tuple(s) for s in list(train_stimuli.sizes) + list(val_stimuli.sizes)}
    if len(sizes) != 1:
        raise RuntimeError(f"expected one SALICON image size, found {sizes}")
    # the (not crossvalidated) center bias only depends on the image size: compute it once
    centerbias_model = BaselineModel(stimuli=train_stimuli, fixations=train_fixations, caching=False, **SALICON_CENTERBIAS)
    centerbias = torch.from_numpy(centerbias_model.log_density(train_stimuli.stimuli[0]).astype(np.float32))[None].to(device)

    model = build_model(scanpath=False, downsample=1.5).to(device)
    train_items = group_by_image(train_stimuli, train_fixations, common.INCLUDED_FIXATIONS, with_history=False)
    val_items = group_by_image(val_stimuli, val_fixations, common.INCLUDED_FIXATIONS, with_history=False)
    train(DeepGazeIIITask(model, common.image_loader(train_stimuli, device), lambda n: centerbias),
          train_items, val_items, [p for p in model.parameters() if p.requires_grad], str(run_dir('salicon')),
          lr=1e-3, min_lr=args.min_lr, milestones=args.milestones or SALICON_MILESTONES, max_epochs=args.max_epochs,
          val_task=DeepGazeIIITask(model, common.image_loader(val_stimuli, device), lambda n: centerbias),
          device=device, log=log, state_dict_fn=lambda: trainable_state(model),
          load_state_fn=lambda state: load_trainable(model, state))


def mit1003_items(variant, fold, device, with_history):
    stimuli, scanpaths = common.load_mit1003_stretched() if variant == 'stretched' else common.load_mit1003()
    cache_name = 'mit1003_stretched' if variant == 'stretched' else 'mit1003'  # 'mit1003' is shared with evaluate.py
    load_centerbias = common.centerbias_cache(cache_name, stimuli, common.mit1003_centerbias_model(stimuli, scanpaths))
    items = {item.index: item for item in group_by_image(stimuli, scanpaths[scanpaths.lengths > 0] if not with_history else scanpaths,
                                                          common.INCLUDED_FIXATIONS, with_history=with_history)}
    train_idx, val_idx, _ = common.split_indices(len(stimuli), fold)
    load_image = common.image_loader(stimuli, device)
    return ([items[n] for n in train_idx if n in items], [items[n] for n in val_idx if n in items],
            load_image, lambda n: load_centerbias(n, device))


def stage_spatial(args, device):
    # train_deepgaze3.ipynb trains the spatial model on the fixations after the initial one
    train_items, val_items, load_image, load_centerbias = mit1003_items(args.variant, args.fold, device, with_history=False)
    model = build_model(scanpath=False, downsample=2).to(device)
    load_trainable(model, best_checkpoint(run_dir('salicon')))
    train(DeepGazeIIITask(model, load_image, load_centerbias), train_items, val_items,
          [p for p in model.parameters() if p.requires_grad], str(run_dir(args.variant, f'fold{args.fold}', 'spatial')),
          lr=1e-3, min_lr=args.min_lr, patience=args.patience, max_epochs=args.max_epochs, device=device, log=log,
          state_dict_fn=lambda: trainable_state(model), load_state_fn=lambda state: load_trainable(model, state))


def stage_scanpath(args, device):
    train_items, val_items, load_image, load_centerbias = mit1003_items(args.variant, args.fold, device, with_history=True)
    base = run_dir(args.variant, f'fold{args.fold}')
    model = build_model(scanpath=True, downsample=2).to(device)
    task = DeepGazeIIITask(model, load_image, load_centerbias)
    common_kwargs = dict(min_lr=args.min_lr, patience=args.patience, max_epochs=args.max_epochs, device=device, log=log,
                         state_dict_fn=lambda: trainable_state(model), load_state_fn=lambda state: load_trainable(model, state))

    # 1) first saliency layers frozen, scanpath network trained from scratch (notebook: lr 1e-3)
    load_trainable(model, best_checkpoint(base / 'spatial'),
                   allow_missing=('scanpath_network.', 'fixation_selection_network.layernorm0.layernorm_part1',
                                  'fixation_selection_network.conv0.conv_part1'))
    for name, param in model.named_parameters():
        if name.startswith(FROZEN_SCOPES):
            param.requires_grad = False
    train(task, train_items, val_items, [p for p in model.parameters() if p.requires_grad],
          str(base / 'scanpath_frozen'), lr=1e-3, **common_kwargs)

    # 2) everything except the backbone (notebook: lr 1e-5)
    load_trainable(model, best_checkpoint(base / 'scanpath_frozen'))
    for name, param in model.named_parameters():
        if not name.startswith('features.'):
            param.requires_grad = True
    train(task, train_items, val_items, [p for p in model.parameters() if p.requires_grad],
          str(base / 'scanpath_full'), lr=1e-5, **common_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['salicon', 'spatial', 'scanpath'])
    parser.add_argument('--variant', choices=['stretched', 'original'])
    parser.add_argument('--fold', type=int)
    parser.add_argument('--milestones', type=int, nargs='*', help='SALICON: epochs at which the learning rate drops')
    parser.add_argument('--min-lr', type=float, default=1e-7)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--max-epochs', type=int, default=100)
    args = parser.parse_args()
    if args.stage != 'salicon' and (args.variant is None or args.fold is None):
        parser.error("--variant and --fold are required for the MIT1003 stages")
    device = torch.device('cuda')
    {'salicon': stage_salicon, 'spatial': stage_spatial, 'scanpath': stage_scanpath}[args.stage](args, device)


if __name__ == '__main__':
    main()
