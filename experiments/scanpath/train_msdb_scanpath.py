"""Train the DeepGaze III scanpath part on top of the frozen DeepGaze MSDB priority maps (MIT1003).

Splits follow the released DeepGaze III (pysaliency defaults): fold ``k`` is the test fold, fold
``k - 1`` the validation fold used for the learning-rate schedule and model selection, the other
eight folds are the training data. Requires ``cache_msdb_saliency.py mit1003``.

    python experiments/scanpath/train_msdb_scanpath.py --fold 0
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root: deepgaze_pytorch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from cache_msdb_saliency import saliency_loader  # noqa: E402
from deepgaze_pytorch.deepgaze3_msdb import DeepGazeIIIMSDB  # noqa: E402
from deepgaze_pytorch.deepgazemsdb import MSDBDataset  # noqa: E402
from deepgaze_pytorch.scanpath_tasks import MSDBScanpathTask  # noqa: E402
from deepgaze_pytorch.scanpath_training import bits_relative_to_uniform, train  # noqa: E402


def centerbias_lls(items, load_centerbias):
    """Per-fixation LL of the center bias (bits relative to uniform), the baseline for IG."""
    result = {}
    for item in items:
        centerbias = load_centerbias(item.index)[0]
        result[item.index] = bits_relative_to_uniform(centerbias[item.ys, item.xs], item.image_size).numpy()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--max-epochs', type=int, default=60)
    parser.add_argument('--chunk-size', type=int, default=48)
    parser.add_argument('--name', default='msdb_scanpath')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, help='only images with index < N (smoke test)')
    args = parser.parse_args()

    device = torch.device(args.device)
    stimuli, scanpaths = common.load_mit1003()
    items = common.items_by_index(stimuli, scanpaths)
    load_centerbias = common.centerbias_cache('mit1003', stimuli, lambda: common.mit1003_centerbias_model(stimuli, scanpaths))
    train_idx, val_idx, _ = common.split_indices(len(stimuli), args.fold)
    keep = (lambda n: n in items) if args.limit is None else (lambda n: n in items and n < args.limit)
    train_items = [items[n] for n in train_idx if keep(n)]
    val_items = [items[n] for n in val_idx if keep(n)]

    model = DeepGazeIIIMSDB(pretrained_msdb=True, pretrained_head=False, with_backbone=False).to(device)
    for param in model.parameters():
        param.requires_grad = False
    head = model.head_parameters()
    for param in head:
        param.requires_grad = True

    task = MSDBScanpathTask(model, lambda n: load_centerbias(n, device), common.MIT1003_PIXEL_PER_DVA,
                            MSDBDataset.MIT1003, saliency_maps=saliency_loader('mit1003', device))
    directory = common.RUNS / args.name / f'fold{args.fold}'
    print(f"fold {args.fold}: {len(train_items)} train / {len(val_items)} val images -> {directory}", flush=True)
    train(task, train_items, val_items, head, str(directory), lr=args.lr,
          val_baseline=centerbias_lls(val_items, load_centerbias), min_lr=args.min_lr, patience=args.patience,
          max_epochs=args.max_epochs, chunk_size=args.chunk_size, device=device,
          state_dict_fn=model.head_state_dict, load_state_fn=model.load_head,
          log=lambda line: print(line, flush=True))


if __name__ == '__main__':
    main()
