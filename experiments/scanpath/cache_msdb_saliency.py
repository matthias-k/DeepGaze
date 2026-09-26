"""Precompute the DeepGaze MSDB priority maps (saliency network output at readout resolution).

The MSDB backbone and saliency network stay frozen while the DeepGaze III scanpath part is trained on
top, so their output per image is computed once here. MIT1003 uses the MIT1003 dataset slot of MSDB,
OSIE (not in MSDB's training data) the averaged parameters (``dataset=None``).

    python experiments/scanpath/cache_msdb_saliency.py mit1003
    python experiments/scanpath/cache_msdb_saliency.py osie
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from deepgaze_pytorch.deepgaze3_msdb import DeepGazeIIIMSDB  # noqa: E402
from deepgaze_pytorch.deepgazemsdb import MSDBDataset  # noqa: E402

DATASETS = {
    'mit1003': (common.load_mit1003, common.MIT1003_PIXEL_PER_DVA, MSDBDataset.MIT1003),
    'osie': (common.load_osie, common.OSIE_PIXEL_PER_DVA, None),
}


def cache_directory(name):
    return common.CACHE / 'msdb_saliency' / name


def saliency_loader(name, device='cpu'):
    directory = cache_directory(name)

    def load(n):
        return torch.from_numpy(np.load(directory / f'{n}.npy')).to(device)
    return load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=sorted(DATASETS))
    args = parser.parse_args()
    load, pixel_per_dva, dataset = DATASETS[args.dataset]
    stimuli, _ = load()
    directory = cache_directory(args.dataset)
    directory.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda')
    model = DeepGazeIIIMSDB(pretrained_msdb=True, with_backbone=True).to(device).eval()
    load_image = common.image_loader(stimuli, device=device)
    todo = [n for n in range(len(stimuli)) if not (directory / f'{n}.npy').exists()]
    print(f"{args.dataset}: {len(todo)} of {len(stimuli)} images to compute", flush=True)
    start = time.time()
    with torch.no_grad():
        for done, n in enumerate(todo, 1):
            saliency = model.saliency(load_image(n), pixel_per_dva, dataset)
            if not torch.isfinite(saliency).all():
                raise RuntimeError(f"non-finite priority map for image {n}")
            np.save(directory / f'{n}.npy', saliency.cpu().numpy().astype(np.float32))
            if done % 50 == 0 or done == len(todo):
                elapsed = time.time() - start
                print(f"{done}/{len(todo)} images, {elapsed / done:.2f} s/image, "
                      f"max memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)


if __name__ == '__main__':
    main()
