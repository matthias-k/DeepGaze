import math
import pytest
import torch
from deepgaze_pytorch.deepgazemsdb import _DatasetAwareFinalizer, _MultiScaleBackbone


def _rand_inputs(B=2, H=16, W=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    readout = torch.rand(B, 1, H, W, generator=g)
    centerbias = torch.rand(B, H, W, generator=g)
    scaling = [7.0] * B
    return readout, centerbias, scaling


def _make_finalizer(seed=0):
    torch.manual_seed(seed)
    fin = _DatasetAwareFinalizer(sigma=1.0, n_datasets=5)
    with torch.no_grad():
        fin.dataset_priority_scalings.copy_(torch.tensor([1.14, 0.72, 1.04, 0.87, 1.25]))
        fin.dataset_center_bias_weights.copy_(torch.tensor([0.5, 0.66, 0.58, 0.58, 0.55]))
        fin.gauss.dataset_sigmas.copy_(torch.tensor([0.74, 0.94, 0.93, 0.36, 0.87]))
    return fin


def test_priority_reference_buffer_preserves_output():
    fin = _make_finalizer()
    readout, cb, sc = _rand_inputs()
    idx = torch.tensor([0, 3])
    before = fin(readout, cb, sc, idx)
    # enabling the fixed reference at the current mean must not change anything
    fin.set_priority_reference(fin.dataset_priority_scalings.mean())
    after = fin(readout, cb, sc, idx)
    assert torch.allclose(before, after, atol=1e-6)
