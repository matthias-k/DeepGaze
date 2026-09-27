import math
from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepgaze_pytorch.deepgaze3 import build_fixation_selection_network, build_saliency_network, build_scanpath_network
from deepgaze_pytorch.deepgaze3_msdb import DeepGazeIIIMSDB
from deepgaze_pytorch.modules import DeepGazeIII, DeepGazeIIIMixture, Finalizer, encode_scanpath_features
from deepgaze_pytorch.scanpath_tasks import DeepGazeIIITask, MSDBScanpathTask
from deepgaze_pytorch.scanpath_training import ImageFixations, evaluate, group_by_image, history_arrays, run_epoch
from deepgaze_pytorch.scanpath_utils import (
    encode_history_dva, encode_history_pixels, full_log_density, log_density_at, nearest_source_index,
)
from deepgaze_pytorch.scanpath_utils import mixture_log_density_at as _log_density_at

SIZES = [(12, 5), (683, 342), (768, 384), (1024, 128), (875, 110), (7, 7), (9, 4)]


@pytest.mark.parametrize('dst, src', SIZES)
def test_nearest_source_index_matches_interpolate(dst, src):
    source = torch.arange(src, dtype=torch.float32)[None, None, :, None]
    upsampled = F.interpolate(source, size=(dst, 1), mode='nearest')[0, 0, :, 0].long()
    assert torch.equal(nearest_source_index(dst, src), upsampled)


def _reference_full(pre, image_size):
    full = F.interpolate(pre[:, None], size=image_size, mode='nearest')[:, 0]
    return full - full.logsumexp(dim=(1, 2), keepdim=True)


@pytest.mark.parametrize('image_size, pre_size', [((683, 1024), (342, 512)), ((600, 800), (300, 400)), ((61, 47), (16, 12))])
def test_log_density_at_equals_upsample_and_normalize(image_size, pre_size):
    g = torch.Generator().manual_seed(0)
    pre = 5 * torch.randn(3, *pre_size, generator=g, dtype=torch.float64)
    ys = torch.randint(0, image_size[0], (3,), generator=g)
    xs = torch.randint(0, image_size[1], (3,), generator=g)
    reference = _reference_full(pre, image_size)
    assert torch.allclose(full_log_density(pre, image_size), reference)
    assert torch.allclose(log_density_at(pre, image_size, ys, xs), reference[torch.arange(3), ys, xs])
    # a single map is read at all positions
    assert torch.allclose(log_density_at(pre[:1], image_size, ys, xs), reference[0, ys, xs])
    with pytest.raises(ValueError):
        log_density_at(pre[:2], image_size, ys, xs)


def _histories(batch, image_size, seed=0, missing=True):
    g = torch.Generator().manual_seed(seed)
    x_hist = torch.rand(batch, 4, generator=g) * image_size[1]
    y_hist = torch.rand(batch, 4, generator=g) * image_size[0]
    if missing:
        x_hist[0, 2:] = float('nan')
        y_hist[0, 2:] = float('nan')
    return x_hist, y_hist


def test_encode_history_pixels_matches_deepgaze3_features():
    image_size, readout_shape = (683, 1024), (86, 128)
    x_hist, y_hist = _histories(3, image_size)
    reference = F.interpolate(encode_scanpath_features(x_hist, y_hist, size=image_size), readout_shape)
    ours = encode_history_pixels(x_hist, y_hist, image_size, readout_shape)
    assert torch.allclose(ours, reference, equal_nan=True)


def test_encode_history_dva_uses_cell_centres_and_degrees():
    image_size, readout_shape = (80, 160), (10, 20)  # 8 x 8 pixel cells
    x_hist = torch.tensor([[8 * 3 + 4.0, float('nan'), float('nan'), float('nan')]])
    y_hist = torch.tensor([[8 * 5 + 4.0, float('nan'), float('nan'), float('nan')]])
    features = encode_history_dva(x_hist, y_hist, image_size, readout_shape, pixel_per_dva=16.0)
    dx, dy, distance = features[0, 0], features[0, 4], features[0, 8]
    assert dx[5, 3] == 0 and dy[5, 3] == 0 and distance[5, 3] == 0
    assert dx[5, 4] == pytest.approx(0.5)  # one cell = 8 px = 0.5 degree to the right
    assert torch.isnan(features[0, 1]).all()


# ---- DeepGaze III on MSDB ------------------------------------------------------------------------

def _msdb_model(seed=0):
    torch.manual_seed(seed)
    model = DeepGazeIIIMSDB(pretrained_msdb=False, pretrained_head=False, with_backbone=False)
    with torch.no_grad():  # non-trivial per-dataset parameters
        model.finalizer.gauss.dataset_sigmas.copy_(torch.tensor([0.74, 0.94, 0.93, 0.36, 0.87]))
        model.finalizer.dataset_center_bias_weights.copy_(torch.tensor([0.5, 0.66, 0.58, 0.58, 0.55]))
        model.finalizer.dataset_priority_scalings.copy_(torch.tensor([1.14, 0.72, 1.04, 0.87, 1.25]))
    return model.eval()


def _msdb_spatial_reference(model, saliency, centerbias, batch, pixel_per_dva, dataset):
    """What DeepGazeMSDB.forward does after the saliency network."""
    image_size = centerbias.shape[-2:]
    x = F.interpolate(saliency, [math.ceil(image_size[0] / 2), math.ceil(image_size[1] / 2)], mode='bilinear')
    indices = None if dataset is None else torch.full((batch,), dataset, dtype=torch.long)
    return model.finalizer(x.expand(batch, -1, -1, -1), centerbias.expand(batch, -1, -1),
                           [pixel_per_dva / 2] * batch, indices)


@pytest.mark.parametrize('dataset', [0, None])
def test_untrained_msdb_scanpath_model_reproduces_msdb(dataset):
    model = _msdb_model()
    image_size = (75, 101)
    g = torch.Generator().manual_seed(1)
    saliency = torch.rand(1, 1, 10, 13, generator=g)
    centerbias = torch.log_softmax(torch.randn(1, image_size[0] * image_size[1], generator=g), dim=1).view(1, *image_size)
    x_hist, y_hist = _histories(4, image_size)
    with torch.no_grad():
        pre = model.pre_log_density(saliency, centerbias, x_hist, y_hist, pixel_per_dva=24.0, dataset=dataset)
        reference = _msdb_spatial_reference(model, saliency, centerbias, 4, 24.0, dataset)
    assert torch.allclose(full_log_density(pre, image_size), reference, atol=1e-5)


def test_released_scanpath_head_matches_the_model():
    from deepgaze_pytorch.deepgaze3_msdb import _HEAD_WEIGHTS
    model = DeepGazeIIIMSDB(pretrained_msdb=False, pretrained_head=False, with_backbone=False)
    state = torch.load(_HEAD_WEIGHTS, map_location='cpu', weights_only=True)
    assert set(state) == set(model.head_state_dict())
    model.load_head(state)
    assert model.fixation_selection_network.conv2.weight.abs().sum() > 0  # trained, not the zero initialization
    with pytest.raises(RuntimeError):
        model.load_head({k: v for k, v in state.items() if not k.startswith('scanpath_network.')})


def test_pretrained_head_requires_the_pretrained_spatial_pathway():
    with pytest.raises(ValueError):
        DeepGazeIIIMSDB(pretrained_msdb=False, pretrained_head=True, with_backbone=False)


def test_batched_gaussian_filter_equals_per_item_filter():
    model = _msdb_model()
    readout = torch.rand(3, 1, 20, 30)
    indices = torch.tensor([2, 2, 2])
    batched = model.finalizer.gauss(readout, [12.0] * 3, indices)
    per_item = torch.cat([model.finalizer.gauss(readout[i:i + 1], [12.0], indices[i:i + 1] * 0 + 2) for i in range(3)])
    # the per-item path is taken when scaling factors differ; force it with distinct floats
    looped = model.finalizer.gauss(readout, [12.0, 12.0 + 1e-12, 12.0], indices)
    assert torch.allclose(batched, per_item, atol=1e-6)
    assert torch.allclose(batched, looped, atol=1e-5)


def test_msdb_training_step_only_changes_the_scanpath_head():
    model = _msdb_model()
    image_size = (48, 64)
    items = [ImageFixations(index=0, image_size=image_size, xs=torch.tensor([3, 40, 60]), ys=torch.tensor([5, 20, 47]),
                            x_hist=_histories(3, image_size)[0], y_hist=_histories(3, image_size)[1])]
    saliency = torch.rand(1, 1, 6, 8)
    centerbias = torch.log_softmax(torch.randn(1, 48 * 64), dim=1).view(1, *image_size)
    task = MSDBScanpathTask(model, lambda n: centerbias, pixel_per_dva=35.0, dataset=0, saliency_maps=lambda n: saliency)
    spatial_before = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith(('scanpath', 'fixation_selection'))}
    head = model.head_parameters()
    before = [p.detach().clone() for p in head]
    optimizer = torch.optim.Adam(head, lr=1e-2)
    lls = run_epoch(task, items, optimizer=optimizer, chunk_size=2, device='cpu')
    assert np.isfinite(lls[0]).all() and lls[0].shape == (3,)
    assert any(not torch.equal(a, b) for a, b in zip(before, head))
    for k, v in model.state_dict().items():
        if k in spatial_before:
            assert torch.equal(v, spatial_before[k]), k


# ---- DeepGaze III tasks --------------------------------------------------------------------------

class _Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 3, stride=4, padding=1)

    def forward(self, x):
        return [self.conv(x / 255.0), torch.relu(self.conv(x / 255.0))]


def _deepgaze3(seed=0):
    torch.manual_seed(seed)
    return DeepGazeIII(features=_Features(), saliency_network=build_saliency_network(12),
                       scanpath_network=build_scanpath_network(), fixation_selection_network=build_fixation_selection_network(),
                       downsample=2, readout_factor=4, saliency_map_factor=4, included_fixations=[-1, -2, -3, -4]).eval()


def _mixture(components=2):
    torch.manual_seed(0)
    return DeepGazeIIIMixture(features=_Features(),
                              saliency_networks=[build_saliency_network(12) for _ in range(components)],
                              scanpath_networks=[build_scanpath_network() for _ in range(components)],
                              fixation_selection_networks=[build_fixation_selection_network() for _ in range(components)],
                              finalizers=[Finalizer(sigma=8.0, learn_sigma=True, saliency_map_factor=4) for _ in range(components)],
                              downsample=2, readout_factor=4, saliency_map_factor=4, included_fixations=[-1, -2, -3, -4]).eval()


@pytest.mark.parametrize('build', [_deepgaze3, _mixture])
def test_deepgaze3_task_reproduces_forward(build):
    model = build()
    image_size = (83, 125)
    g = torch.Generator().manual_seed(3)
    image = torch.rand(1, 3, *image_size, generator=g) * 255
    centerbias = torch.log_softmax(torch.randn(1, image_size[0] * image_size[1], generator=g), dim=1).view(1, *image_size)
    x_hist, y_hist = _histories(3, image_size)
    ys, xs = torch.tensor([4, 50, 82]), torch.tensor([0, 77, 124])
    with torch.no_grad():
        reference = model(image.expand(3, -1, -1, -1), centerbias.expand(3, -1, -1), x_hist, y_hist)
        if reference.dim() == 4:
            reference = reference[:, 0]
        task = DeepGazeIIITask(model, lambda n: image, lambda n: centerbias)
        pre = task.pre_log_density(task.image_context(ImageFixations(0, image_size, xs, ys, x_hist, y_hist)), x_hist, y_hist)
        ours = _log_density_at(pre, image_size, ys, xs)
    assert torch.allclose(ours, reference[torch.arange(3), ys, xs], atol=1e-5)


def test_evaluate_metrics_match_direct_computation():
    model = _deepgaze3()
    image_size = (83, 125)
    image = torch.rand(1, 3, *image_size) * 255
    centerbias = torch.log_softmax(torch.randn(1, image_size[0] * image_size[1]), dim=1).view(1, *image_size)
    x_hist, y_hist = _histories(2, image_size, missing=False)
    ys, xs = torch.tensor([10, 60]), torch.tensor([20, 100])
    item = ImageFixations(0, image_size, xs, ys, x_hist, y_hist)
    results = evaluate(DeepGazeIIITask(model, lambda n: image, lambda n: centerbias), [item], device='cpu')[0]
    with torch.no_grad():
        maps = model(image.expand(2, -1, -1, -1), centerbias.expand(2, -1, -1), x_hist, y_hist).double()
    for b in range(2):
        value = maps[b, ys[b], xs[b]]
        assert results['LL'][b] == pytest.approx(float((value + math.log(83 * 125)) / math.log(2)), abs=1e-4)
        assert results['AUC'][b] == pytest.approx(float((maps[b] < value).double().mean() + 0.5 * (maps[b] == value).double().mean()))
        density = maps[b].exp()
        assert results['NSS'][b] == pytest.approx(float((value.exp() - density.mean()) / density.std(unbiased=False)), rel=1e-4)


# ---- data grouping --------------------------------------------------------------------------------

def test_history_arrays_and_grouping_follow_pysaliency_scanpaths(tmp_path):
    import pysaliency
    from PIL import Image
    files = []
    for i, (w, h) in enumerate([(40, 30), (20, 20)]):
        path = tmp_path / f'{i}.png'
        Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(path)
        files.append(str(path))
    stimuli = pysaliency.FileStimuli(files)
    train_xs = [np.array([20.5, 5.0, 30.2, 10.0, 39.0, 1.0]), np.array([10.0, 2.0])]
    train_ys = [np.array([15.0, 3.0, 20.0, 29.0, 0.0, 5.0]), np.array([10.0, 19.0])]
    train_ts = [np.arange(6, dtype=float), np.arange(2, dtype=float)]
    fixations = pysaliency.FixationTrains.from_fixation_trains(train_xs, train_ys, train_ts, [0, 1], [0, 0])
    items = {item.index: item for item in group_by_image(stimuli, fixations, [-1, -2, -3, -4])}
    first = items[0]
    assert first.image_size == (30, 40)
    assert first.xs.tolist() == [5, 30, 10, 39, 1]  # the initial fixation is not scored
    # history of the last fixation: the 4 previous fixations, most recent first
    assert first.x_hist[-1].tolist() == pytest.approx([39.0, 10.0, 30.2, 5.0])
    # the second fixation has only the initial fixation as history
    assert first.x_hist[0, 0].item() == pytest.approx(20.5)
    assert torch.isnan(first.x_hist[0, 1:]).all()
    assert items[1].xs.tolist() == [2] and items[1].x_hist[0, 0].item() == pytest.approx(10.0)


def test_rescaled_task_maps_predictions_back_to_the_original_grid():
    from deepgaze_pytorch.scanpath_tasks import RescaledTask
    from deepgaze_pytorch.scanpath_utils import mixture_full_log_density
    model = _deepgaze3()
    image_size = (60, 80)
    image = torch.rand(1, 3, *image_size) * 255
    centerbias = torch.log_softmax(torch.randn(1, 60 * 80), dim=1).view(1, *image_size)
    x_hist, y_hist = _histories(2, image_size, missing=False)
    item = ImageFixations(0, image_size, torch.tensor([1, 2]), torch.tensor([1, 2]), x_hist, y_hist)
    inner = DeepGazeIIITask(model, lambda n: image, lambda n: centerbias)
    with torch.no_grad():
        direct = mixture_full_log_density(inner.pre_log_density(inner.image_context(item), x_hist, y_hist), image_size)
        same = RescaledTask(inner, 1.0, 1.0)
        assert torch.allclose(same.log_density_maps(same.image_context(item), x_hist, y_hist, image_size), direct, atol=1e-5)
        big = F.interpolate(image, size=(90, 120), mode='bilinear')
        big_cb = torch.log_softmax(torch.randn(1, 90 * 120), dim=1).view(1, 90, 120)
        rescaled = RescaledTask(DeepGazeIIITask(model, lambda n: big, lambda n: big_cb), 1.5, 1.5)
        maps = rescaled.log_density_maps(rescaled.image_context(item), x_hist, y_hist, image_size)
    assert maps.shape == (2, 60, 80)
    assert torch.allclose(maps.exp().sum(dim=(1, 2)), torch.ones(2, dtype=maps.dtype), atol=1e-5)


def _spatial_deepgaze3():
    from deepgaze_pytorch.layers import Bias, Conv2dMultiInput, LayerNorm, LayerNormMultiInput
    torch.manual_seed(0)
    selection = nn.Sequential(OrderedDict([
        ('layernorm0', LayerNormMultiInput([1, 0])),
        ('conv0', Conv2dMultiInput([1, 0], 8, (1, 1), bias=False)),
        ('bias0', Bias(8)),
        ('softplus0', nn.Softplus()),
        ('conv1', nn.Conv2d(8, 1, (1, 1), bias=False)),
    ]))
    return DeepGazeIII(features=_Features(), saliency_network=build_saliency_network(12), scanpath_network=None,
                       fixation_selection_network=selection, downsample=2, readout_factor=4, saliency_map_factor=4,
                       included_fixations=[])


def test_spatial_task_scores_all_fixations_with_one_prediction():
    model = _spatial_deepgaze3().eval()
    image_size = (83, 125)
    image = torch.rand(1, 3, *image_size) * 255
    centerbias = torch.log_softmax(torch.randn(1, image_size[0] * image_size[1]), dim=1).view(1, *image_size)
    nan = torch.full((4, 4), float('nan'))
    item = ImageFixations(0, image_size, torch.tensor([0, 30, 124, 30]), torch.tensor([0, 40, 82, 40]), nan, nan)
    task = DeepGazeIIITask(model, lambda n: image, lambda n: centerbias)
    assert task.history_independent
    lls = run_epoch(task, [item], device='cpu')[0]
    with torch.no_grad():
        reference = model(image, centerbias)[0]
    expected = (reference[item.ys, item.xs] + math.log(83 * 125)) / math.log(2)
    assert np.allclose(lls, expected.numpy(), atol=1e-4)
    trainable = [p for p in model.parameters() if p.requires_grad and not any(p is q for q in model.features.parameters())]
    optimizer = torch.optim.Adam(trainable, lr=1e-2)
    before = [p.detach().clone() for p in trainable]
    run_epoch(task, [item], optimizer=optimizer, device='cpu')
    assert any(not torch.equal(a, b) for a, b in zip(before, trainable))


def test_spatial_grouping_keeps_all_fixations_in_order(tmp_path):
    import pysaliency
    from PIL import Image
    files = []
    for i in range(3):
        path = tmp_path / f'{i}.png'
        Image.fromarray(np.zeros((10, 12, 3), np.uint8)).save(path)
        files.append(str(path))
    stimuli = pysaliency.FileStimuli(files)
    fixations = pysaliency.Fixations.create_without_history(
        x=np.array([1.0, 2.0, 3.0, 4.0, 5.0]), y=np.array([1.0, 1.0, 2.0, 2.0, 3.0]), n=np.array([2, 0, 2, 0, 2]))
    items = group_by_image(stimuli, fixations, [-1, -2, -3, -4], with_history=False)
    assert [item.index for item in items] == [0, 2]
    assert items[0].xs.tolist() == [2, 4] and items[1].xs.tolist() == [1, 3, 5]
    assert items[1].x_hist.shape == (3, 0)
