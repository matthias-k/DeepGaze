import math
import pytest
import torch
from deepgaze_pytorch.deepgazemsdb import (
    _DatasetAwareFinalizer,
    _MultiScaleBackbone,
    _freeze_all_but_new_slot,
)


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


def test_finalizer_add_dataset_matches_none_and_freezes_old():
    fin = _make_finalizer()
    readout, cb, sc = _rand_inputs()

    none_pred = fin(readout, cb, sc, None)            # averaged prediction (before surgery)
    old0 = fin(readout, cb, sc, torch.tensor([0]))    # dataset 0 before surgery

    new_idx = fin.add_dataset(n_generalization_datasets=5)
    assert new_idx == 5
    assert fin.dataset_center_bias_weights.shape == (6,)
    assert fin.gauss.dataset_sigmas.shape == (6,)
    assert fin.dataset_priority_scalings.shape == (6,)

    # new slot reproduces dataset=None
    new_pred = fin(readout, cb, sc, torch.tensor([5, 5]))
    assert torch.allclose(new_pred, none_pred, atol=1e-5)
    # old slot unchanged
    old0_after = fin(readout, cb, sc, torch.tensor([0]))
    assert torch.allclose(old0, old0_after, atol=1e-6)


def test_finalizer_none_unchanged_after_add_dataset():
    fin = _make_finalizer(seed=1)
    readout, cb, sc = _rand_inputs(seed=2)
    before = fin(readout, cb, sc, None)
    fin.add_dataset(n_generalization_datasets=5)
    after = fin(readout, cb, sc, None)
    assert torch.allclose(before, after, atol=1e-6)


def test_multiscale_add_dataset_logsumexp_init_and_old_columns_frozen():
    # build only the weight-carrying module; the backbone is unused for this test
    mod = _MultiScaleBackbone.__new__(_MultiScaleBackbone)
    torch.nn.Module.__init__(mod)
    torch.manual_seed(0)
    mod.pixel_per_dva_weights = torch.nn.Parameter(torch.randn(5, 5))
    mod.size_weights = torch.nn.Parameter(torch.randn(5, 5))
    W0 = mod.pixel_per_dva_weights.detach().clone()
    S0 = mod.size_weights.detach().clone()

    new_idx = mod.add_dataset(n_generalization_datasets=5)
    assert new_idx == 5
    assert mod.pixel_per_dva_weights.shape == (5, 6)
    assert mod.size_weights.shape == (5, 6)
    # existing columns byte-identical
    assert torch.equal(mod.pixel_per_dva_weights.detach()[:, :5], W0)
    assert torch.equal(mod.size_weights.detach()[:, :5], S0)
    # new column = log of the arithmetic mean of exp(orig)  (NOT the geometric mean W.mean)
    expected_w = torch.logsumexp(W0, dim=1) - math.log(5)
    expected_s = torch.logsumexp(S0, dim=1) - math.log(5)
    assert torch.allclose(mod.pixel_per_dva_weights.detach()[:, 5], expected_w, atol=1e-6)
    assert torch.allclose(mod.size_weights.detach()[:, 5], expected_s, atol=1e-6)
    # sanity: the geometric-mean init would differ (guards against the 5.4x DAEMONS trap)
    assert not torch.allclose(expected_w, W0.mean(dim=1), atol=1e-3)


def _dataset_param_shapes(n):
    # same shapes/last-axis layout as DeepGazeMSDB.dataset_parameters(): two (5, n) + three (n,)
    return [torch.nn.Parameter(torch.randn(5, n)),
            torch.nn.Parameter(torch.randn(5, n)),
            torch.nn.Parameter(torch.randn(n)),
            torch.nn.Parameter(torch.randn(n)),
            torch.nn.Parameter(torch.randn(n))]


def test_gradient_mask_trains_only_the_new_slot():
    # fresh 5-slot model: new slot appended at index 5, width 6
    params = _dataset_param_shapes(6)
    _freeze_all_but_new_slot(params, new_index=5)
    sum(p.sum() for p in params).backward()
    for p in params:
        assert torch.equal(p.grad[..., :5], torch.zeros_like(p.grad[..., :5]))  # all originals frozen
        assert (p.grad[..., 5] != 0).any()                                       # only the new slot moves


def test_gradient_mask_new_index_differs_from_generalization_count():
    # regression for the boundary bug: the new slot index (5) must drive the mask, NOT
    # n_generalization_datasets. With n=3 the old code froze only cols 0-2, leaking grads
    # into built-in datasets 3 and 4.
    params = _dataset_param_shapes(6)
    _freeze_all_but_new_slot(params, new_index=5)  # new slot is always the last column
    sum(p.sum() for p in params).backward()
    for p in params:
        assert torch.equal(p.grad[..., 3], torch.zeros_like(p.grad[..., 3]))  # DAEMONS stays frozen
        assert torch.equal(p.grad[..., 4], torch.zeros_like(p.grad[..., 4]))  # FIGRIM stays frozen


def test_gradient_mask_second_added_slot_freezes_first():
    # after two add_dataset() calls the model is width 7; only the last slot (index 6) trains,
    # and the first-added slot (index 5) must stay frozen.
    params = _dataset_param_shapes(7)
    _freeze_all_but_new_slot(params, new_index=6)
    sum(p.sum() for p in params).backward()
    for p in params:
        assert torch.equal(p.grad[..., :6], torch.zeros_like(p.grad[..., :6]))  # incl. first-added slot 5
        assert (p.grad[..., 6] != 0).any()


@pytest.mark.slow
def test_model_add_dataset_end_to_end():
    from deepgaze_pytorch import DeepGazeMSDB, MSDBDataset
    torch.manual_seed(0)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepGazeMSDB(pretrained=True).to(dev)
    model.eval()
    image = torch.randint(0, 256, (1, 3, 384, 384)).float().to(dev)
    cb = torch.zeros(1, 384, 384).to(dev)
    with torch.no_grad():
        none_pred = model(image, cb, pixel_per_dva=21.75, dataset=None)
        mit_pred = model(image, cb, pixel_per_dva=21.75, dataset=MSDBDataset.MIT1003)

    idx = model.add_dataset()
    assert idx == 5
    # saliency + backbone frozen; per-dataset tensors stay trainable (full width, grad-masked)
    assert all(not p.requires_grad for p in model.saliency_network.parameters())
    dp = model.dataset_parameters()
    assert all(p.requires_grad for p in dp)

    with torch.no_grad():
        new_pred = model(image, cb, pixel_per_dva=21.75, dataset=idx)
        mit_after = model(image, cb, pixel_per_dva=21.75, dataset=MSDBDataset.MIT1003)
    assert torch.allclose(new_pred, none_pred, atol=1e-4)   # init reproduces averaged
    assert torch.allclose(mit_pred, mit_after, atol=1e-6)   # old dataset frozen

    # gradient masking: an optimizer step moves only the new slot, not the original 5
    before_old = {id(p): p.detach()[..., :5].clone() for p in dp}
    before_new = torch.cat([p.detach()[..., 5:6].reshape(-1).clone() for p in dp])
    opt = torch.optim.Adam(dp, lr=0.1)
    loss = model(image, cb, pixel_per_dva=21.75, dataset=idx).sum()
    opt.zero_grad(); loss.backward(); opt.step()
    for p in dp:
        assert torch.equal(p.detach()[..., :5], before_old[id(p)])
    after_new = torch.cat([p.detach()[..., 5:6].reshape(-1) for p in dp])
    assert not torch.allclose(after_new, before_new)

    # head_state_dict has no backbone and round-trips after a fresh add_dataset
    sd = model.head_state_dict()
    assert not any(k.startswith('features.backbone') for k in sd)
    fresh = DeepGazeMSDB(pretrained=True); fresh.add_dataset()
    missing, unexpected = fresh.load_state_dict(sd, strict=False)
    assert not unexpected


@pytest.mark.slow
def test_msdb_train_returns_self():
    # regression: .train()/.eval() must return self (nn.Module convention) so chaining works
    from deepgaze_pytorch import DeepGazeMSDB
    model = DeepGazeMSDB(pretrained=False)
    assert model.train() is model
    assert model.eval() is model


def test_fixed_geometry_wrapper_absorbs_scanpath_args_and_delegates_state_dict():
    from deepgaze_pytorch.msdb_adaptation import FixedGeometryMSDB

    class _StubMSDB(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(3))
        def forward(self, image, centerbias, pixel_per_dva, dataset=None):
            return (float(pixel_per_dva), dataset)

    inner = _StubMSDB()
    wrapped = FixedGeometryMSDB(inner, pixel_per_dva=21.75, dataset=5)
    # absorbs the scanpath kwargs the shared training loop passes, forwards fixed geometry
    out = wrapped(torch.ones(1), torch.zeros(1),
                  x_hist=torch.tensor([]), y_hist=torch.tensor([]), durations=torch.tensor([]))
    assert out == (21.75, 5)
    # state_dict delegates to the inner model (no 'model.' prefix), so checkpoints stay compatible
    assert set(wrapped.state_dict().keys()) == {'w'}
