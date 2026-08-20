import inspect
import torch
from deepgaze_pytorch.training import (
    _finalizer_scalar_for_logging,
    _forward_with_supported_kwargs,
    _train,
)


class _FakeFinalizerNoScalars:  # MSDB-like: no .gauss.sigma / .center_bias_weight
    pass


def test_finalizer_scalar_helper_is_optional():
    assert _finalizer_scalar_for_logging(_FakeFinalizerNoScalars(), 'gauss.sigma') is None


def test_finalizer_scalar_helper_reads_existing():
    class G: sigma = torch.tensor(1.5)
    class M: gauss = G()
    val = _finalizer_scalar_for_logging(M(), 'gauss.sigma')
    assert float(val) == 1.5


def test_train_accepts_state_dict_fn():
    assert 'state_dict_fn' in inspect.signature(_train).parameters


class _MSDBLike(torch.nn.Module):
    def forward(self, image, centerbias, pixel_per_dva, dataset=None):
        return (float(pixel_per_dva), dataset)


class _ScanpathLike(torch.nn.Module):
    def forward(self, image, centerbias, x_hist=None, y_hist=None, durations=None, **kwargs):
        return ('scan', x_hist, kwargs)


def test_dispatch_passes_only_supported_kwargs_to_explicit_model():
    m = _MSDBLike()
    out = _forward_with_supported_kwargs(
        m, torch.ones(1), torch.zeros(1),
        x_hist=1, y_hist=2, durations=3, pixel_per_dva=35.0, dataset=None)
    assert out == (35.0, None)   # scanpath kwargs dropped, msdb kwargs kept


def test_dispatch_passes_scanpath_kwargs_to_varkw_model():
    m = _ScanpathLike()
    out = _forward_with_supported_kwargs(
        m, torch.ones(1), torch.zeros(1),
        x_hist=7, y_hist=8, durations=9, pixel_per_dva=35.0)
    # scanpath model receives x_hist explicitly; pixel_per_dva flows via **kwargs
    assert out[1] == 7 and out[2].get('pixel_per_dva') == 35.0
