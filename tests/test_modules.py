import sys

import torch
import torch.nn as nn

import deepgaze_pytorch
from deepgaze_pytorch.features.dino import DINOTransformersFeatureExtractor
from deepgaze_pytorch.modules import DeepGazeII, DeepGazeIII, DeepGazeIIIMixture, Finalizer, MixtureModel


class _Features(nn.Module):
    """Stand-in for a frozen backbone with BatchNorm (DenseNet, EfficientNet, ...)."""
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        return [self.bn(x)]


class _SelectSaliency(nn.Module):
    def forward(self, inputs):
        return inputs[0]


def _mixture():
    return DeepGazeIIIMixture(
        features=_Features(),
        saliency_networks=[nn.Conv2d(3, 1, (1, 1))],
        scanpath_networks=[None],
        fixation_selection_networks=[_SelectSaliency()],
        finalizers=[Finalizer(sigma=1.0, saliency_map_factor=2)],
        downsample=1,
        readout_factor=1,
    )


def test_mixture_train_keeps_backbone_in_eval():
    model = _mixture()
    assert model.train() is model
    assert model.training
    assert model.saliency_networks[0].training
    assert not model.features.training
    assert not model.features.bn.training


def test_mixture_training_step_does_not_touch_backbone_batchnorm():
    # regression: train() used to put the frozen backbone's BatchNorm into training mode,
    # which overwrote its running statistics during fine-tuning
    model = _mixture().train()
    running_mean = model.features.bn.running_mean.clone()
    model(torch.rand(2, 3, 8, 8) * 255, torch.zeros(2, 8, 8))
    assert torch.equal(model.features.bn.running_mean, running_mean)


def test_mixture_model_propagates_train_to_components():
    model = MixtureModel([_mixture(), _mixture()])
    model.train()
    for component in model.models:
        assert component.saliency_networks[0].training
        assert not component.features.training


def test_deepgaze_ii_and_iii_train_and_eval_return_self():
    models = [
        DeepGazeII(features=_Features(), readout_network=nn.Conv2d(3, 1, (1, 1))),
        DeepGazeIII(features=_Features(), saliency_network=nn.Conv2d(3, 1, (1, 1)),
                    scanpath_network=None, fixation_selection_network=_SelectSaliency()),
    ]
    for model in models:
        assert model.train() is model
        assert model.training and not model.features.training
        assert model.eval() is model
        assert not model.training


def test_package_import_does_not_require_clip():
    # DeepGaze I/IIE/III must be usable without the MSDB-only CLIP dependency
    assert deepgaze_pytorch.DeepGazeIII is not None
    assert 'clip' not in sys.modules


class _FakeViT(nn.Module):
    """Minimal DINOv2-like ViT: CLS token followed by row-major patch tokens."""
    def __init__(self, patch_size=2, dim=4):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.patch_size = (patch_size, patch_size)
        self.patch_embed.proj = nn.Conv2d(3, dim, patch_size, patch_size)
        self.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])

    def forward(self, x):
        patches = self.patch_embed.proj(x)
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = torch.cat([torch.zeros(tokens.shape[0], 1, tokens.shape[2]), tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


def test_dino_extractor_restores_spatial_layout():
    features = nn.Sequential(nn.Identity(), _FakeViT())
    extractor = DINOTransformersFeatureExtractor(features, targets=['1.blocks.1'])
    x = torch.randn(2, 3, 6, 8)  # non-square, so a swapped h/w would be caught
    with torch.no_grad():
        output, = extractor(x)
        expected = features[1].patch_embed.proj(x)
    assert output.shape == expected.shape
    assert torch.allclose(output, expected)
