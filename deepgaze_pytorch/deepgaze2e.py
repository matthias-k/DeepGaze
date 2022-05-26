from collections import OrderedDict
import importlib
import os


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import model_zoo

from .modules import FeatureExtractor, Finalizer, DeepGazeIIIMixture, MixtureModel

from .layers import (
    Conv2dMultiInput,
    LayerNorm,
    LayerNormMultiInput,
    Bias,
)


BACKBONES = [
    {
        'type': 'deepgaze_pytorch.features.shapenet.RGBShapeNetC',
        'used_features': [
            '1.module.layer3.0.conv2',
            '1.module.layer3.3.conv2',
            '1.module.layer3.5.conv1',
            '1.module.layer3.5.conv2',
            '1.module.layer4.1.conv2',
            '1.module.layer4.2.conv2',
        ],
        'channels': 2048,
    },
    {
        'type': 'deepgaze_pytorch.features.efficientnet.RGBEfficientNetB5',
        'used_features': [
            '1._blocks.24._depthwise_conv',
            '1._blocks.26._depthwise_conv',
            '1._blocks.35._project_conv',
        ],
        'channels': 2416,
    },
    {
        'type': 'deepgaze_pytorch.features.densenet.RGBDenseNet201',
        'used_features': [
            '1.features.denseblock4.denselayer32.norm1',
            '1.features.denseblock4.denselayer32.conv1',
            '1.features.denseblock4.denselayer31.conv2',
        ],
        'channels': 2048,
    },
    {
        'type': 'deepgaze_pytorch.features.resnext.RGBResNext50',
        'used_features': [
            '1.layer3.5.conv1',
            '1.layer3.5.conv2',
            '1.layer3.4.conv2',
            '1.layer4.2.conv2',
        ],
        'channels': 2560,
    },
]


def build_saliency_network(input_channels):
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNorm(input_channels)),
        ('conv0', nn.Conv2d(input_channels, 8, (1, 1), bias=False)),
        ('bias0', Bias(8)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(8)),
        ('conv1', nn.Conv2d(8, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),

        ('layernorm2', LayerNorm(16)),
        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
        ('bias2', Bias(1)),
        ('softplus3', nn.Softplus()),
    ]))


def build_fixation_selection_network():
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNormMultiInput([1, 0])),
        ('conv0', Conv2dMultiInput([1, 0], 128, (1, 1), bias=False)),
        ('bias0', Bias(128)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),

        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
    ]))


def build_deepgaze_mixture(backbone_config, components=10):
    feature_class = import_class(backbone_config['type'])
    features = feature_class()

    feature_extractor = FeatureExtractor(features, backbone_config['used_features'])

    saliency_networks = []
    scanpath_networks = []
    fixation_selection_networks = []
    finalizers = []
    for component in range(components):
        saliency_network = build_saliency_network(backbone_config['channels'])
        fixation_selection_network = build_fixation_selection_network()

        saliency_networks.append(saliency_network)
        scanpath_networks.append(None)
        fixation_selection_networks.append(fixation_selection_network)
        finalizers.append(Finalizer(sigma=8.0, learn_sigma=True, saliency_map_factor=2))

    return DeepGazeIIIMixture(
        features=feature_extractor,
        saliency_networks=saliency_networks,
        scanpath_networks=scanpath_networks,
        fixation_selection_networks=fixation_selection_networks,
        finalizers=finalizers,
        downsample=2,
        readout_factor=16,
        saliency_map_factor=2,
        included_fixations=[],
    )


class DeepGazeIIE(MixtureModel):
    """DeepGazeIIE model

    :note
    See Linardos, A., Kümmerer, M., Press, O., & Bethge, M. (2021). Calibrated prediction in and out-of-domain for state-of-the-art saliency modeling. ArXiv:2105.12441 [Cs], http://arxiv.org/abs/2105.12441
    """
    def __init__(self, pretrained=True):
        # we average over 3 instances per backbone, each instance has 10 crossvalidation folds
        backbone_models = [build_deepgaze_mixture(backbone_config, components=3 * 10) for backbone_config in BACKBONES]
        super().__init__(backbone_models)

        if pretrained:
            self.load_state_dict(model_zoo.load_url('https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/deepgaze2e.pth', map_location=torch.device('cpu')))


def import_class(name):
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
