from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import model_zoo

from .features.densenet import RGBDenseNet201
from .modules import FeatureExtractor, Finalizer, DeepGazeIIIMixture
from .layers import FlexibleScanpathHistoryEncoding

from .layers import (
    Conv2dMultiInput,
    LayerNorm,
    LayerNormMultiInput,
    Bias,
)


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
        ('softplus2', nn.Softplus()),
    ]))


def build_scanpath_network():
    return nn.Sequential(OrderedDict([
        ('encoding0', FlexibleScanpathHistoryEncoding(in_fixations=4, channels_per_fixation=3, out_channels=128, kernel_size=[1, 1], bias=True)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
    ]))


def build_fixation_selection_network():
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNormMultiInput([1, 16])),
        ('conv0', Conv2dMultiInput([1, 16], 128, (1, 1), bias=False)),
        ('bias0', Bias(128)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),

        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
    ]))


class DeepGazeIII(DeepGazeIIIMixture):
    """DeepGazeIII model

    :note
    See Kümmerer, M., Bethge, M., & Wallis, T.S.A. (2022). DeepGaze III: Modeling free-viewing human scanpaths with deep learning. Journal of Vision 2022, https://doi.org/10.1167/jov.22.5.7
    """
    def __init__(self, pretrained=True):
        features = RGBDenseNet201()

        feature_extractor = FeatureExtractor(features, [
            '1.features.denseblock4.denselayer32.norm1',
            '1.features.denseblock4.denselayer32.conv1',
            '1.features.denseblock4.denselayer31.conv2',
        ])

        saliency_networks = []
        scanpath_networks = []
        fixation_selection_networks = []
        finalizers = []
        for component in range(10):
            saliency_network = build_saliency_network(2048)
            scanpath_network = build_scanpath_network()
            fixation_selection_network = build_fixation_selection_network()

            saliency_networks.append(saliency_network)
            scanpath_networks.append(scanpath_network)
            fixation_selection_networks.append(fixation_selection_network)
            finalizers.append(Finalizer(sigma=8.0, learn_sigma=True, saliency_map_factor=4))

        super().__init__(
            features=feature_extractor,
            saliency_networks=saliency_networks,
            scanpath_networks=scanpath_networks,
            fixation_selection_networks=fixation_selection_networks,
            finalizers=finalizers,
            downsample=2,
            readout_factor=4,
            saliency_map_factor=4,
            included_fixations=[-1, -2, -3, -4]
        )

        if pretrained:
            self.load_state_dict(model_zoo.load_url('https://github.com/matthias-k/DeepGaze/releases/download/v1.1.0/deepgaze3.pth', map_location=torch.device('cpu')))