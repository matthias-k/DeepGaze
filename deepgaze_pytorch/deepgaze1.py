from collections import OrderedDict

import torch
import torch.nn as nn

from torch.utils import model_zoo

from .features.alexnet import RGBalexnet
from .modules import FeatureExtractor, Finalizer, DeepGazeII as TorchDeepGazeII


class DeepGazeI(TorchDeepGazeII):
    """DeepGaze I model

    Kümmerer, M., Theis, L., & Bethge, M. (2015). Deep Gaze I: Boosting Saliency Prediction with Feature Maps Trained on ImageNet. ICLR Workshop Track. http://arxiv.org/abs/1411.1045
    """
    def __init__(self, pretrained=True):
        features = RGBalexnet()
        feature_extractor = FeatureExtractor(features, ['1.features.10'])

        readout_network = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(256, 1, (1, 1), bias=False)),
        ]))

        super().__init__(
            features=feature_extractor,
            readout_network=readout_network,
            downsample=2,
            readout_factor=4,
            saliency_map_factor=4,
        )

        if pretrained:
            raise NotImplementedError()
