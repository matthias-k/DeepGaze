from collections import OrderedDict

import torch
import torch.nn as nn

from torch.utils import model_zoo

from .features.alexnet import RGBalexnet
from .modules import FeatureExtractor, Finalizer, DeepGazeII as TorchDeepGazeII


class DeepGazeI(TorchDeepGazeII):
    """DeepGaze I model

    Please note that this version of DeepGaze I is not exactly the one from the original paper.
    The original model used caffe for AlexNet and theano for the linear readout and was trained using the SFO optimizer.
    Here, we use the torch implementation of AlexNet (without any adaptations), which doesn't use the two-steam architecture,
    and the DeepGaze II torch implementation with a simple linear readout network.
    The model has been retrained with Adam, but still on the same dataset (all images of MIT1003 which are of size 1024x768).
    Also, we don't use the sparsity penalty anymore.

    Reference:
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
            self.load_state_dict(model_zoo.load_url('https://github.com/matthias-k/DeepGaze/releases/download/v1.01/deepgaze1.pth', map_location=torch.device('cpu')))
