from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .normalizer import Normalizer


class RGBvgg19(nn.Sequential):
    def __init__(self):
        super(RGBvgg19, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        self.normalizer = Normalizer()
        super(RGBvgg19, self).__init__(self.normalizer, self.model)


class RGBvgg11(nn.Sequential):
    def __init__(self):
        super(RGBvgg11, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11', pretrained=True)
        self.normalizer = Normalizer()
        super(RGBvgg11, self).__init__(self.normalizer, self.model)