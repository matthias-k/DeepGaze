from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .normalizer import Normalizer




class RGBInceptionV3(nn.Sequential):
    def __init__(self):
        super(RGBInceptionV3, self).__init__()
        self.resnext = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
        self.normalizer = Normalizer()
        super(RGBInceptionV3, self).__init__(self.normalizer, self.resnext)


