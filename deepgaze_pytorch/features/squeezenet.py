from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .normalizer import Normalizer

    
class RGBSqueezeNet(nn.Sequential):
    def __init__(self):
        super(RGBSqueezeNet, self).__init__()
        self.squeezenet = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_0', pretrained=True)
        self.normalizer = Normalizer()
        super(RGBSqueezeNet, self).__init__(self.normalizer, self.squeezenet)

