from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .normalizer import Normalizer



class RGBalexnet(nn.Sequential):
    def __init__(self):
        super(RGBalexnet, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        self.normalizer = Normalizer()
        super(RGBalexnet, self).__init__(self.normalizer, self.model)

