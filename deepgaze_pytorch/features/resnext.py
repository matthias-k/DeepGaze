from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .normalizer import Normalizer

    

class RGBResNext50(nn.Sequential):
    def __init__(self):
        super(RGBResNext50, self).__init__()
        self.resnext = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True) 
        self.normalizer = Normalizer()
        super(RGBResNext50, self).__init__(self.normalizer, self.resnext)


class RGBResNext101(nn.Sequential):
    def __init__(self):
        super(RGBResNext101, self).__init__()
        self.resnext = torch.hub.load('pytorch/vision:v0.6.0', 'resnext101_32x8d', pretrained=True) 
        self.normalizer = Normalizer()
        super(RGBResNext101, self).__init__(self.normalizer, self.resnext)


