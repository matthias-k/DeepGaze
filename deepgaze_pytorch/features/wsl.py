from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .normalizer import Normalizer

    

class RGBResNext50(nn.Sequential):
    def __init__(self):
        super(RGBResNext50, self).__init__()
        self.resnext = torch.hub.load('facebookresearch/WSL-Images', 'resnext50_32x16d_wsl')
        self.normalizer = Normalizer()
        super(RGBResNext50, self).__init__(self.normalizer, self.resnext)


class RGBResNext101(nn.Sequential):
    def __init__(self):
        super(RGBResNext101, self).__init__()
        self.resnext = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
        self.normalizer = Normalizer()
        super(RGBResNext101, self).__init__(self.normalizer, self.resnext)


