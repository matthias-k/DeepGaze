from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .normalizer import Normalizer



class RGBDenseNet201(nn.Sequential):
    def __init__(self):
        super(RGBDenseNet201, self).__init__()
        self.densenet = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201', pretrained=True)
        self.normalizer = Normalizer()
        super(RGBDenseNet201, self).__init__(self.normalizer, self.densenet)


