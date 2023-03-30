from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .normalizer import Normalizer



class RGBMobileNetV2(nn.Sequential):
    def __init__(self):
        super(RGBMobileNetV2, self).__init__()
        self.mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=True)
        self.normalizer = Normalizer()
        super(RGBMobileNetV2, self).__init__(self.normalizer, self.mobilenet_v2)
