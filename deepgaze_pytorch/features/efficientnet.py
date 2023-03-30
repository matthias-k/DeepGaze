from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .efficientnet_pytorch import EfficientNet


from .normalizer import Normalizer



class RGBEfficientNetB5(nn.Sequential):
    def __init__(self):
        super(RGBEfficientNetB5, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b5')
        self.normalizer = Normalizer()
        super(RGBEfficientNetB5, self).__init__(self.normalizer, self.efficientnet)



class RGBEfficientNetB7(nn.Sequential):
    def __init__(self):
        super(RGBEfficientNetB7, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')
        self.normalizer = Normalizer()
        super(RGBEfficientNetB7, self).__init__(self.normalizer, self.efficientnet)


