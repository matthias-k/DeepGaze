from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .efficientnet_pytorch import EfficientNet


class Normalizer(nn.Module):
    def __init__(self):
        super(Normalizer, self).__init__()
        self.mean = torch.Tensor([0.485, 0.456, 0.406])
        self.std = torch.Tensor([0.229, 0.224, 0.225])

    def forward(self, input):
        t = input/255
        for i in range(3):
            t[0][i]=(t[0][i]-self.mean[i])/self.std[i]

        return t


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


