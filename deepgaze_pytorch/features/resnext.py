from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision

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


