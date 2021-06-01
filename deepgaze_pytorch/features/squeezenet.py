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
    
class RGBSqueezeNet(nn.Sequential):
    def __init__(self):
        super(RGBSqueezeNet, self).__init__()
        self.squeezenet = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_0', pretrained=True)
        self.normalizer = Normalizer()
        super(RGBSqueezeNet, self).__init__(self.normalizer, self.squeezenet)

