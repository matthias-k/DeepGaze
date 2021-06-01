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


class RGBResNet34(nn.Sequential):
    def __init__(self):
        super(RGBResNet34, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)
        self.normalizer = Normalizer()
        super(RGBResNet34, self).__init__(self.normalizer, self.resnet)


class RGBResNet50(nn.Sequential):
    def __init__(self):
        super(RGBResNet50, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.normalizer = Normalizer()
        super(RGBResNet50, self).__init__(self.normalizer, self.resnet)


class RGBResNet50_alt(nn.Sequential):
    def __init__(self):
        super(RGBResNet50, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.normalizer = Normalizer()
        state_dict = torch.load("Resnet-AlternativePreTrain.pth")
        model.load_state_dict(state_dict)
        super(RGBResNet50, self).__init__(self.normalizer, self.resnet)



class RGBResNet101(nn.Sequential):
    def __init__(self):
        super(RGBResNet101, self).__init__()
        self.resnet = torchvision.models.resnet101(pretrained=True)
        self.normalizer = Normalizer()
        super(RGBResNet101, self).__init__(self.normalizer, self.resnet)
