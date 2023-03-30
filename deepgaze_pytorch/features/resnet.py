from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .normalizer import Normalizer



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
