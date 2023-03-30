from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .normalizer import Normalizer

    
    

class RGBSwav(nn.Sequential):
    def __init__(self):
        super(RGBSwav, self).__init__()        
        self.swav = torch.hub.load('facebookresearch/swav', 'resnet50', pretrained=True) 
        self.normalizer = Normalizer()
        super(RGBSwav, self).__init__(self.normalizer, self.swav)


