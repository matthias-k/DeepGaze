from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision

class Normalizer(nn.Module):
    def __init__(self):
        super(Normalizer, self).__init__()
        mean = np.array([0.485, 0.456, 0.406])
        mean = mean[:, np.newaxis, np.newaxis]

        std = np.array([0.229, 0.224, 0.225])
        std = std[:, np.newaxis, np.newaxis]

        # don't persist to keep old checkpoints working
        self.register_buffer('mean', torch.tensor(mean), persistent=False)
        self.register_buffer('std', torch.tensor(std), persistent=False)


    def forward(self, tensor):
        tensor = tensor / 255.0

        tensor -= self.mean
        tensor /= self.std

        return tensor