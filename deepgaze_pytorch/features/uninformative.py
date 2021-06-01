from collections import OrderedDict

import torch
import torch.nn as nn


class OnesLayer(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        self.size = size

    def forward(self, tensor):
        shape = list(tensor.shape)
        shape[1] = 1  # return only one channel

        if self.size is not None:
            shape[2], shape[3] = self.size

        return torch.ones(shape, dtype=torch.float32, device=tensor.device)


class UninformativeFeatures(torch.nn.Sequential):
    def __init__(self):
        super().__init__(OrderedDict([
            ('ones', OnesLayer(size=(1, 1))),
        ]))
