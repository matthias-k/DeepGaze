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


class CLIP_Normalizer(nn.Module):
    """Normalizer using CLIP's preprocessing statistics."""
    def __init__(self):
        super(CLIP_Normalizer, self).__init__()
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        mean = mean[:, np.newaxis, np.newaxis]

        std = np.array([0.26862954, 0.26130258, 0.27577711])
        std = std[:, np.newaxis, np.newaxis]

        self.register_buffer('mean', torch.tensor(mean), persistent=False)
        self.register_buffer('std', torch.tensor(std), persistent=False)

    def forward(self, tensor):
        tensor = tensor / 255.0

        tensor -= self.mean
        tensor /= self.std

        return tensor