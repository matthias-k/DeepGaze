from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision


class VGGInputNormalization(torch.nn.Module):
    def __init__(self, inplace=True):
        super().__init__()

        self.inplace = inplace

        mean = np.array([0.485, 0.456, 0.406])
        mean = mean[:, np.newaxis, np.newaxis]

        std = np.array([0.229, 0.224, 0.225])
        std = std[:, np.newaxis, np.newaxis]
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))

    def forward(self, tensor):
        if self.inplace:
            tensor /= 255.0
        else:
            tensor = tensor / 255.0

        tensor -= self.mean
        tensor /= self.std

        return tensor


class VGG19BNNamedFeatures(torch.nn.Sequential):
    def __init__(self):
        names = []
        for block in range(5):
            block_size = 2 if block < 2 else 4
            for layer in range(block_size):
                names.append(f'conv{block+1}_{layer+1}')
                names.append(f'bn{block+1}_{layer+1}')
                names.append(f'relu{block+1}_{layer+1}')
            names.append(f'pool{block+1}')

        vgg = torchvision.models.vgg19_bn(pretrained=True)
        vgg_features = vgg.features
        vgg.classifier = torch.nn.Sequential()

        assert len(names) == len(vgg_features)

        named_features = OrderedDict({'normalize': VGGInputNormalization()})

        for name, feature in zip(names, vgg_features):
            if isinstance(feature, nn.MaxPool2d):
                feature.ceil_mode = True
            named_features[name] = feature

        super().__init__(named_features)


class VGG19NamedFeatures(torch.nn.Sequential):
    def __init__(self):
        names = []
        for block in range(5):
            block_size = 2 if block < 2 else 4
            for layer in range(block_size):
                names.append(f'conv{block+1}_{layer+1}')
                names.append(f'relu{block+1}_{layer+1}')
            names.append(f'pool{block+1}')

        vgg = torchvision.models.vgg19(pretrained=True)
        vgg_features = vgg.features
        vgg.classifier = torch.nn.Sequential()

        assert len(names) == len(vgg_features)

        named_features = OrderedDict({'normalize': VGGInputNormalization()})

        for name, feature in zip(names, vgg_features):
            if isinstance(feature, nn.MaxPool2d):
                feature.ceil_mode = True

            named_features[name] = feature

        super().__init__(named_features)
