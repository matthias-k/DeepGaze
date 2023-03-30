"""
This code was adapted from: https://github.com/rgeirhos/texture-vs-shape
"""
import os
import sys
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
import torchvision.models
from torch.utils import model_zoo

from .normalizer import Normalizer


def load_model(model_name):

    model_urls = {
            'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
            'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
            'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
              'vgg16_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar',
            'alexnet_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar',
    }

    if "resnet50" in model_name:
        #print("Using the ResNet50 architecture.")
        model = torchvision.models.resnet50(pretrained=False)
        #model = torch.nn.DataParallel(model)  # .cuda()
        # fake DataParallel structrue
        model = torch.nn.Sequential(OrderedDict([('module', model)]))
        checkpoint = model_zoo.load_url(model_urls[model_name], map_location=torch.device('cpu'))
    elif "vgg16" in model_name:
        #print("Using the VGG-16 architecture.")

        # download model from URL manually and save to desired location
        filepath = "./vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar"

        assert os.path.exists(filepath), "Please download the VGG model yourself from the following link and save it locally: https://drive.google.com/drive/folders/1A0vUWyU6fTuc-xWgwQQeBvzbwi6geYQK (too large to be downloaded automatically like the other models)"

        model = torchvision.models.vgg16(pretrained=False)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))


    elif "alexnet" in model_name:
        #print("Using the AlexNet architecture.")
        model = torchvision.models.alexnet(pretrained=False)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        checkpoint = model_zoo.load_url(model_urls[model_name], map_location=torch.device('cpu'))
    else:
        raise ValueError("unknown model architecture.")

    model.load_state_dict(checkpoint["state_dict"])
    return model

# --- DeepGaze Adaptation ----




class RGBShapeNetA(nn.Sequential):
    def __init__(self):
        super(RGBShapeNetA, self).__init__()
        self.shapenet = load_model("resnet50_trained_on_SIN")
        self.normalizer = Normalizer()
        super(RGBShapeNetA, self).__init__(self.normalizer, self.shapenet)



class RGBShapeNetB(nn.Sequential):
    def __init__(self):
        super(RGBShapeNetB, self).__init__()
        self.shapenet = load_model("resnet50_trained_on_SIN_and_IN")
        self.normalizer = Normalizer()
        super(RGBShapeNetB, self).__init__(self.normalizer, self.shapenet)


class RGBShapeNetC(nn.Sequential):
    def __init__(self):
        super(RGBShapeNetC, self).__init__()
        self.shapenet = load_model("resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN")
        self.normalizer = Normalizer()
        super(RGBShapeNetC, self).__init__(self.normalizer, self.shapenet)



