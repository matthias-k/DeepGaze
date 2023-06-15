dependencies = ['torch', 'numpy']

from deepgaze_pytorch.deepgaze1 import DeepGazeI as _DeepGazeI
from deepgaze_pytorch.deepgaze2e import DeepGazeIIE as _DeepGazeIIE
from deepgaze_pytorch.deepgaze3 import DeepGazeIII as _DeepGazeIII


def DeepGazeI(pretrained=False, **kwargs):
    model = _DeepGazeI(pretrained=pretrained, **kwargs)
    return model


def DeepGazeIIE(pretrained=False, **kwargs):
    model = _DeepGazeIIE(pretrained=pretrained, **kwargs)
    return model


def DeepGazeIII(pretrained=False, **kwargs):
    model = _DeepGazeIII(pretrained=pretrained, **kwargs)
    return model