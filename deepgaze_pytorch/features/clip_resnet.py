"""CLIP ResNet feature extractor for DeepGaze MSDB."""

import torch.nn as nn

try:
    import clip
except ImportError:
    raise ImportError("clip is required for CLIP feature extractors. Install with: pip install git+https://github.com/openai/CLIP.git")

from .normalizer import CLIP_Normalizer


class CLIPResNet50x64(nn.Sequential):
    """CLIP ResNet-50x64 backbone (used by DeepGaze MSDB)."""
    def __init__(self):
        super(CLIPResNet50x64, self).__init__()
        self.clip_model, _ = clip.load("RN50x64")
        self.visual_clip_model = self.clip_model.visual
        self.visual_clip_model.attnpool = nn.Sequential()

        self.normalizer = CLIP_Normalizer()

        super().__init__(self.normalizer, self.visual_clip_model)
