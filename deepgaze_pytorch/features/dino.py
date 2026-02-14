"""DINOv2 feature extractor for DeepGaze MSDB."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from einops import rearrange
except ImportError:
    raise ImportError("einops is required for DINOv2 feature extractors. Install with: pip install einops")

from .normalizer import Normalizer


class DINOv2_ViTB14(nn.Sequential):
    """DINOv2 ViT-B/14 backbone (used by DeepGaze MSDB)."""
    def __init__(self):
        super(DINOv2_ViTB14, self).__init__()
        self.dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2:6a62615', 'dinov2_vitb14', skip_validation=True)
        self.normalizer = Normalizer()
        super(DINOv2_ViTB14, self).__init__(self.normalizer, self.dinov2_vitb14)


class DINOTransformersFeatureExtractor(nn.Module):
    """Feature extractor for DINOv2 Vision Transformers.

    Extracts intermediate block outputs and reshapes them from sequence format
    to spatial feature maps.
    """
    def __init__(self, features, targets):
        super().__init__()
        self.features = features
        self.targets = targets
        self.outputs = {}

        if isinstance(self.features[1].patch_embed.patch_size, tuple):
            self.patch_size = self.features[1].patch_embed.patch_size[0]
        else:
            self.patch_size = self.features[1].patch_embed.patch_size
        self.output_dimension_providing_layer = self.features[1].patch_embed.proj

        for target in targets:
            layer = dict([*self.features.named_modules()])[target]
            layer.register_forward_hook(self.save_outputs_hook(target))

    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self.outputs[layer_id] = output.clone()
        return fn

    def forward(self, x):
        self.outputs.clear()

        # Reshape to nearest multiple of patch size (requirement of DINOv2)
        h = self.patch_size * math.ceil(x.shape[-2] / self.patch_size)
        w = self.patch_size * math.ceil(x.shape[-1] / self.patch_size)
        x = F.interpolate(x, (h, w), recompute_scale_factor=False)

        temp_shape = self.output_dimension_providing_layer(x).shape
        h_ = temp_shape[-2]
        w_ = temp_shape[-1]

        self.features(x)

        # Reshape transformer output from (B, N, C) to (B, C, H, W)
        for target in self.targets:
            # Remove CLS token (first token) and reshape
            self.outputs[target] = rearrange(self.outputs[target][:, 1:, :], 'b (h w) c -> b h w c', h=h_, w=w_)
            self.outputs[target] = rearrange(self.outputs[target], 'b h w c -> b c h w')

        return [self.outputs[target] for target in self.targets]
