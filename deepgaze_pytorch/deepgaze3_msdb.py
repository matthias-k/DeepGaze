"""DeepGaze III scanpath model on top of the DeepGaze MSDB spatial pathway.

The spatial priority map is the one of DeepGaze MSDB (CLIP ResNet50x64 + DINOv2 features at several
scales, dataset-specific scale weights, blur, center-bias weight and priority scaling). The scanpath
and fixation-selection networks of DeepGaze III combine it with the previous fixations.

Two choices differ from DeepGaze III:

- The fixation selection is residual and its last layer starts at zero, so an untrained model
  reproduces DeepGaze MSDB exactly; training only adds the effect of the fixation history.
- History fixations are encoded in degrees of visual angle instead of pixels, so the model transfers
  between datasets presented at different resolutions, as the MSDB spatial pathway does.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

from .deepgaze3 import build_fixation_selection_network, build_scanpath_network
from .deepgazemsdb import (
    _INPUT_CHANNELS,
    _N_DATASETS,
    _PIXEL_PER_DVA_SCALES,
    _READOUT_FACTOR,
    _SALIENCY_MAP_FACTOR,
    _SIZE_SCALES,
    _WEIGHTS_URL,
    _build_backbone,
    _build_saliency_network,
    _DatasetAwareFinalizer,
    _MultiScaleBackbone,
)
from .scanpath_utils import encode_history_dva, full_log_density

_HEAD_PREFIXES = ('scanpath_network.', 'fixation_selection_network.')


class DeepGazeIIIMSDB(nn.Module):
    """Scanpath model: DeepGaze MSDB priority map + DeepGaze III fixation history.

    Args:
        pretrained_msdb: load the released DeepGaze MSDB weights for the spatial pathway.
        with_backbone: build the CLIP + DINOv2 backbone. Without it the model only works on
            precomputed priority maps (``pre_log_density``), which is how it is trained.
    """
    included_fixations = [-1, -2, -3, -4]

    def __init__(self, pretrained_msdb: bool = True, with_backbone: bool = True):
        super().__init__()
        if with_backbone:
            self.features = _MultiScaleBackbone(
                backbone=_build_backbone(),
                readout_factor=_READOUT_FACTOR,
                n_datasets=_N_DATASETS,
                resolutions_pixel_per_dva=_PIXEL_PER_DVA_SCALES,
                resolutions_size=_SIZE_SCALES,
                feature_interpolation_mode='bilinear',
            )
            for param in self.features.backbone.parameters():
                param.requires_grad = False
            self.features.backbone.eval()
        else:
            self.features = None

        self.saliency_network = _build_saliency_network(_INPUT_CHANNELS)
        self.finalizer = _DatasetAwareFinalizer(sigma=1.0, n_datasets=_N_DATASETS)
        self.scanpath_network = build_scanpath_network()
        self.fixation_selection_network = build_fixation_selection_network()
        nn.init.zeros_(self.fixation_selection_network.conv2.weight)

        if pretrained_msdb:
            state = model_zoo.load_url(_WEIGHTS_URL, map_location=torch.device('cpu'))
            if self.features is None:
                state = {k: v for k, v in state.items() if not k.startswith('features.')}
            missing, unexpected = self.load_state_dict(state, strict=False)
            not_loaded = [k for k in missing if not k.startswith(_HEAD_PREFIXES + ('features.backbone.',))]
            if unexpected or not_loaded:
                raise RuntimeError(f"MSDB checkpoint does not match: unexpected {unexpected}, missing {not_loaded}")

    def head_parameters(self):
        """Parameters of the scanpath part (the only ones trained on top of a frozen MSDB)."""
        return list(self.scanpath_network.parameters()) + list(self.fixation_selection_network.parameters())

    def saliency(self, image: torch.Tensor, pixel_per_dva: float, dataset: Optional[int] = None) -> torch.Tensor:
        """MSDB priority map at readout resolution, (B, 1, ceil(H/8), ceil(W/8))."""
        if self.features is None:
            raise RuntimeError("model was built without backbone; pass precomputed priority maps")
        dataset_indices = None
        if dataset is not None:
            dataset_indices = torch.full((image.shape[0],), dataset, dtype=torch.long, device=image.device)
        x = self.features(image.to(torch.float32), dataset_index=dataset_indices,
                          pixel_per_dva=[pixel_per_dva] * image.shape[0])
        return self.saliency_network(x)

    def pre_log_density(self, saliency: torch.Tensor, centerbias: torch.Tensor, x_hist: torch.Tensor,
                        y_hist: torch.Tensor, pixel_per_dva: float, dataset: Optional[int] = None) -> torch.Tensor:
        """Unnormalized log density of the next fixation at half resolution, one map per history.

        Args:
            saliency: (1 or B, 1, h, w) priority map; one map is shared by all B histories.
            centerbias: (1 or B, H, W) center-bias log density at image resolution.
            x_hist, y_hist: (B, 4) previous fixations in pixels, most recent first, NaN if missing.
        Returns:
            (B, ceil(H/2), ceil(W/2)); ``scanpath_utils.full_log_density`` / ``log_density_at`` turn it
            into the normalized full-resolution log density.
        """
        batch = x_hist.shape[0]
        image_size = centerbias.shape[-2:]
        saliency = saliency.expand(batch, -1, -1, -1)

        scanpath_features = encode_history_dva(x_hist, y_hist, image_size, saliency.shape[-2:], pixel_per_dva)
        x = saliency + self.fixation_selection_network((saliency, self.scanpath_network(scanpath_features)))

        saliency_shape = [math.ceil(image_size[0] / _SALIENCY_MAP_FACTOR), math.ceil(image_size[1] / _SALIENCY_MAP_FACTOR)]
        x = F.interpolate(x, saliency_shape, mode='bilinear')
        downscaled_centerbias = F.interpolate(centerbias[:, None], size=saliency_shape)[:, 0].expand(batch, -1, -1)

        dataset_indices = None
        if dataset is not None:
            dataset_indices = torch.full((batch,), dataset, dtype=torch.long, device=x.device)
        return self.finalizer.combine(x[:, 0:1], downscaled_centerbias,
                                      [pixel_per_dva / _SALIENCY_MAP_FACTOR] * batch, dataset_indices)

    def forward(self, image: torch.Tensor, centerbias: torch.Tensor, x_hist: torch.Tensor, y_hist: torch.Tensor,
                pixel_per_dva: float, dataset: Optional[int] = None) -> torch.Tensor:
        """Log density (B, H, W) of the next fixation, for B images with one history each."""
        saliency = self.saliency(image, pixel_per_dva, dataset)
        pre = self.pre_log_density(saliency, centerbias, x_hist, y_hist, pixel_per_dva, dataset)
        return full_log_density(pre, centerbias.shape[-2:])

    def train(self, mode: bool = True):
        super().train(mode=mode)
        if self.features is not None:
            self.features.backbone.eval()
        return self
