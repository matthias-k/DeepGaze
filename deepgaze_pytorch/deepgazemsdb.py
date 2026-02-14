"""DeepGaze MSDB (Multi-Scale Dataset Bias) model.

This model is from the ICCV 2025 paper:
    Kümmerer, M., Khanuja, H., & Bethge, M. (2025). Modeling Saliency Dataset Bias.
    In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).
    https://openaccess.thecvf.com/content/ICCV2025/html/Kummerer_Modeling_Saliency_Dataset_Bias_ICCV_2025_paper.html

The model combines CLIP ResNet50x64 and DINOv2 ViT-B/14 features processed at multiple
scales, with learned per-dataset parameters for optimal performance on known datasets
or averaged parameters for generalization to new datasets.
"""

from collections import OrderedDict
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

from .layers import Bias, LayerNorm, gaussian_filter_1d
from .modules import FeatureExtractor
from .features.clip_resnet import CLIPResNet50x64
from .features.dino import DINOv2_ViTB14, DINOTransformersFeatureExtractor


class MSDBDataset:
    """Dataset indices for DeepGaze MSDB dataset-specific parameters.

    Use these constants when evaluating on known datasets to use
    dataset-specific learned parameters.

    Example:
        >>> model(image, centerbias, pixel_per_dva=35.0, dataset=MSDBDataset.MIT1003)
    """
    MIT1003 = 0
    CAT2000 = 1
    COCO_FREEVIEW = 2
    DAEMONS = 3
    FIGRIM = 4


# Model configuration constants
_PIXEL_PER_DVA_SCALES = [5.0, 10.0, 17.5, 24.0, 30.0]
_SIZE_SCALES = [128, 256, 512, 768, 1024]
_INPUT_CHANNELS = 2560  # CLIP (1792) + DINOv2 (768)
_N_DATASETS = 5
_READOUT_FACTOR = 8
_SALIENCY_MAP_FACTOR = 2


def _build_saliency_network(input_channels: int) -> nn.Sequential:
    """Build the saliency network.

    Architecture: LayerNorm → Conv → Softplus repeated with channel sizes:
    input_channels → 8 → 16 → 1 → 128 → 1
    """
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNorm(input_channels)),
        ('conv0', nn.Conv2d(input_channels, 8, (1, 1), bias=False)),
        ('bias0', Bias(8)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(8)),
        ('conv1', nn.Conv2d(8, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),

        ('layernorm2', LayerNorm(16)),
        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
        ('bias2', Bias(1)),
        ('softplus2', nn.Softplus()),

        ('layernorm3', LayerNorm(1)),
        ('conv3', nn.Conv2d(1, 128, (1, 1), bias=False)),
        ('bias3', Bias(128)),
        ('softplus3', nn.Softplus()),

        ('layernorm4', LayerNorm(128)),
        ('conv4', nn.Conv2d(128, 1, (1, 1), bias=False)),
        ('bias4', Bias(1)),
        ('softplus4', nn.Softplus()),
    ]))


class _DatasetAwareGaussianFilter(nn.Module):
    """Gaussian filter with per-dataset learned sigma values."""

    def __init__(self, dims: List[int], sigma: float, n_datasets: int, truncate: float = 3):
        super().__init__()
        self.dims = dims
        self.truncate = truncate
        self.dataset_sigmas = nn.Parameter(
            torch.ones(n_datasets, dtype=torch.float32) * sigma,
            requires_grad=True
        )

    def forward(self, tensor: torch.Tensor, scaling_factors: List[float],
                dataset_indices: Optional[torch.Tensor]) -> torch.Tensor:
        if dataset_indices is None:
            # Average over all datasets
            sigma = self.dataset_sigmas.mean()
            sigmas = [sigma for _ in range(tensor.shape[0])]
        else:
            sigmas = self.dataset_sigmas[dataset_indices]

        outputs = []
        for image_data, scaling, sigma in zip(tensor, scaling_factors, sigmas):
            item = image_data.unsqueeze(0)
            for dim in self.dims:
                item = gaussian_filter_1d(
                    item,
                    dim=dim,
                    sigma=sigma * scaling,
                    truncate=self.truncate,
                )
            outputs.append(item)

        return torch.cat(outputs, dim=0)


class _DatasetAwareFinalizer(nn.Module):
    """Finalizer with per-dataset learned parameters for sigma, center bias weight, and priority scaling."""

    def __init__(self, sigma: float, n_datasets: int):
        super().__init__()
        self.gauss = _DatasetAwareGaussianFilter([2, 3], sigma, n_datasets=n_datasets, truncate=3)
        self.dataset_center_bias_weights = nn.Parameter(torch.ones(n_datasets), requires_grad=True)
        self.dataset_priority_scalings = nn.Parameter(torch.zeros(n_datasets), requires_grad=True)

    def forward(self, readout: torch.Tensor, centerbias: torch.Tensor,
                scaling_factors: List[float], dataset_indices: Optional[torch.Tensor]) -> torch.Tensor:
        # Downscale centerbias to match readout
        downscaled_centerbias = F.interpolate(
            centerbias.view(centerbias.shape[0], 1, centerbias.shape[1], centerbias.shape[2]),
            size=(readout.shape[2], readout.shape[3])
        )[:, 0, :, :]

        # Apply gaussian filter
        out = self.gauss(readout, scaling_factors, dataset_indices)
        out = out[:, 0, :, :]

        # Apply priority scaling
        if dataset_indices is not None:
            # Normalize w.r.t geometric mean to make numbers comparable
            dataset_priority_scalings_mean_log = torch.mean(self.dataset_priority_scalings)
            dataset_priority_scalings = torch.exp(self.dataset_priority_scalings - dataset_priority_scalings_mean_log)
            priority_scalings = dataset_priority_scalings[dataset_indices].view(-1, 1, 1)
        else:
            priority_scalings = torch.ones(out.shape[0], 1, 1, device=out.device)

        out = out * priority_scalings

        # Add center bias
        if dataset_indices is not None:
            centerbias_weights = self.dataset_center_bias_weights[dataset_indices].view(-1, 1, 1)
        else:
            centerbias_weight = self.dataset_center_bias_weights.mean()
            centerbias_weights = centerbias_weight.view(1, 1, 1)

        out = out + centerbias_weights * downscaled_centerbias

        # Upscale to original size
        out = F.interpolate(
            out[:, np.newaxis, :, :],
            size=[centerbias.shape[1], centerbias.shape[2]],
            mode='nearest',
        )[:, 0, :, :]

        # Normalize to log probability
        out = out - out.logsumexp(dim=(1, 2), keepdim=True)

        return out


class _MultiScaleBackbone(nn.Module):
    """Multi-scale backbone that processes images at multiple resolutions.

    Combines features from:
    - Multiple pixel-per-degree-of-visual-angle scales (resolution-based)
    - Multiple absolute size scales (size-based)

    With learned per-dataset weights for each scale.
    """

    def __init__(self, backbone: nn.Module, readout_factor: int, n_datasets: int,
                 resolutions_pixel_per_dva: List[float], resolutions_size: List[int],
                 feature_interpolation_mode: str = 'bilinear'):
        super().__init__()
        self.backbone = backbone
        self.readout_factor = readout_factor
        self.resolutions_pixel_per_dva = resolutions_pixel_per_dva
        self.resolutions_size = resolutions_size
        self.feature_interpolation_mode = feature_interpolation_mode

        # Learned per-dataset weights for each scale (stored in log space)
        self.pixel_per_dva_weights = nn.Parameter(
            torch.zeros((len(resolutions_pixel_per_dva), n_datasets)),
            requires_grad=True
        )
        self.size_weights = nn.Parameter(
            torch.zeros((len(resolutions_size), n_datasets)),
            requires_grad=True
        )

    def _process_pixel_per_dva(self, x: torch.Tensor, image_pixel_per_dvas: List[float],
                                target_pixel_per_dva: float, readout_shape: List[int]) -> torch.Tensor:
        """Process images rescaled to target pixel-per-dva resolution."""
        factors = [target_pixel_per_dva / external_resolution for external_resolution in image_pixel_per_dvas]

        x_rescaled = [
            F.interpolate(
                item.unsqueeze(0),
                [math.ceil(item.shape[1] * factor), math.ceil(item.shape[2] * factor)],
                mode='bicubic',
            ) for item, factor in zip(x, factors)
        ]

        with torch.no_grad():
            x_features = [self.backbone(y) for y in x_rescaled]

        image_features = []
        for image_data in x_features:
            image_data = [item.to(torch.float32) for item in image_data]
            image_data = [F.interpolate(item, readout_shape, mode=self.feature_interpolation_mode) for item in image_data]
            image_data = torch.cat(image_data, dim=1)
            image_features.append(image_data)

        return torch.cat(image_features, dim=0)

    def _process_size(self, x: torch.Tensor, target_size: int, readout_shape: List[int]) -> torch.Tensor:
        """Process images rescaled to target absolute size."""
        image_shape = (x.shape[2], x.shape[3])
        maximal_side = max(image_shape)
        factor = target_size / maximal_side
        target_shape = [math.ceil(image_shape[0] * factor), math.ceil(image_shape[1] * factor)]

        x_rescaled = F.interpolate(x, target_shape, mode='bicubic')

        with torch.no_grad():
            activations = self.backbone(x_rescaled)

        activations = [item.to(torch.float32) for item in activations]
        activations = [F.interpolate(item, readout_shape, mode=self.feature_interpolation_mode) for item in activations]

        return torch.cat(activations, dim=1)

    def forward(self, x: torch.Tensor, dataset_index: Optional[torch.Tensor],
                pixel_per_dva: List[float]) -> torch.Tensor:
        orig_shape = x.shape
        readout_shape = [math.ceil(orig_shape[2] / self.readout_factor), math.ceil(orig_shape[3] / self.readout_factor)]

        # Compute normalized weights from log space
        pixel_per_dva_weights = torch.exp(self.pixel_per_dva_weights)
        size_weights = torch.exp(self.size_weights)

        if dataset_index is None:
            # Average weights across all datasets
            dataset_index = torch.zeros(orig_shape[0], dtype=torch.long, device=x.device)
            pixel_per_dva_weights = pixel_per_dva_weights.mean(dim=1, keepdim=True)
            size_weights = size_weights.mean(dim=1, keepdim=True)

        # Normalize weights to sum to 1
        weight_sum = pixel_per_dva_weights.sum(dim=0, keepdim=True) + size_weights.sum(dim=0, keepdim=True)
        pixel_per_dva_weights = pixel_per_dva_weights / weight_sum
        size_weights = size_weights / weight_sum

        backbone_activations = None

        # Process pixel-per-dva scales
        for weight, target_pixel_per_dva in zip(pixel_per_dva_weights, self.resolutions_pixel_per_dva):
            resolution_features = self._process_pixel_per_dva(x, pixel_per_dva, target_pixel_per_dva, readout_shape)
            this_weight = weight[dataset_index].view(-1, 1, 1, 1)
            if backbone_activations is None:
                backbone_activations = this_weight * resolution_features
            else:
                backbone_activations = backbone_activations + this_weight * resolution_features

        # Process size scales
        for weight, size in zip(size_weights, self.resolutions_size):
            resolution_features = self._process_size(x, size, readout_shape)
            this_weight = weight[dataset_index].view(-1, 1, 1, 1)
            if backbone_activations is None:
                backbone_activations = this_weight * resolution_features
            else:
                backbone_activations = backbone_activations + this_weight * resolution_features

        return backbone_activations

    def train(self, mode: bool = True):
        super().train(mode=mode)
        # Keep backbone frozen
        self.backbone.eval()


class _BackboneConcatenator(nn.Module):
    """Concatenates features from multiple backbone feature extractors."""

    def __init__(self, feature_extractors: OrderedDict):
        super().__init__()
        self.feature_extractors = nn.ModuleDict(feature_extractors)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        all_features = []
        for extractor in self.feature_extractors.values():
            features = extractor(x)
            all_features.extend(features)
        return all_features


def _build_backbone() -> _BackboneConcatenator:
    """Build the CLIP + DINOv2 backbone used by DeepGaze MSDB."""
    # CLIP ResNet50x64
    # CLIPResNet50x64 is Sequential(normalizer, visual_clip_model)
    # Target '1.layer4.2.conv2' refers to visual_clip_model.layer4[2].conv2
    clip_features = CLIPResNet50x64()
    clip_extractor = FeatureExtractor(
        clip_features,
        targets=['1.layer4.2.conv2']
    )

    # DINOv2 ViT-B/14
    # DINOv2_ViTB14 is Sequential(normalizer, dinov2_vitb14)
    # Targets '1.blocks.10', '1.blocks.6' refer to dinov2_vitb14.blocks[10], blocks[6]
    dinov2_features = DINOv2_ViTB14()
    dinov2_extractor = DINOTransformersFeatureExtractor(
        dinov2_features,
        targets=['1.blocks.10', '1.blocks.6']
    )

    return _BackboneConcatenator(OrderedDict([
        ('CLIP', clip_extractor),
        ('DINOv2', dinov2_extractor),
    ]))


class DeepGazeMSDB(nn.Module):
    """DeepGaze MSDB (Multi-Scale Dataset Bias) saliency model.

    This model predicts fixation density maps from images. It combines features from
    CLIP ResNet50x64 and DINOv2 ViT-B/14 processed at multiple scales, with
    learned dataset-specific parameters.

    Args:
        pretrained: If True, load pretrained weights. Default: True.

    Example:
        >>> model = DeepGazeMSDB(pretrained=True)
        >>> # For known dataset (e.g., MIT1003)
        >>> log_density = model(image, centerbias, pixel_per_dva=35.0, dataset=MSDBDataset.MIT1003)
        >>> # For new/unknown dataset (uses averaged parameters)
        >>> log_density = model(image, centerbias, pixel_per_dva=21.7, dataset=None)

    Note:
        - `image`: RGB tensor of shape (B, 3, H, W) with values in [0, 255]
        - `centerbias`: Log density tensor of shape (B, H, W)
        - `pixel_per_dva`: Pixels per degree of visual angle (depends on display setup)
        - `dataset`: Dataset index (see MSDBDataset class) or None for averaged parameters

    Reference:
        Kümmerer, M., Khanuja, H., & Bethge, M. (2025). Modeling Saliency Dataset Bias.
        In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).
        https://openaccess.thecvf.com/content/ICCV2025/html/Kummerer_Modeling_Saliency_Dataset_Bias_ICCV_2025_paper.html
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Build backbone
        backbone = _build_backbone()

        # Multi-scale feature processing
        self.features = _MultiScaleBackbone(
            backbone=backbone,
            readout_factor=_READOUT_FACTOR,
            n_datasets=_N_DATASETS,
            resolutions_pixel_per_dva=_PIXEL_PER_DVA_SCALES,
            resolutions_size=_SIZE_SCALES,
            feature_interpolation_mode='bilinear',
        )

        # Freeze backbone
        for param in self.features.backbone.parameters():
            param.requires_grad = False
        self.features.backbone.eval()

        # Saliency network
        self.saliency_network = _build_saliency_network(_INPUT_CHANNELS)

        # Finalizer
        self.finalizer = _DatasetAwareFinalizer(sigma=1.0, n_datasets=_N_DATASETS)

        # Load pretrained weights (strict=False because backbone weights are already
        # loaded by CLIP/DINOv2, checkpoint only contains head weights)
        if pretrained:
            self.load_state_dict(
                model_zoo.load_url(
                    'https://github.com/matthias-k/DeepGaze/releases/download/v1.2.0/deepgazemsdb.pth',
                    map_location=torch.device('cpu')
                ),
                strict=False
            )

    def forward(self, image: torch.Tensor, centerbias: torch.Tensor,
                pixel_per_dva: float, dataset: Optional[int] = None) -> torch.Tensor:
        """Predict fixation log density map.

        Args:
            image: RGB image tensor of shape (B, 3, H, W) with values in [0, 255].
            centerbias: Log density centerbias of shape (B, H, W).
            pixel_per_dva: Pixels per degree of visual angle for the display setup.
            dataset: Dataset index (0-4, see MSDBDataset class) for dataset-specific
                    parameters, or None to use averaged parameters for generalization.

        Returns:
            Log density map of shape (B, H, W), normalized to sum to 1 in probability space.
        """
        orig_shape = image.shape
        image = image.to(torch.float32)

        # Prepare pixel_per_dva as list for batch processing
        if isinstance(pixel_per_dva, (int, float)):
            pixel_per_dva_list = [pixel_per_dva] * orig_shape[0]
        else:
            pixel_per_dva_list = list(pixel_per_dva)

        # Prepare dataset indices
        if dataset is not None:
            if isinstance(dataset, int):
                dataset_indices = torch.full((orig_shape[0],), dataset, dtype=torch.long, device=image.device)
            else:
                dataset_indices = dataset.to(image.device)
        else:
            dataset_indices = None

        # Compute sigma scaling factors
        sigma_scaling_factors = [dva / _SALIENCY_MAP_FACTOR for dva in pixel_per_dva_list]

        # Extract multi-scale features
        x = self.features(image, dataset_index=dataset_indices, pixel_per_dva=pixel_per_dva_list)

        # Saliency network
        x = self.saliency_network(x)

        # Interpolate to saliency map resolution
        saliency_shape = [math.ceil(orig_shape[2] / _SALIENCY_MAP_FACTOR), math.ceil(orig_shape[3] / _SALIENCY_MAP_FACTOR)]
        x = F.interpolate(x, saliency_shape, mode='bilinear')

        # Finalize (gaussian blur, add centerbias, normalize)
        x = self.finalizer(x, centerbias, sigma_scaling_factors, dataset_indices=dataset_indices)

        return x

    def train(self, mode: bool = True):
        """Set training mode, keeping backbone frozen."""
        self.features.train(mode=mode)
        self.saliency_network.train(mode=mode)
        self.finalizer.train(mode=mode)
