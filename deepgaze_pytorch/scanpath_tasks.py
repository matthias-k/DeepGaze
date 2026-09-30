"""Adapters that plug scanpath models into ``scanpath_training`` (image-grouped training/evaluation).

Each task computes the per-image part once (``image_context``) and the unnormalized log density for a
chunk of fixation histories (``pre_log_density``). A task may return a list of maps, one per mixture
component; they are combined as an equal-weight mixture in probability space.
"""
import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from .scanpath_utils import encode_history_pixels, mixture_full_log_density


class DeepGazeIIITask:
    """DeepGaze III (``modules.DeepGazeIII`` or the released ``DeepGazeIIIMixture``) with the frozen
    backbone evaluated once per image. Reproduces ``forward`` exactly.

    Args:
        model: the model.
        load_image: index -> (1, 3, H, W) float tensor in [0, 255] on the device.
        load_centerbias: index -> (1, H, W) center-bias log density on the device.
        components: for a ``DeepGazeIIIMixture``, the components to use (default: all).
    """

    def __init__(self, model, load_image: Callable, load_centerbias: Callable, components: Optional[Sequence[int]] = None):
        self.model = model
        self.load_image = load_image
        self.load_centerbias = load_centerbias
        self.mixture = hasattr(model, 'saliency_networks')
        if self.mixture:
            self.components = list(range(len(model.saliency_networks))) if components is None else list(components)
        elif components is not None:
            raise ValueError("components only apply to a DeepGazeIIIMixture")
        # a spatial DeepGaze III (no scanpath network) predicts the same map for every fixation
        self.history_independent = all(part[1] is None for part in self._parts())

    def _parts(self):
        m = self.model
        if not self.mixture:
            return [(m.saliency_network, m.scanpath_network, m.fixation_selection_network, m.finalizer)]
        return [(m.saliency_networks[c], m.scanpath_networks[c], m.fixation_selection_networks[c], m.finalizers[c])
                for c in self.components]

    def image_context(self, item):
        m = self.model
        image = self.load_image(item.index)
        centerbias = self.load_centerbias(item.index)
        height, width = image.shape[-2:]
        # same calls as DeepGazeIII.forward / DeepGazeIIIMixture.forward
        if self.mixture:
            x = F.interpolate(image, scale_factor=1 / m.downsample, recompute_scale_factor=False)
        else:
            x = F.interpolate(image, scale_factor=1 / m.downsample)
        with torch.no_grad():
            features = m.features(x)
        readout_shape = [math.ceil(height / m.downsample / m.readout_factor), math.ceil(width / m.downsample / m.readout_factor)]
        features = torch.cat([F.interpolate(item_, readout_shape) for item_ in features], dim=1)
        parts = []
        for saliency_network, _, _, finalizer in self._parts():
            parts.append((saliency_network(features), finalizer.downscale_centerbias(centerbias)))
        return {'parts': parts, 'image_size': (height, width), 'readout_shape': readout_shape}

    def pre_log_density(self, context, x_hist, y_hist):
        batch = x_hist.shape[0]
        scanpath_features = encode_history_pixels(x_hist, y_hist, context['image_size'], context['readout_shape'])
        outputs = []
        for (saliency, downscaled_centerbias), (_, scanpath_network, selection_network, finalizer) in zip(context['parts'], self._parts()):
            y = scanpath_network(scanpath_features) if scanpath_network is not None else None
            x = selection_network((saliency.expand(batch, -1, -1, -1), y))
            outputs.append(finalizer.combine(x, downscaled_centerbias.expand(batch, -1, -1)))
        return outputs if self.mixture else outputs[0]


class MSDBScanpathTask:
    """``DeepGazeIIIMSDB`` on precomputed MSDB priority maps (training) or with its backbone (evaluation).

    Args:
        model: a ``DeepGazeIIIMSDB``.
        load_centerbias: index -> (1, H, W) center-bias log density on the device.
        pixel_per_dva: presentation resolution of the dataset.
        dataset: MSDB dataset slot or None for the averaged parameters.
        saliency_maps: index -> (1, 1, h, w) precomputed priority map; if None the backbone is used
            with ``load_image`` (index -> (1, 3, H, W) in [0, 255]).
    """

    def __init__(self, model, load_centerbias: Callable, pixel_per_dva: float, dataset: Optional[int],
                 saliency_maps: Optional[Callable] = None, load_image: Optional[Callable] = None):
        if saliency_maps is None and load_image is None:
            raise ValueError("need precomputed saliency maps or an image loader")
        self.model = model
        self.load_centerbias = load_centerbias
        self.pixel_per_dva = pixel_per_dva
        self.dataset = dataset
        self.saliency_maps = saliency_maps
        self.load_image = load_image

    def image_context(self, item):
        centerbias = self.load_centerbias(item.index)
        if self.saliency_maps is not None:
            saliency = self.saliency_maps(item.index)
        else:
            with torch.no_grad():
                saliency = self.model.saliency(self.load_image(item.index), self.pixel_per_dva, self.dataset)
        return {'saliency': saliency, 'centerbias': centerbias}

    def pre_log_density(self, context, x_hist, y_hist):
        return self.model.pre_log_density(context['saliency'], context['centerbias'], x_hist, y_hist,
                                          self.pixel_per_dva, self.dataset)


class RescaledTask:
    """Evaluate ``task`` at a different image scale and return log densities on the original grid.

    The inner task's loaders must provide the rescaled images and center biases (so that, e.g., a model
    trained at 35 pixels per degree sees images at that resolution). Histories are rescaled to that
    image, and the predicted probability mass is averaged back into the original pixels and
    renormalized.
    """

    def __init__(self, task, scale_y: float, scale_x: float):
        self.task = task
        self.model = task.model
        self.scale_y = scale_y
        self.scale_x = scale_x

    def image_context(self, item):
        return self.task.image_context(item)

    def log_density_maps(self, context, x_hist, y_hist, image_size):
        pre = self.task.pre_log_density(context, x_hist * self.scale_x, y_hist * self.scale_y)
        maps = mixture_full_log_density(pre, context['image_size'])
        pooled = F.adaptive_avg_pool2d(maps.exp()[:, None], tuple(image_size))[:, 0]
        return torch.log(pooled) - torch.log(pooled.sum(dim=(1, 2), keepdim=True))
