# pylint: disable=missing-module-docstring,invalid-name
# pylint: disable=missing-docstring
# pylint: disable=line-too-long

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization`_ .

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = nn.LayerNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = nn.LayerNorm(10)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450
    """
    __constants__ = ['features', 'weight', 'bias', 'eps', 'center', 'scale']

    def __init__(self, features, eps=1e-12, center=True, scale=True):
        super(LayerNorm, self).__init__()
        self.features = features
        self.eps = eps
        self.center = center
        self.scale = scale

        if self.scale:
            self.weight = nn.Parameter(torch.Tensor(self.features))
        else:
            self.register_parameter('weight', None)

        if self.center:
            self.bias = nn.Parameter(torch.Tensor(self.features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.scale:
            nn.init.ones_(self.weight)

        if self.center:
            nn.init.zeros_(self.bias)

    def adjust_parameter(self, tensor, parameter):
        return torch.repeat_interleave(
            torch.repeat_interleave(
                parameter.view(-1, 1, 1),
                repeats=tensor.shape[2],
                dim=1),
            repeats=tensor.shape[3],
            dim=2
        )

    def forward(self, input):
        normalized_shape = (self.features, input.shape[2], input.shape[3])
        weight = self.adjust_parameter(input, self.weight)
        bias = self.adjust_parameter(input, self.bias)
        return F.layer_norm(
            input, normalized_shape, weight, bias, self.eps)

    def extra_repr(self):
        return '{features}, eps={eps}, ' \
            'center={center}, scale={scale}'.format(**self.__dict__)


def gaussian_filter_1d(tensor, dim, sigma, truncate=4, kernel_size=None, padding_mode='replicate', padding_value=0.0):
    sigma = torch.as_tensor(sigma, device=tensor.device, dtype=tensor.dtype)

    if kernel_size is not None:
        kernel_size = torch.as_tensor(kernel_size, device=tensor.device, dtype=torch.int64)
    else:
        kernel_size = torch.as_tensor(2 * torch.ceil(truncate * sigma) + 1, device=tensor.device, dtype=torch.int64)

    kernel_size = kernel_size.detach()

    kernel_size_int = kernel_size.detach().cpu().numpy()

    mean = (torch.as_tensor(kernel_size, dtype=tensor.dtype) - 1) / 2

    grid = torch.arange(kernel_size, device=tensor.device) - mean

    kernel_shape = (1, 1, kernel_size)
    grid = grid.view(kernel_shape)

    grid = grid.detach()

    source_shape = tensor.shape

    tensor = torch.movedim(tensor, dim, len(source_shape)-1)
    dim_last_shape = tensor.shape
    assert tensor.shape[-1] == source_shape[dim]

    # we need reshape instead of view for batches like B x C x H x W
    tensor = tensor.reshape(-1, 1, source_shape[dim])

    padding = (math.ceil((kernel_size_int - 1) / 2), math.ceil((kernel_size_int - 1) / 2))
    tensor_ = F.pad(tensor, padding, padding_mode, padding_value)

    # create gaussian kernel from grid using current sigma
    kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # convolve input with gaussian kernel
    tensor_ = F.conv1d(tensor_, kernel)
    tensor_ = tensor_.view(dim_last_shape)
    tensor_ = torch.movedim(tensor_, len(source_shape)-1, dim)

    assert tensor_.shape == source_shape

    return tensor_


class GaussianFilterNd(nn.Module):
    """A differentiable gaussian filter"""

    def __init__(self, dims, sigma, truncate=4, kernel_size=None, padding_mode='replicate', padding_value=0.0,
                 trainable=False):
        """Creates a 1d gaussian filter

        Args:
            dims ([int]): the dimensions to which the gaussian filter is applied. Negative values won't work
            sigma (float): standard deviation of the gaussian filter (blur size)
            input_dims (int, optional): number of input dimensions ignoring batch and channel dimension,
                i.e. use input_dims=2 for images (default: 2).
            truncate (float, optional): truncate the filter at this many standard deviations (default: 4.0).
                This has no effect if the `kernel_size` is explicitely set
            kernel_size (int): size of the gaussian kernel convolved with the input
            padding_mode (string, optional): Padding mode implemented by `torch.nn.functional.pad`.
            padding_value (string, optional): Value used for constant padding.
        """
        # IDEA determine input_dims dynamically for every input
        super(GaussianFilterNd, self).__init__()

        self.dims = dims
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32), requires_grad=trainable)  # default: no optimization
        self.truncate = truncate
        self.kernel_size = kernel_size

        # setup padding
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def forward(self, tensor):
        """Applies the gaussian filter to the given tensor"""
        for dim in self.dims:
            tensor = gaussian_filter_1d(
                tensor,
                dim=dim,
                sigma=self.sigma,
                truncate=self.truncate,
                kernel_size=self.kernel_size,
                padding_mode=self.padding_mode,
                padding_value=self.padding_value,
            )

        return tensor


class Conv2dMultiInput(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        for k, _in_channels in enumerate(in_channels):
            if _in_channels:
                setattr(self, f'conv_part{k}', nn.Conv2d(_in_channels, out_channels, kernel_size, bias=bias))

    def forward(self, tensors):
        assert len(tensors) == len(self.in_channels)

        out = None
        for k, (count, tensor) in enumerate(zip(self.in_channels, tensors)):
            if not count:
                continue
            _out = getattr(self, f'conv_part{k}')(tensor)

            if out is None:
                out = _out
            else:
                out += _out

        return out

#    def extra_repr(self):
#        return f'{self.in_channels}'


class LayerNormMultiInput(nn.Module):
    __constants__ = ['features', 'weight', 'bias', 'eps', 'center', 'scale']

    def __init__(self, features, eps=1e-12, center=True, scale=True):
        super().__init__()
        self.features = features
        self.eps = eps
        self.center = center
        self.scale = scale

        for k, _features in enumerate(features):
            if _features:
                setattr(self, f'layernorm_part{k}', LayerNorm(_features, eps=eps, center=center, scale=scale))

    def forward(self, tensors):
        assert len(tensors) == len(self.features)

        out = []
        for k, (count, tensor) in enumerate(zip(self.features, tensors)):
            if not count:
                assert tensor is None
                out.append(None)
                continue
            out.append(getattr(self, f'layernorm_part{k}')(tensor))

        return out


class Bias(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, tensor):
        return tensor + self.bias[np.newaxis, :, np.newaxis, np.newaxis]

    def extra_repr(self):
        return f'channels={self.channels}'


class SelfAttention(nn.Module):
    """ Self attention Layer

    adapted from https://discuss.pytorch.org/t/attention-in-image-classification/80147/3
    """

    def __init__(self, in_channels, out_channels=None, key_channels=None, activation=None, skip_connection_with_convolution=False, return_attention=True):
        super().__init__()
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        if key_channels is None:
            key_channels = in_channels // 8
        self.key_channels = key_channels
        self.activation = activation
        self.skip_connection_with_convolution = skip_connection_with_convolution
        if not self.skip_connection_with_convolution:
            if self.out_channels != self.in_channels:
                raise ValueError("out_channels has to be equal to in_channels with true skip connection!")
        self.return_attention = return_attention

        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=key_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=key_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.skip_connection_with_convolution:
            self.skip_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, self.out_channels, width, height)

        if self.skip_connection_with_convolution:
            skip_connection = self.skip_conv(x)
        else:
            skip_connection = x
        out = self.gamma * out + skip_connection

        if self.activation is not None:
            out = self.activation(out)

        if self.return_attention:
            return out, attention

        return out


class MultiHeadSelfAttention(nn.Module):
    """ Self attention Layer

    adapted from https://discuss.pytorch.org/t/attention-in-image-classification/80147/3
    """

    def __init__(self, in_channels, heads, out_channels=None, key_channels=None, activation=None, skip_connection_with_convolution=False):
        super().__init__()
        self.heads = heads
        self.heads = nn.ModuleList([SelfAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            key_channels=key_channels,
            activation=activation,
            skip_connection_with_convolution=skip_connection_with_convolution,
            return_attention=False,
        ) for _ in range(heads)])

    def forward(self, tensor):
        outs = [head(tensor) for head in self.heads]
        out = torch.cat(outs, dim=1)
        return out


class FlexibleScanpathHistoryEncoding(nn.Module):
    """
    a convolutional layer which works for different numbers of previous fixations.

    Nonexistent fixations will deactivate the respective convolutions
    the bias will be added per fixation (if the given fixation is present)
    """
    def __init__(self, in_fixations, channels_per_fixation, out_channels, kernel_size, bias=True,):
        super().__init__()
        self.in_fixations = in_fixations
        self.channels_per_fixation = channels_per_fixation
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.convolutions = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.channels_per_fixation,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                bias=self.bias
            ) for i in range(in_fixations)
        ])

    def forward(self, tensor):
        results = None
        valid_fixations = ~torch.isnan(
            tensor[:, :self.in_fixations, 0, 0]
        )
        # print("valid fix", valid_fixations)

        for fixation_index in range(self.in_fixations):
            valid_indices = valid_fixations[:, fixation_index]
            if not torch.any(valid_indices):
                continue
            this_input = tensor[
                valid_indices,
                fixation_index::self.in_fixations
            ]
            this_result = self.convolutions[fixation_index](
                this_input
            )
            # TODO: This will break if all data points
            # in the batch don't have a single fixation
            # but that's not a case I intend to train
            # anyway.
            if results is None:
                b, _, _, _ = tensor.shape
                _, _, h, w = this_result.shape
                results = torch.zeros(
                    (b, self.out_channels, h, w),
                    dtype=tensor.dtype,
                    device=tensor.device
                )
            results[valid_indices] += this_result

        return results
