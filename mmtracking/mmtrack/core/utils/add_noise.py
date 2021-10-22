import os
import random

import torch
from torch.distributions.normal import Normal
from torch.distributions.poisson import Poisson
import numpy as np
import matplotlib.pylab as plt

from .metrics import psnr, ssim


def general_clean_noise_pairs(clean, constant=None):  # b,c,h,w
    device = clean.device
    bgr_clean = clean.permute(0, 2, 3, 1).float()  # dim bxcxwxh -> bxwxhxc
    bs, h, w, c = bgr_clean.size()

    if isinstance(constant, (tuple, list)) and len(constant) == 2:
        ratio = [constant[0] for _ in range(bs)]
        am = constant[1]
    else:
        _ratio = random.uniform(0, 1)
        ratio = [_ratio for _ in range(bs)]
        am = random.uniform(0, 1)

    poisson_tensor = torch.tensor([[10.4304723, 9.05125669, 16.68142166, 8.92170499]])\
        .reshape([1, 1, 1, 4]).repeat(bs, h, w, 1).to(device)
    gaussian_tensor = torch.tensor([92.5, 80, 286, 80]).reshape([1, 1, 1, 4]).repeat(bs, h, w, 1).to(device)
    wi_tensor = torch.tensor([0.08, 0.035, 0.25, 0.035]).reshape([1, 1, 4]).repeat(bs, h, 1).to(device)
    ratio_tensor1 = torch.tensor(ratio).reshape(bs, 1, 1, 1).repeat(1, h, w, 4).to(device)
    ratio_tensor2 = torch.tensor(ratio).reshape(bs, 1, 1).repeat(1, h, 4).to(device)

    poisson_tensorOP = poisson_tensor * ratio_tensor1
    gaussian_tensorOP = gaussian_tensor * ratio_tensor1
    wi_tensorOP = wi_tensor * ratio_tensor2**3

    b, g, r = torch.split(bgr_clean, [1, 1, 1], 3)
    raw = torch.cat([r, g, b, g], 3) * am

    # add poisson
    peak = raw / poisson_tensorOP
    pnoisy = poisson_tensorOP * Poisson(rate=peak).sample()
    # add gaussian
    gnoisy = poisson_tensorOP * Normal(loc=0, scale=torch.sqrt(gaussian_tensorOP)).sample()
    # add width
    k_noisy = Normal(loc=1, scale=torch.sqrt(wi_tensorOP)).sample()
    # two noisy
    noisy = (pnoisy + gnoisy) * k_noisy.reshape([bs, h, 1, 4]).repeat(1, 1, w, 1)
    _r, _g, _b, _g = torch.split(noisy, [1, 1, 1, 1], 3)

    rgb_noise = torch.clamp(torch.cat([_r, _g, _b], 3), 0, 255.0)
    rgb_clean = torch.clamp(torch.cat([r, g, b], 3), 0, 255.0)

    # plot clean and noise imgs
    # plot_noise_clean_pairs(rgb_clean/255.0, rgb_noise/255.0)
    clean = np.asarray(rgb_clean[0].cpu(), dtype=np.uint8)
    noise = np.asarray(rgb_noise[0].cpu(), dtype=np.uint8)
    plt.imsave('./plot/clean_clean_image.jpg', clean)
    plt.imsave('./plot/clean_noise_image.jpg', noise)
    psnr_ = psnr(noise, clean)
    ssim_ = ssim(noise, clean)
    print(f'psnr -> {psnr_}')
    print(f'ssim -> {ssim_}')

    noise = rgb_noise.permute(0, 3, 1, 2).contiguous()
    clean = rgb_clean.permute(0, 3, 1, 2).contiguous()

    # noise = normalize(noise, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    # clean = normalize(clean, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

    noise = normalize(noise, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])
    clean = normalize(clean, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])

    return noise, clean


def normalize(tensor, mean, std, inplace=True):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor


def plot_noise_clean_pairs(clean, noise, save_dir='./plot/clean_noise_image.jpg'):
    bs = clean.shape[0]
    fig = plt.figure(figsize=(10, bs*4+2))
    clean, noise = np.asarray(clean.cpu()), np.asarray(noise.cpu())
    for b in range(bs):
        # clean
        cl = plt.subplot2grid((bs, 2), (b, 0))
        # cl = cl.axis('off')
        cl.set_title(f'bs{b}_clean')
        icl = cl.imshow(clean[b, ...])

        # noise
        no = plt.subplot2grid((bs, 2), (b, 1))
        # no = no.axis('off')
        no.set_title(f'bs{b}_noise')
        ino = no.imshow(noise[b, ...])

        fig.colorbar(ino, ax=[cl, no])
    plt.savefig(save_dir, dpi=150)