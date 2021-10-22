import random

import cv2
import torch
from torch.distributions.normal import Normal
from torch.distributions.poisson import Poisson
import numpy as np
import matplotlib.pylab as plt

from mmtrack.core.utils.metrics import psnr, ssim, normalize
from mmtrack.core.utils.visualization import plot_noise_clean_pairs


def gaussian_noise(clean, mode='random',
                   am=0.8, var=2500.0):
    device = clean.device
    if mode == 'random':
        am = random.choice([0.7, 0.8, 0.9])
        var = random.choice([2500, 3600, 4900, 6400, 8100, 10000])
    clean = clean * am
    var = torch.tensor(var, dtype=torch.float32).to(device)
    return Normal(loc=clean, scale=torch.sqrt(var)).sample()


def possion_gaussian_noise(clean, mode='random',
                           am=0.8, p_mean=10.0, g_var=1600.0):
    device = clean.device
    if mode == 'random':
        am = random.choice([0.7, 0.8, 0.9])
        p_mean = random.choice([25, 50, 75, 100, 125, 150, 175, 200])
        g_var = random.choice([1600, 2500, 3600, 4900, 6400, 8100])
    clean = clean * am
    p_mean = torch.tensor(p_mean, dtype=torch.float32).to(device)
    g_var = torch.tensor(g_var, dtype=torch.float32).to(device)
    possion = p_mean * Poisson(clean / p_mean).sample()
    gaussian = Normal(loc=possion, scale=torch.sqrt(g_var)).sample()
    return gaussian


def real_camera_noise_a7s3(clean, mode='random',
                           am=0.8, k_ratio=200, read_ratio=30):
    device = clean.device
    t, h, w, c = clean.shape

    if mode == 'random':
        am = random.choice([0.7, 0.8, 0.9])
        k_ratio = random.choice([25, 30, 35])
        read_ratio = random.choice([200, 250, 300])

    k = torch.tensor([[0.6015, 0.3724, 0.7122]]).reshape([1, 1, 1, 3])\
        .repeat(t, h, w, 1).to(device) * k_ratio
    var_beta = torch.tensor([[0.0055, 0.0044, 0.0064]])\
        .reshape([1, 1, 1, 3]).repeat(t, h, 1, 1).to(device)
    var_read = torch.tensor([[0.2066, 0.1303, 0.0763]])\
        .reshape([1, 1, 1, 3]).repeat(t, h, w, 1).to(device) * read_ratio
    n = torch.tensor([[0.4120, 0.6862, 0.3422]])\
        .reshape([1, 1, 1, 3]).repeat(t, h, w, 1).to(device)

    # add noise
    dark_img = clean * am  # darken clean image
    shot = Poisson(rate=dark_img/k).sample()
    dark = Poisson(rate=n).sample()
    read = Normal(loc=0, scale=torch.sqrt(var_read)).sample()
    dsn = Normal(loc=1, scale=torch.sqrt(var_beta)).sample()
    noise = k * dsn.repeat(1, 1, w, 1) * (shot + dark + read)
    return noise


def real_camera_noie_a7s3_jpg(clean, mode=None,
                           am=0.8, k_ratio=200, read_ratio=30):
    device = clean.device
    t, h, w, c = clean.shape

    if mode == 'random':
        am = random.choice([0.7, 0.8, 0.9])
        k_ratio = random.choice([25, 30, 35])
        read_ratio = random.choice([200, 250, 300])

    k = torch.tensor([[2.036, 1.220, 2.578]]).reshape([1, 1, 1, 3])\
        .repeat(t, h, w, 1).to(device) * k_ratio
    var_beta = torch.tensor([[0.015, 0.004, 0.009]])\
        .reshape([1, 1, 1, 3]).repeat(t, h, 1, 1).to(device)
    var_read = torch.tensor([[0.120, 1.730, 0.145]])\
        .reshape([1, 1, 1, 3]).repeat(t, h, w, 1).to(device) * read_ratio
    n = torch.tensor([[0.355, 1.513, 0.517]])\
        .reshape([1, 1, 1, 3]).repeat(t, h, w, 1).to(device)

    # add noise
    dark_img = clean * am  # darken clean image
    shot = Poisson(rate=dark_img/k).sample()
    dark = Poisson(rate=n).sample()
    read = Normal(loc=0, scale=torch.sqrt(var_read)).sample()
    dsn = Normal(loc=1, scale=torch.sqrt(var_beta)).sample()
    noise = k * dsn.repeat(1, 1, w, 1) * (shot + dark + read)
    return noise


def add_noise_clean_pairs(clean, noise_type='gauss', noise_level=None):
    clean = clean.permute(0, 2, 3, 1).float()  # b,c,w,h -> b,w,h,c
    # clean = clean.float()  # b,c,w,h -> b,w,h,c
    clean = clean[..., [2, 1, 0]]  # bgr -> rgb

    if noise_type == 'gauss':
        noise = gaussian_noise(clean, **noise_level)
    elif noise_type == 'mix':
        noise = possion_gaussian_noise(clean, **noise_level)
    elif noise_type == 'a7s3':
        noise = real_camera_noise_a7s3(clean, **noise_level)
    elif noise_type == 'a7s3_jpg':
        noise = real_camera_noie_a7s3_jpg(clean, **noise_level)
    elif noise_type == 'no_add':
        noise = clean
    else:
        raise NameError(f'not support this type -> {noise_type}')

    noise = torch.clamp(noise, 0, 255.0)
    clean = torch.clamp(clean, 0, 255.0)

    # plot
    # plot_noise_clean_pairs(clean / 255.0, noise / 255.0)

    # clean = np.asarray(clean[0][..., [2, 1, 0]], dtype=np.uint8)   # for test
    # clean = np.asarray(clean[0][..., [2, 1, 0]].cpu(), dtype=np.uint8)
    # noise = np.asarray(noise[0][..., [2, 1, 0]], dtype=np.uint8)   # for test
    # noise = np.asarray(noise[0][..., [2, 1, 0]].cpu(), dtype=np.uint8)
    # cv2.imwrite('/home/dabingreat666/cvpr2022_result/a7s3噪声标定/image_for_test/raw/vid_srgb/clean_clean_image.JPG', clean)
    # cv2.imwrite('/home/dabingreat666/cvpr2022_result/a7s3噪声标定/image_for_test/raw/vid_srgb/clean_noise_image.JPG', noise)

    # psnr_ = psnr(noise, clean)
    # ssim_ = ssim(noise, clean)
    # print(f'psnr with noise and clean is {psnr_} !')
    # print(f'ssim with noise and clean is {ssim_} !')

    # normalize
    noise = noise.permute(0, 3, 1, 2).contiguous()
    clean = clean.permute(0, 3, 1, 2).contiguous()
    noise = normalize(noise, mean=[123.675, 116.28, 103.53],
                      std=[58.395, 57.12, 57.375])
    clean = normalize(clean, mean=[123.675, 116.28, 103.53],
                      std=[58.395, 57.12, 57.375])
    return noise, clean


if __name__ == '__main__':
    path = '/home/dabingreat666/cvpr2022_result/a7s3噪声标定/image_for_test/raw/vid_srgb/2.JPEG'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1000, 600))
    img = torch.from_numpy(img)
    img = img.unsqueeze(dim=0)
    # add_noise_clean_pairs(img, noise_type='gauss', noise_level=dict(mode=None, am=0.9, var=1600))
    # add_noise_clean_pairs(img, noise_type='mix', noise_level=dict(mode=None, am=0.8, p_mean=200.0, g_var=6400.0))
    add_noise_clean_pairs(img, noise_type='a7s3_jpg', noise_level=dict(mode=None, am=0.8, k_ratio=3.0, read_ratio=2000.0))
    # a7s3_jpg 0.9, 3.0, 800.0
    # a7s3     0.8, 200, 6400
