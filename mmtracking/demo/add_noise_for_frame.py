import random

import torch
from torch.distributions.normal import Normal
from torch.distributions.poisson import Poisson
import numpy as np


def add_noise_for_frames(frame, constant=[0.5, 0.5]):  # np array uint8
    clean = torch.from_numpy(frame).unsqueeze(0)
    bgr_clean = clean.float()  # dim bxcxwxh -> bxwxhxc
    bs, h, w, c = bgr_clean.size()

    if isinstance(constant, (tuple, list)) and len(constant) == 2:
        ratio = [constant[0] for _ in range(bs)]
        am = constant[1]
    else:
        _ratio = random.uniform(0, 1)
        ratio = [_ratio for _ in range(bs)]
        am = random.uniform(0, 1)

    poisson_tensor = torch.tensor([[10.4304723, 9.05125669, 16.68142166, 8.92170499]]) \
        .reshape([1, 1, 1, 4]).repeat(bs, h, w, 1)
    gaussian_tensor = torch.tensor([92.5, 80, 286, 80]).reshape([1, 1, 1, 4]).repeat(bs, h, w, 1)
    wi_tensor = torch.tensor([0.08, 0.035, 0.25, 0.035]).reshape([1, 1, 4]).repeat(bs, h, 1)
    ratio_tensor1 = torch.tensor(ratio).reshape(bs, 1, 1, 1).repeat(1, h, w, 4)
    ratio_tensor2 = torch.tensor(ratio).reshape(bs, 1, 1).repeat(1, h, 4)

    poisson_tensorOP = poisson_tensor * ratio_tensor1
    gaussian_tensorOP = gaussian_tensor * ratio_tensor1
    wi_tensorOP = wi_tensor * ratio_tensor2 ** 3

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

    rgb_noise = torch.clamp(torch.cat([_b, _g, _r], 3), 0, 255.0)
    rgb_clean = torch.clamp(torch.cat([b, g, r], 3), 0, 255.0)

    # plot clean and noise imgs
    # plot_noise_clean_pairs(rgb_clean/255.0, rgb_noise/255.0)

    noise = rgb_noise.squeeze(0)
    clean = rgb_clean.squeeze(0)

    noise, clean = np.asarray(noise, dtype=np.uint8), np.asarray(clean, dtype=np.uint8)
    return noise, clean
