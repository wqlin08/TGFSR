

import torch
import torch.nn as nn


def psd_loss(sr_img, hr_img):

    sr_fft = torch.fft.rfft2(sr_img, norm='backward')
    hr_fft = torch.fft.rfft2(hr_img, norm='backward')

    sr_psd = torch.abs(sr_fft) ** 2
    hr_psd = torch.abs(hr_fft) ** 2

    loss_func = nn.L1Loss()

    loss = loss_func(torch.log(sr_psd + 1e-8), torch.log(hr_psd + 1e-8))

    return loss