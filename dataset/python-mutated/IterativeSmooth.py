import math
import os
import numpy as np
import torch
import torch.nn as nn

def gauss(t, r=0, window_size=3):
    if False:
        while True:
            i = 10
    '\n    @param window_size is the size of window over which gaussian to be applied\n    @param t is the index of current point\n    @param r is the index of point in window\n\n    @return guassian weights over a window size\n    '
    if np.abs(r - t) > window_size:
        return 0
    else:
        return np.exp(-9 * (r - t) ** 2 / window_size ** 2)

def generateSmooth(originPath, kernel=None, repeat=20):
    if False:
        return 10
    smooth = originPath
    temp_smooth_3 = originPath[:, :, 3:-3, :, :]
    kernel = kernel
    if kernel is None:
        kernel = torch.Tensor([gauss(i) for i in range(-3, 4)]).to(originPath.device)
        kernel = torch.cat([kernel[:3], kernel[4:]])
        kernel = kernel.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        kernel = kernel.repeat(*originPath.shape)
    abskernel = torch.abs(kernel)
    lambda_t = 100
    for _ in range(repeat):
        temp_smooth = torch.zeros_like(smooth, device=smooth.device)
        temp_smooth_0 = smooth[:, :, 0:-6, :, :] * kernel[:, 0:1, 3:-3, :, :] * lambda_t
        temp_smooth_1 = smooth[:, :, 1:-5, :, :] * kernel[:, 1:2, 3:-3, :, :] * lambda_t
        temp_smooth_2 = smooth[:, :, 2:-4, :, :] * kernel[:, 2:3, 3:-3, :, :] * lambda_t
        temp_smooth_4 = smooth[:, :, 4:-2, :, :] * kernel[:, 3:4, 3:-3, :, :] * lambda_t
        temp_smooth_5 = smooth[:, :, 5:-1, :, :] * kernel[:, 4:5, 3:-3, :, :] * lambda_t
        temp_smooth_6 = smooth[:, :, 6:, :, :] * kernel[:, 5:6, 3:-3, :, :] * lambda_t
        temp_value_01 = 1 + lambda_t * torch.sum(abskernel[:, :, 3:-3, :, :], dim=1, keepdim=True)
        temp_smooth[:, :, 3:-3, :, :] = (temp_smooth_0 + temp_smooth_1 + temp_smooth_2 + temp_smooth_3 + temp_smooth_4 + temp_smooth_5 + temp_smooth_6) / temp_value_01
        temp = smooth[:, :, 1:4, :, :]
        temp_smooth[:, :, 0, :, :] = (torch.sum(kernel[:, 3:, 0, :, :].unsqueeze(1) * temp, 2) * lambda_t + originPath[:, :, 0, :, :]) / (1 + lambda_t * torch.sum(abskernel[:, 3:, 0, :, :].unsqueeze(1), 2))
        temp = torch.cat([smooth[:, :, :1, :, :], smooth[:, :, 2:5, :, :]], 2)
        temp_smooth[:, :, 1, :, :] = (torch.sum(kernel[:, 2:, 1, :, :].unsqueeze(1) * temp, 2) * lambda_t + originPath[:, :, 1, :, :]) / (1 + lambda_t * torch.sum(abskernel[:, 2:, 1, :, :].unsqueeze(1), 2))
        temp = torch.cat([smooth[:, :, :2, :, :], smooth[:, :, 3:6, :, :]], 2)
        temp_smooth[:, :, 2, :, :] = (torch.sum(kernel[:, 1:, 2, :, :].unsqueeze(1) * temp, 2) * lambda_t + originPath[:, :, 2, :, :]) / (1 + lambda_t * torch.sum(abskernel[:, 1:, 2, :, :].unsqueeze(1), 2))
        temp = smooth[:, :, -4:-1]
        temp_value_11 = torch.sum(kernel[:, :3, -1, :, :].unsqueeze(1) * temp, 2)
        temp_value_08 = temp_value_11 * lambda_t + originPath[:, :, -1, :, :]
        temp_value_10 = torch.sum(abskernel[:, :3, -1, :, :].unsqueeze(1), 2)
        temp_value_09 = 1 + lambda_t * temp_value_10
        temp_smooth[:, :, -1, :, :] = temp_value_08 / temp_value_09
        temp = torch.cat([smooth[:, :, -5:-2, :, :], smooth[:, :, -1:, :, :]], 2)
        temp_value_07 = torch.sum(kernel[:, :4, -2, :, :].unsqueeze(1) * temp, 2)
        temp_value_04 = temp_value_07 * lambda_t + originPath[:, :, -2, :, :]
        temp_value_06 = torch.sum(abskernel[:, :4, -2, :, :].unsqueeze(1), 2)
        temp_value_05 = 1 + lambda_t * temp_value_06
        temp_smooth[:, :, -2, :, :] = temp_value_04 / temp_value_05
        temp = torch.cat([smooth[:, :, -6:-3, :, :], smooth[:, :, -2:, :, :]], 2)
        temp_value_02 = torch.sum(kernel[:, :5, -3, :, :].unsqueeze(1) * temp, 2) * lambda_t + originPath[:, :, -3, :, :]
        temp_value_03 = 1 + lambda_t * torch.sum(abskernel[:, :5, -3, :, :].unsqueeze(1), 2)
        temp_smooth[:, :, -3, :, :] = temp_value_02 / temp_value_03
        smooth = temp_smooth
    return smooth