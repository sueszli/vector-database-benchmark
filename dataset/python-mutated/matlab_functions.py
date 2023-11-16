import math
import numpy as np
import torch

def cubic(x):
    if False:
        return 10
    'cubic function used for calculate_weights_indices.'
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    a = 1.5 * absx3 - 2.5 * absx2 + 1
    b = (absx <= 1).type_as(absx)
    c = -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
    return a * b + c * ((absx > 1) * (absx <= 2)).type_as(absx)

def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if False:
        i = 10
        return i + 15
    'Calculate weights and indices, used for imresize function.\n    Args:\n        in_length (int): Input length.\n        out_length (int): Output length.\n        scale (float): Scale factor.\n        kernel_width (int): Kernel width.\n        antialiasing (bool): Whether to apply anti-aliasing when downsampling.\n    '
    if scale < 1 and antialiasing:
        kernel_width = kernel_width / scale
    x = torch.linspace(1, out_length, out_length)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = torch.floor(u - kernel_width / 2)
    p = math.ceil(kernel_width) + 2
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(out_length, p)
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices
    if scale < 1 and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)
    weights_zero_tmp = torch.sum(weights == 0, 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-06):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-06):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return (weights, indices, int(sym_len_s), int(sym_len_e))

@torch.no_grad()
def imresize(img, scale, antialiasing=True):
    if False:
        print('Hello World!')
    'imresize function same as MATLAB.\n    It now only supports bicubic.\n    The same scale applies for both height and width.\n    Args:\n        img (Tensor | Numpy array):\n            Tensor: Input image with shape (c, h, w), [0, 1] range.\n            Numpy: Input image with shape (h, w, c), [0, 1] range.\n        scale (float): Scale factor. The same scale applies for both height\n            and width.\n        antialiasing (bool): Whether to apply anti-aliasing when downsampling.\n            Default: True.\n    Returns:\n        Tensor: Output image with shape (c, h, w), [0, 1] range, w/o round.\n    '
    squeeze_flag = False
    if type(img).__module__ == np.__name__:
        numpy_type = True
        if img.ndim == 2:
            img = img[:, :, None]
            squeeze_flag = True
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    else:
        numpy_type = False
        if img.ndim == 2:
            img = img.unsqueeze(0)
            squeeze_flag = True
    (in_c, in_h, in_w) = img.size()
    (out_h, out_w) = (math.ceil(in_h * scale), math.ceil(in_w * scale))
    kernel_width = 4
    kernel = 'cubic'
    (weights_h, indices_h, sym_len_hs, sym_len_he) = calculate_weights_indices(in_h, out_h, scale, kernel, kernel_width, antialiasing)
    (weights_w, indices_w, sym_len_ws, sym_len_we) = calculate_weights_indices(in_w, out_w, scale, kernel, kernel_width, antialiasing)
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(img)
    sym_patch = img[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)
    sym_patch = img[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)
    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)
    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)
    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)
    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_w[i])
    if squeeze_flag:
        out_2 = out_2.squeeze(0)
    if numpy_type:
        out_2 = out_2.numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)
    return out_2