from typing import Dict
import cv2
import numpy as np
import torch
from modelscope.metainfo import Metrics
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS, MetricKeys

@METRICS.register_module(group_key=default_group, module_name=Metrics.image_denoise_metric)
class ImageDenoiseMetric(Metric):
    """The metric computation class for image denoise classes.
    """
    pred_name = 'pred'
    label_name = 'target'

    def __init__(self):
        if False:
            return 10
        super(ImageDenoiseMetric, self).__init__()
        self.preds = []
        self.labels = []

    def add(self, outputs: Dict, inputs: Dict):
        if False:
            while True:
                i = 10
        ground_truths = outputs[ImageDenoiseMetric.label_name]
        eval_results = outputs[ImageDenoiseMetric.pred_name]
        self.preds.append(eval_results)
        self.labels.append(ground_truths)

    def evaluate(self):
        if False:
            while True:
                i = 10
        (psnr_list, ssim_list) = ([], [])
        for (pred, label) in zip(self.preds, self.labels):
            psnr_list.append(calculate_psnr(label[0], pred[0], crop_border=0))
            ssim_list.append(calculate_ssim(label[0], pred[0], crop_border=0))
        return {MetricKeys.PSNR: np.mean(psnr_list), MetricKeys.SSIM: np.mean(ssim_list)}

    def merge(self, other: 'ImageDenoiseMetric'):
        if False:
            return 10
        self.preds.extend(other.preds)
        self.labels.extend(other.labels)

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        return (self.preds, self.labels)

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        self.__init__()
        (self.preds, self.labels) = state

def reorder_image(img, input_order='HWC'):
    if False:
        print('Hello World!')
    "Reorder images to 'HWC' order.\n    If the input_order is (h, w), return (h, w, 1);\n    If the input_order is (c, h, w), return (h, w, c);\n    If the input_order is (h, w, c), return as it is.\n    Args:\n        img (ndarray): Input image.\n        input_order (str): Whether the input order is 'HWC' or 'CHW'.\n            If the input image shape is (h, w), input_order will not have\n            effects. Default: 'HWC'.\n    Returns:\n        ndarray: reordered image.\n    "
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def calculate_psnr(img1, img2, crop_border, input_order='HWC'):
    if False:
        return 10
    "Calculate PSNR (Peak Signal-to-Noise Ratio).\n    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio\n    Args:\n        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].\n        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].\n        crop_border (int): Cropped pixels in each edge of an image. These\n            pixels are not involved in the PSNR calculation.\n        input_order (str): Whether the input order is 'HWC' or 'CHW'.\n            Default: 'HWC'.\n        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.\n    Returns:\n        float: psnr result.\n    "
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    def _psnr(img1, img2):
        if False:
            print('Hello World!')
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_value = 1.0 if img1.max() <= 1 else 255.0
        return 20.0 * np.log10(max_value / np.sqrt(mse))
    return _psnr(img1, img2)

def calculate_ssim(img1, img2, crop_border, input_order='HWC', ssim3d=True):
    if False:
        i = 10
        return i + 15
    "Calculate SSIM (structural similarity).\n    Ref:\n    Image quality assessment: From error visibility to structural similarity\n    The results are the same as that of the official released MATLAB code in\n    https://ece.uwaterloo.ca/~z70wang/research/ssim/.\n    For three-channel images, SSIM is calculated for each channel and then\n    averaged.\n    Args:\n        img1 (ndarray): Images with range [0, 255].\n        img2 (ndarray): Images with range [0, 255].\n        crop_border (int): Cropped pixels in each edge of an image. These\n            pixels are not involved in the SSIM calculation.\n        input_order (str): Whether the input order is 'HWC' or 'CHW'.\n            Default: 'HWC'.\n        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.\n    Returns:\n        float: ssim result.\n    "
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    def _cal_ssim(img1, img2):
        if False:
            for i in range(10):
                print('nop')
        ssims = []
        max_value = 1 if img1.max() <= 1 else 255
        with torch.no_grad():
            final_ssim = _ssim_3d(img1, img2, max_value) if ssim3d else _ssim(img1, img2, max_value)
            ssims.append(final_ssim)
        return np.array(ssims).mean()
    return _cal_ssim(img1, img2)

def _ssim(img, img2, max_value):
    if False:
        i = 10
        return i + 15
    "Calculate SSIM (structural similarity) for one channel images.\n    It is called by func:`calculate_ssim`.\n    Args:\n        img (ndarray): Images with range [0, 255] with order 'HWC'.\n        img2 (ndarray): Images with range [0, 255] with order 'HWC'.\n    Returns:\n        float: SSIM result.\n    "
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    tmp1 = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    tmp2 = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = tmp1 / tmp2
    return ssim_map.mean()

def _3d_gaussian_calculator(img, conv3d):
    if False:
        for i in range(10):
            print('nop')
    out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out

def _generate_3d_gaussian_kernel():
    if False:
        return 10
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d

def _ssim_3d(img1, img2, max_value):
    if False:
        while True:
            i = 10
    assert len(img1.shape) == 3 and len(img2.shape) == 3
    "Calculate SSIM (structural similarity) for one channel images.\n    It is called by func:`calculate_ssim`.\n    Args:\n        img1 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.\n        img2 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.\n    Returns:\n        float: ssim result.\n    "
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = _generate_3d_gaussian_kernel().cuda()
    img1 = torch.tensor(img1).float().cuda()
    img2 = torch.tensor(img2).float().cuda()
    mu1 = _3d_gaussian_calculator(img1, kernel)
    mu2 = _3d_gaussian_calculator(img2, kernel)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _3d_gaussian_calculator(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(img2 ** 2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(img1 * img2, kernel) - mu1_mu2
    tmp1 = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    tmp2 = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = tmp1 / tmp2
    return float(ssim_map.mean())