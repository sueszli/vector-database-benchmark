from typing import Dict
import cv2
import numpy as np
from modelscope.metainfo import Metrics
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS, MetricKeys

def bgr2ycbcr(img, y_only=False):
    if False:
        i = 10
        return i + 15
    'Convert a BGR image to YCbCr image.\n\n    The bgr version of rgb2ycbcr.\n    It implements the ITU-R BT.601 conversion for standard-definition\n    television. See more details in\n    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.\n\n    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.\n    In OpenCV, it implements a JPEG conversion. See more details in\n    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.\n\n    Args:\n        img (ndarray): The input image. It accepts:\n            1. np.uint8 type with range [0, 255];\n            2. np.float32 type with range [0, 1].\n        y_only (bool): Whether to only return Y channel. Default: False.\n\n    Returns:\n        ndarray: The converted YCbCr image. The output image has the same type\n            and range as input image.\n    '
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

def reorder_image(img, input_order='HWC'):
    if False:
        return 10
    "Reorder images to 'HWC' order.\n\n    If the input_order is (h, w), return (h, w, 1);\n    If the input_order is (c, h, w), return (h, w, c);\n    If the input_order is (h, w, c), return as it is.\n\n    Args:\n        img (ndarray): Input image.\n        input_order (str): Whether the input order is 'HWC' or 'CHW'.\n            If the input image shape is (h, w), input_order will not have\n            effects. Default: 'HWC'.\n\n    Returns:\n        ndarray: reordered image.\n    "
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def to_y_channel(img):
    if False:
        i = 10
        return i + 15
    'Change to Y channel of YCbCr.\n\n    Args:\n        img (ndarray): Images with range [0, 255].\n\n    Returns:\n        (ndarray): Images with range [0, 255] (float type) without round.\n    '
    img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.0

def _ssim(img, img2):
    if False:
        while True:
            i = 10
    "Calculate SSIM (structural similarity) for one channel images.\n\n    It is called by func:`calculate_ssim`.\n\n    Args:\n        img (ndarray): Images with range [0, 255] with order 'HWC'.\n        img2 (ndarray): Images with range [0, 255] with order 'HWC'.\n\n    Returns:\n        float: SSIM result.\n    "
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
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

def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    if False:
        return 10
    "Calculate PSNR (Peak Signal-to-Noise Ratio).\n\n    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio\n\n    Args:\n        img (ndarray): Images with range [0, 255].\n        img2 (ndarray): Images with range [0, 255].\n        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.\n        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.\n        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.\n\n    Returns:\n        float: PSNR result.\n    "
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 * 255.0 / mse)

def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    if False:
        i = 10
        return i + 15
    "Calculate SSIM (structural similarity).\n\n    Ref:\n    Image quality assessment: From error visibility to structural similarity\n\n    The results are the same as that of the official released MATLAB code in\n    https://ece.uwaterloo.ca/~z70wang/research/ssim/.\n\n    For three-channel images, SSIM is calculated for each channel and then\n    averaged.\n\n    Args:\n        img (ndarray): Images with range [0, 255].\n        img2 (ndarray): Images with range [0, 255].\n        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.\n        input_order (str): Whether the input order is 'HWC' or 'CHW'.\n            Default: 'HWC'.\n        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.\n\n    Returns:\n        float: SSIM result.\n    "
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()

@METRICS.register_module(group_key=default_group, module_name=Metrics.image_color_enhance_metric)
class ImageColorEnhanceMetric(Metric):
    """The metric computation class for image color enhance classes.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.preds = []
        self.targets = []

    def add(self, outputs: Dict, inputs: Dict):
        if False:
            while True:
                i = 10
        ground_truths = outputs['target']
        eval_results = outputs['pred']
        self.preds.extend(eval_results)
        self.targets.extend(ground_truths)

    def evaluate(self):
        if False:
            i = 10
            return i + 15
        psnrs = [calculate_psnr(pred, target, 2, test_y_channel=False) for (pred, target) in zip(self.preds, self.targets)]
        ssims = [calculate_ssim(pred, target, 2, test_y_channel=False) for (pred, target) in zip(self.preds, self.targets)]
        return {MetricKeys.PSNR: sum(psnrs) / len(psnrs), MetricKeys.SSIM: sum(ssims) / len(ssims)}

    def merge(self, other: 'ImageColorEnhanceMetric'):
        if False:
            i = 10
            return i + 15
        self.preds.extend(other.preds)
        self.targets.extend(other.targets)

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        return (self.preds, self.targets)

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        (self.preds, self.targets) = state