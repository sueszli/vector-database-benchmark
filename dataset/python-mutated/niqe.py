import math
import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.special import gamma
from modelscope.hub.file_download import model_file_download
from modelscope.metrics.video_super_resolution_metric.matlab_functions import imresize
from modelscope.metrics.video_super_resolution_metric.metric_util import reorder_image, to_y_channel
downloaded_file_path = model_file_download(model_id='damo/cv_realbasicvsr_video-super-resolution_videolq', file_path='niqe_pris_params.npz')

def estimate_aggd_param(block):
    if False:
        for i in range(10):
            print('nop')
    'Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.\n    Args:\n        block (ndarray): 2D Image block.\n    Returns:\n        tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD\n            distribution (Estimating the parameters in Equation 7 in the paper).\n    '
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))
    left_std = np.sqrt(np.mean(block[block < 0] ** 2))
    right_std = np.sqrt(np.mean(block[block > 0] ** 2))
    gammahat = left_std / right_std
    rhat = np.mean(np.abs(block)) ** 2 / np.mean(block ** 2)
    rhat1 = rhat * (gammahat ** 3 + 1) * (gammahat + 1)
    rhatnorm = rhat1 / (gammahat ** 2 + 1) ** 2
    array_position = np.argmin((r_gam - rhatnorm) ** 2)
    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)

def compute_feature(block):
    if False:
        i = 10
        return i + 15
    'Compute features.\n    Args:\n        block (ndarray): 2D Image block.\n    Returns:\n        list: Features with length of 18.\n    '
    feat = []
    (alpha, beta_l, beta_r) = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = np.roll(block, shifts[i], axis=(0, 1))
        (alpha, beta_l, beta_r) = estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat

def niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size_h=96, block_size_w=96):
    if False:
        while True:
            i = 10
    'Calculate NIQE (Natural Image Quality Evaluator) metric.\n    ``Paper: Making a "Completely Blind" Image Quality Analyzer``\n    This implementation could produce almost the same results as the official\n    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip\n    Note that we do not include block overlap height and width, since they are\n    always 0 in the official implementation.\n    For good performance, it is advisable by the official implementation to\n    divide the distorted image in to the same size patched as used for the\n    construction of multivariate Gaussian model.\n    Args:\n        img (ndarray): Input image whose quality needs to be computed. The\n            image must be a gray or Y (of YCbCr) image with shape (h, w).\n            Range [0, 255] with float type.\n        mu_pris_param (ndarray): Mean of a pre-defined multivariate Gaussian\n            model calculated on the pristine dataset.\n        cov_pris_param (ndarray): Covariance of a pre-defined multivariate\n            Gaussian model calculated on the pristine dataset.\n        gaussian_window (ndarray): A 7x7 Gaussian window used for smoothing the\n            image.\n        block_size_h (int): Height of the blocks in to which image is divided.\n            Default: 96 (the official recommended value).\n        block_size_w (int): Width of the blocks in to which image is divided.\n            Default: 96 (the official recommended value).\n    '
    assert img.ndim == 2, 'Input image must be a gray or Y (of YCbCr) image with shape (h, w).'
    (h, w) = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]
    distparam = []
    for scale in (1, 2):
        mu = convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(np.abs(convolve(np.square(img), gaussian_window, mode='nearest') - np.square(mu)))
        img_nomalized = (img - mu) / (sigma + 1)
        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                block = img_nomalized[idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale, idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale]
                feat.append(compute_feature(block))
        distparam.append(np.array(feat))
        if scale == 1:
            img = imresize(img / 255.0, scale=0.5, antialiasing=True)
            img = img * 255.0
    distparam = np.concatenate(distparam, axis=1)
    mu_distparam = np.nanmean(distparam, axis=0)
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(np.matmul(mu_pris_param - mu_distparam, invcov_param), np.transpose(mu_pris_param - mu_distparam))
    quality = np.sqrt(quality)
    quality = float(np.squeeze(quality))
    return quality

def calculate_niqe(img, crop_border, input_order='HWC', convert_to='y', **kwargs):
    if False:
        i = 10
        return i + 15
    'Calculate NIQE (Natural Image Quality Evaluator) metric.\n    ``Paper: Making a "Completely Blind" Image Quality Analyzer``\n    This implementation could produce almost the same results as the official\n    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip\n    > MATLAB R2021a result for tests/data/baboon.png: 5.72957338 (5.7296)\n    > Our re-implementation result for tests/data/baboon.png: 5.7295763 (5.7296)\n    We use the official params estimated from the pristine dataset.\n    We use the recommended block size (96, 96) without overlaps.\n    Args:\n        img (ndarray): Input image whose quality needs to be computed.\n            The input image must be in range [0, 255] with float/int type.\n            The input_order of image can be \'HW\' or \'HWC\' or \'CHW\'. (BGR order)\n            If the input order is \'HWC\' or \'CHW\', it will be converted to gray\n            or Y (of YCbCr) image according to the ``convert_to`` argument.\n        crop_border (int): Cropped pixels in each edge of an image. These\n            pixels are not involved in the metric calculation.\n        input_order (str): Whether the input order is \'HW\', \'HWC\' or \'CHW\'.\n            Default: \'HWC\'.\n        convert_to (str): Whether converted to \'y\' (of MATLAB YCbCr) or \'gray\'.\n            Default: \'y\'.\n    Returns:\n        float: NIQE result.\n    '
    niqe_pris_params = np.load(downloaded_file_path)
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']
    img = img.astype(np.float32)
    if input_order != 'HW':
        img = reorder_image(img, input_order=input_order)
        if convert_to == 'y':
            img = to_y_channel(img)
        elif convert_to == 'gray':
            img = cv2.cvtColor(img / 255.0, cv2.COLOR_BGR2GRAY) * 255.0
        img = np.squeeze(img)
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]
    img = img.round()
    niqe_result = niqe(img, mu_pris_param, cov_pris_param, gaussian_window)
    return niqe_result