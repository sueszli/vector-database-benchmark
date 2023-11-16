from typing import Dict
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torchvision import models
from modelscope.metainfo import Metrics
from modelscope.models.cv.image_inpainting.modules.inception import InceptionV3
from modelscope.utils.registry import default_group
from modelscope.utils.tensor_utils import torch_nested_detach, torch_nested_numpify
from .base import Metric
from .builder import METRICS, MetricKeys
from .image_denoise_metric import calculate_psnr
from .image_inpainting_metric import FIDScore

@METRICS.register_module(group_key=default_group, module_name=Metrics.image_colorization_metric)
class ImageColorizationMetric(Metric):
    """The metric computation class for image colorization.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.preds = []
        self.targets = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def add(self, outputs: Dict, inputs: Dict):
        if False:
            print('Hello World!')
        ground_truths = outputs['preds']
        eval_results = outputs['targets']
        self.preds.append(eval_results)
        self.targets.append(ground_truths)

    def evaluate(self):
        if False:
            while True:
                i = 10
        psnr_list = []
        cf_list = []
        fid = calculate_fid(self.preds, self.targets, device=self.device)
        for (pred, target) in zip(self.preds, self.targets):
            cf_list.append(calculate_colorfulness(pred))
            psnr_list.append(calculate_psnr(target[0], pred[0], crop_border=0))
        return {MetricKeys.PSNR: np.mean(psnr_list), MetricKeys.FID: fid, MetricKeys.Colorfulness: np.mean(cf_list)}

    def merge(self, other: 'ImageColorizationMetric'):
        if False:
            print('Hello World!')
        self.preds.extend(other.preds)
        self.targets.extend(other.targets)

    def __getstate__(self):
        if False:
            print('Hello World!')
        return (self.preds, self.targets)

    def __setstate__(self, state):
        if False:
            return 10
        self.__init__()
        (self.preds, self.targets) = state

def image_colorfulness(image):
    if False:
        return 10
    image = image * 255.0
    (R, G, B) = (image[0], image[1], image[2])
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt(rbStd ** 2 + ybStd ** 2)
    meanRoot = np.sqrt(rbMean ** 2 + ybMean ** 2)
    return stdRoot + 0.3 * meanRoot

def calculate_colorfulness(pred):
    if False:
        for i in range(10):
            print('nop')
    total_colorfulness = 0
    for img_tensor in pred:
        img_np = img_tensor.cpu().numpy()
        C = image_colorfulness(img_np)
        total_colorfulness += C
    colorfulness = total_colorfulness / len(pred)
    return colorfulness

class INCEPTION_V3_FID(nn.Module):
    """pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {64: 0, 192: 1, 768: 2, 2048: 3}

    def __init__(self, incep_state_dict, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True):
        if False:
            for i in range(10):
                print('nop')
        'Build pretrained InceptionV3\n        Args:\n            output_blocks (list of int):\n                Indices of blocks to return features of. Possible values are:\n                - 0: corresponds to output of first max pooling\n                - 1: corresponds to output of second max pooling\n                - 2: corresponds to output which is fed to aux classifier\n                - 3: corresponds to output of final average pooling\n            resize_input (bool):\n                If true, bilinearly resizes input to width and height 299 before\n                feeding input to model. As the network without fully connected\n                layers is fully convolutional, it should be able to handle inputs\n                of arbitrary size, so resizing might not be strictly needed\n            normalize_input (bool):\n                If true, normalizes the input to the statistics the pretrained\n                Inception network expects\n        '
        super(INCEPTION_V3_FID, self).__init__()
        self.resize_input = resize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        inception = models.inception_v3()
        inception.load_state_dict(incep_state_dict)
        for param in inception.parameters():
            param.requires_grad = False
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))

    def forward(self, inp):
        if False:
            i = 10
            return i + 15
        'Get Inception feature maps\n        Args:\n            inp (torch.tensor):\n                Input tensor of shape Bx3xHxW. Values are expected to be in\n                range (0, 1)\n        Returns:\n            List of torch.tensor corresponding to the selected output\n                block, sorted ascending by index\n        '
        outp = []
        x = inp
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear')
        x = x.clone()
        x = x * 0.5 + 0.5
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        for (idx, block) in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp

def get_activations(images, model, batch_size, verbose=False):
    if False:
        print('Hello World!')
    'Calculates the activations of the pool_3 layer for all images.\n    Args:\n        images: Numpy array of dimension (n_images, 3, hi, wi). The values\n            must lie between 0 and 1.\n        model: Instance of inception model\n        batch_size: the images numpy array is split into batches with\n            batch size batch_size. A reasonable batch size depends\n            on the hardware.\n        verbose: If set to True and parameter out_step is given, the number\n            of calculated batches is reported.\n    Returns:\n        A numpy array of dimension (num images, dims) that contains the\n            activations of the given tensor when feeding inception with the\n            query tensor.\n    '
    model.eval()
    d0 = int(images.size(0))
    if batch_size > d0:
        print('Warning: batch size is bigger than the data size. Setting batch size to data size')
        batch_size = d0
    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, 2048))
    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        start = i * batch_size
        end = start + batch_size
        'batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)\n        batch = Variable(batch, volatile=True)\n        if cfg.CUDA:\n            batch = batch.cuda()'
        batch = images[start:end]
        pred = model(batch)[0]
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)
    if verbose:
        print(' done')
    return pred_arr

def calculate_activation_statistics(act):
    if False:
        return 10
    'Calculation of the statistics used by the FID.\n    Args:\n        act: Numpy array of dimension (n_images, dim (e.g. 2048)).\n    Returns:\n        mu: The mean over samples of the activations of the pool_3 layer of\n            the inception model.\n        sigma: The covariance matrix of the activations of the pool_3 layer of\n            the inception model.\n    '
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return (mu, sigma)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-06):
    if False:
        print('Hello World!')
    "Numpy implementation of the Frechet Distance.\n    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)\n    and X_2 ~ N(mu_2, C_2) is\n            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).\n    Stable version by Dougal J. Sutherland.\n    Args:\n        mu1: Numpy array containing the activations of a layer of the\n            nception net (like returned by the function 'get_predictions')\n            or generated samples.\n        mu2: The sample mean over activations, precalculated on an\n            representive data set.\n        sigma1: The covariance matrix over activations for generated samples.\n        sigma2: The covariance matrix over activations, precalculated on an\n            epresentive data set.\n    Returns:\n        The Frechet Distance.\n    "
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
    diff = mu1 - mu2
    (covmean, _) = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = 'fid calculation produces singular product; adding %s to diagonal of cov estimates' % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=0.001):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_fid(preds, targets, device):
    if False:
        print('Hello World!')
    incep_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
    try:
        from torchvision.models.utils import load_state_dict_from_url
    except ImportError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url
    incep_state_dict = load_state_dict_from_url(incep_url, progress=True)
    block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[2048]
    inception_model_fid = INCEPTION_V3_FID(incep_state_dict, [block_idx])
    inception_model_fid.to(device)
    inception_model_fid.eval()
    (fake_acts_set, acts_set) = ([], [])
    with torch.no_grad():
        for (pred, gt) in zip(preds, targets):
            (pred, gt) = (pred.to(device), gt.to(device))
            fake_act = get_activations(pred, inception_model_fid, pred.shape[0])
            real_act = get_activations(gt, inception_model_fid, gt.shape[0])
            fake_acts_set.append(fake_act)
            acts_set.append(real_act)
        acts_set = np.concatenate(acts_set, 0)
        fake_acts_set = np.concatenate(fake_acts_set, 0)
        (real_mu, real_sigma) = calculate_activation_statistics(acts_set)
        (fake_mu, fake_sigma) = calculate_activation_statistics(fake_acts_set)
        fid_score = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
    return fid_score