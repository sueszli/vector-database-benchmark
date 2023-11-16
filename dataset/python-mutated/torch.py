import os
import numpy as np
from sahi.utils.import_utils import is_available
if is_available('torch'):
    import torch
else:
    torch = None

def empty_cuda_cache():
    if False:
        return 10
    if is_torch_cuda_available():
        return torch.cuda.empty_cache()

def to_float_tensor(img):
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range\n    [0, 255] to a torch.FloatTensor of shape (C x H x W).\n    Args:\n        img: np.ndarray\n    Returns:\n        torch.tensor\n    '
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(np.array(img)).float()
    if img.max() > 1:
        img /= 255
    return img

def torch_to_numpy(img):
    if False:
        while True:
            i = 10
    img = img.numpy()
    if img.max() > 1:
        img /= 255
    return img.transpose((1, 2, 0))

def is_torch_cuda_available():
    if False:
        print('Hello World!')
    if is_available('torch'):
        return torch.cuda.is_available()
    else:
        return False

def select_device(device: str):
    if False:
        print('Hello World!')
    '\n    Selects torch device\n\n    Args:\n        device: str\n            "cpu", "mps", "cuda", "cuda:0", "cuda:1", etc.\n\n    Returns:\n        torch.device\n\n    Inspired by https://github.com/ultralytics/yolov5/blob/6371de8879e7ad7ec5283e8b95cc6dd85d6a5e72/utils/torch_utils.py#L107\n    '
    if device == 'cuda':
        device = 'cuda:0'
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')
    cpu = device == 'cpu'
    mps = device == 'mps'
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
    if not cpu and (not mps) and is_torch_cuda_available():
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():
        arg = 'mps'
    else:
        arg = 'cpu'
    return torch.device(arg)