"""This module converts objects into numpy array."""
import numpy as np
import torch

def make_np(x):
    if False:
        return 10
    '\n    Convert an object into numpy array.\n\n    Args:\n      x: An instance of torch tensor or caffe blob name\n\n    Returns:\n        numpy.array: Numpy array\n    '
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, str):
        return _prepare_caffe2(x)
    if np.isscalar(x):
        return np.array([x])
    if isinstance(x, torch.Tensor):
        return _prepare_pytorch(x)
    raise NotImplementedError(f'Got {type(x)}, but numpy array, torch tensor, or caffe2 blob name are expected.')

def _prepare_pytorch(x):
    if False:
        print('Hello World!')
    x = x.detach().cpu().numpy()
    return x

def _prepare_caffe2(x):
    if False:
        for i in range(10):
            print('nop')
    from caffe2.python import workspace
    x = workspace.FetchBlob(x)
    return x