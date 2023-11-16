import math
from functools import partial, lru_cache
from typing import Optional, Dict, Any
import numpy as np
import torch
from ding.compatibility import torch_ge_180
from ding.torch_utils import one_hot
num_first_one_hot = partial(one_hot, num_first=True)

def sqrt_one_hot(v: torch.Tensor, max_val: int) -> torch.Tensor:
    if False:
        return 10
    "\n    Overview:\n        Sqrt the input value ``v`` and transform it into one-hot.\n    Arguments:\n        - v (:obj:`torch.Tensor`): the value to be processed with `sqrt` and `one-hot`\n        - max_val (:obj:`int`): the input ``v``'s estimated max value, used to calculate one-hot bit number.             ``v`` would be clamped by (0, max_val).\n    Returns:\n        - ret (:obj:`torch.Tensor`): the value processed after `sqrt` and `one-hot`\n    "
    num = int(math.sqrt(max_val)) + 1
    v = v.float()
    v = torch.floor(torch.sqrt(torch.clamp(v, 0, max_val))).long()
    return one_hot(v, num)

def div_one_hot(v: torch.Tensor, max_val: int, ratio: int) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    "\n    Overview:\n        Divide the input value ``v`` by ``ratio`` and transform it into one-hot.\n    Arguments:\n        - v (:obj:`torch.Tensor`): the value to be processed with `divide` and `one-hot`\n        - max_val (:obj:`int`): the input ``v``'s estimated max value, used to calculate one-hot bit number.             ``v`` would be clamped by (0, ``max_val``).\n        - ratio (:obj:`int`): input ``v`` would be divided by ``ratio``\n    Returns:\n        - ret (:obj:`torch.Tensor`): the value processed after `divide` and `one-hot`\n    "
    num = int(max_val / ratio) + 1
    v = v.float()
    v = torch.floor(torch.clamp(v, 0, max_val) / ratio).long()
    return one_hot(v, num)

def div_func(inputs: torch.Tensor, other: float, unsqueeze_dim: int=1):
    if False:
        return 10
    '\n    Overview:\n        Divide ``inputs`` by ``other`` and unsqueeze if needed.\n    Arguments:\n        - inputs (:obj:`torch.Tensor`): the value to be unsqueezed and divided\n        - other (:obj:`float`): input would be divided by ``other``\n        - unsqueeze_dim (:obj:`int`): the dim to implement unsqueeze\n    Returns:\n        - ret (:obj:`torch.Tensor`): the value processed after `unsqueeze` and `divide`\n    '
    inputs = inputs.float()
    if unsqueeze_dim is not None:
        inputs = inputs.unsqueeze(unsqueeze_dim)
    return torch.div(inputs, other)

def clip_one_hot(v: torch.Tensor, num: int) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        Clamp the input ``v`` in (0, num-1) and make one-hot mapping.\n    Arguments:\n        - v (:obj:`torch.Tensor`): the value to be processed with `clamp` and `one-hot`\n        - num (:obj:`int`): number of one-hot bits\n    Returns:\n        - ret (:obj:`torch.Tensor`): the value processed after `clamp` and `one-hot`\n    '
    v = v.clamp(0, num - 1)
    return one_hot(v, num)

def reorder_one_hot(v: torch.LongTensor, dictionary: Dict[int, int], num: int, transform: Optional[np.ndarray]=None) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        Reorder each value in input ``v`` according to reorder dict ``dictionary``, then make one-hot mapping\n    Arguments:\n        - v (:obj:`torch.LongTensor`): the original value to be processed with `reorder` and `one-hot`\n        - dictionary (:obj:`Dict[int, int]`): a reorder lookup dict,             map original value to new reordered index starting from 0\n        - num (:obj:`int`): number of one-hot bits\n        - transform (:obj:`int`): an array to firstly transform the original action to general action\n    Returns:\n        - ret (:obj:`torch.Tensor`): one-hot data indicating reordered index\n    '
    assert len(v.shape) == 1
    assert isinstance(v, torch.Tensor)
    new_v = torch.zeros_like(v)
    for idx in range(v.shape[0]):
        if transform is None:
            val = v[idx].item()
        else:
            val = transform[v[idx].item()]
        new_v[idx] = dictionary[val]
    return one_hot(new_v, num)

def reorder_one_hot_array(v: torch.LongTensor, array: np.ndarray, num: int, transform: Optional[np.ndarray]=None) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        Reorder each value in input ``v`` according to reorder dict ``dictionary``, then make one-hot mapping.\n        The difference between this function and ``reorder_one_hot`` is\n        whether the type of reorder lookup data structure is `np.ndarray` or `dict`.\n    Arguments:\n        - v (:obj:`torch.LongTensor`): the value to be processed with `reorder` and `one-hot`\n        - array (:obj:`np.ndarray`): a reorder lookup array, map original value to new reordered index starting from 0\n        - num (:obj:`int`): number of one-hot bits\n        - transform (:obj:`np.ndarray`): an array to firstly transform the original action to general action\n    Returns:\n        - ret (:obj:`torch.Tensor`): one-hot data indicating reordered index\n    '
    v = v.numpy()
    if transform is None:
        val = array[v]
    else:
        val = array[transform[v]]
    return one_hot(torch.LongTensor(val), num)

def reorder_boolean_vector(v: torch.LongTensor, dictionary: Dict[int, int], num: int, transform: Optional[np.ndarray]=None) -> torch.Tensor:
    if False:
        while True:
            i = 10
    '\n    Overview:\n        Reorder each value in input ``v`` to new index according to reorder dict ``dictionary``,\n        then set corresponding position in return tensor to 1.\n    Arguments:\n        - v (:obj:`torch.LongTensor`): the value to be processed with `reorder`\n        - dictionary (:obj:`Dict[int, int]`): a reorder lookup dict,             map original value to new reordered index starting from 0\n        - num (:obj:`int`): total number of items, should equals to max index + 1\n        - transform (:obj:`np.ndarray`): an array to firstly transform the original action to general action\n    Returns:\n        - ret (:obj:`torch.Tensor`): boolean data containing only 0 and 1,             indicating whether corresponding original value exists in input ``v``\n    '
    ret = torch.zeros(num)
    for item in v:
        try:
            if transform is None:
                val = item.item()
            else:
                val = transform[item.item()]
            idx = dictionary[val]
        except KeyError as e:
            raise KeyError('{}_{}_'.format(num, e))
        ret[idx] = 1
    return ret

@lru_cache(maxsize=32)
def get_to_and(num_bits: int) -> np.ndarray:
    if False:
        print('Hello World!')
    '\n    Overview:\n        Get an np.ndarray with ``num_bits`` elements, each equals to :math:`2^n` (n decreases from num_bits-1 to 0).\n        Used by ``batch_binary_encode`` to make bit-wise `and`.\n    Arguments:\n        - num_bits (:obj:`int`): length of the generating array\n    Returns:\n        - to_and (:obj:`np.ndarray`): an array with ``num_bits`` elements,             each equals to :math:`2^n` (n decreases from num_bits-1 to 0)\n    '
    return 2 ** np.arange(num_bits - 1, -1, -1).reshape([1, num_bits])

def batch_binary_encode(x: torch.Tensor, bit_num: int) -> torch.Tensor:
    if False:
        while True:
            i = 10
    '\n    Overview:\n        Big endian binary encode ``x`` to float tensor\n    Arguments:\n        - x (:obj:`torch.Tensor`): the value to be unsqueezed and divided\n        - bit_num (:obj:`int`): number of bits, should satisfy :math:`2^{bit num} > max(x)`\n    Example:\n        >>> batch_binary_encode(torch.tensor([131,71]), 10)\n        tensor([[0., 0., 1., 0., 0., 0., 0., 0., 1., 1.],\n                [0., 0., 0., 1., 0., 0., 0., 1., 1., 1.]])\n    Returns:\n        - ret (:obj:`torch.Tensor`): the binary encoded tensor, containing only `0` and `1`\n    '
    x = x.numpy()
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    to_and = get_to_and(bit_num)
    return torch.FloatTensor((x & to_and).astype(bool).astype(float).reshape(xshape + [bit_num]))

def compute_denominator(x: torch.Tensor) -> torch.Tensor:
    if False:
        while True:
            i = 10
    '\n    Overview:\n        Compute the denominator used in ``get_postion_vector``.         Divide 1 at the last step, so you can use it as an multiplier.\n    Arguments:\n        - x (:obj:`torch.Tensor`): Input tensor, which is generated from torch.arange(0, d_model).\n    Returns:\n        - ret (:obj:`torch.Tensor`): Denominator result tensor.\n    '
    if torch_ge_180():
        x = torch.div(x, 2, rounding_mode='trunc') * 2
    else:
        x = torch.div(x, 2) * 2
    x = torch.div(x, 64.0)
    x = torch.pow(10000.0, x)
    x = torch.div(1.0, x)
    return x

def get_postion_vector(x: list) -> torch.Tensor:
    if False:
        while True:
            i = 10
    '\n    Overview:\n        Get position embedding used in `Transformer`, even and odd :math:`\x07lpha` are stored in ``POSITION_ARRAY``\n    Arguments:\n        - x (:obj:`list`): original position index, whose length should be 32\n    Returns:\n        - v (:obj:`torch.Tensor`): position embedding tensor in 64 dims\n    '
    POSITION_ARRAY = compute_denominator(torch.arange(0, 64, dtype=torch.float))
    v = torch.zeros(64, dtype=torch.float)
    x = torch.FloatTensor(x)
    v[0::2] = torch.sin(x * POSITION_ARRAY[0::2])
    v[1::2] = torch.cos(x * POSITION_ARRAY[1::2])
    return v

def affine_transform(data: Any, action_clip: Optional[bool]=True, alpha: Optional[float]=None, beta: Optional[float]=None, min_val: Optional[float]=None, max_val: Optional[float]=None) -> Any:
    if False:
        return 10
    '\n    Overview:\n        do affine transform for data in range [-1, 1], :math:`\x07lpha \times data + \x08eta`\n    Arguments:\n        - data (:obj:`Any`): the input data\n        - action_clip (:obj:`bool`): whether to do action clip operation ([-1, 1])\n        - alpha (:obj:`float`): affine transform weight\n        - beta (:obj:`float`): affine transform bias\n        - min_val (:obj:`float`): min value, if `min_val` and `max_val` are indicated, scale input data            to [min_val, max_val]\n        - max_val (:obj:`float`): max value\n    Returns:\n        - transformed_data (:obj:`Any`): affine transformed data\n    '
    if action_clip:
        data = np.clip(data, -1, 1)
    if min_val is not None:
        assert max_val is not None
        alpha = (max_val - min_val) / 2
        beta = (max_val + min_val) / 2
    assert alpha is not None
    beta = beta if beta is not None else 0.0
    return data * alpha + beta

def save_frames_as_gif(frames: list, path: str) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        save frames as gif to a specified path.\n    Arguments:\n        - frames (:obj:`List`): list of frames\n        - path (:obj:`str`): the path to save gif\n    '
    try:
        import imageio
    except ImportError:
        from ditk import logging
        import sys
        logging.warning('Please install imageio first.')
        sys.exit(1)
    imageio.mimsave(path, frames, fps=20)