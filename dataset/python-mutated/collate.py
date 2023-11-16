"""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These methods are used to collate samples fetched from dataset into Tensor(s).
These **needs** to be in global scope since Py2 doesn't support serializing
static methods.

`default_collate` and `default_convert` are exposed to users via 'dataloader.py'.
"""
import collections
import contextlib
import re
import torch
from typing import Callable, Dict, Optional, Tuple, Type, Union
np_str_obj_array_pattern = re.compile('[SaUO]')

def default_convert(data):
    if False:
        i = 10
        return i + 15
    "\n    Convert each NumPy array element into a :class:`torch.Tensor`.\n\n    If the input is a `Sequence`, `Collection`, or `Mapping`, it tries to convert each element inside to a :class:`torch.Tensor`.\n    If the input is not an NumPy array, it is left unchanged.\n    This is used as the default function for collation when both `batch_sampler` and `batch_size`\n    are NOT defined in :class:`~torch.utils.data.DataLoader`.\n\n    The general input type to output type mapping is similar to that\n    of :func:`~torch.utils.data.default_collate`. See the description there for more details.\n\n    Args:\n        data: a single data point to be converted\n\n    Examples:\n        >>> # xdoctest: +SKIP\n        >>> # Example with `int`\n        >>> default_convert(0)\n        0\n        >>> # Example with NumPy array\n        >>> default_convert(np.array([0, 1]))\n        tensor([0, 1])\n        >>> # Example with NamedTuple\n        >>> Point = namedtuple('Point', ['x', 'y'])\n        >>> default_convert(Point(0, 0))\n        Point(x=0, y=0)\n        >>> default_convert(Point(np.array(0), np.array(0)))\n        Point(x=tensor(0), y=tensor(0))\n        >>> # Example with List\n        >>> default_convert([np.array([0, 1]), np.array([2, 3])])\n        [tensor([0, 1]), tensor([2, 3])]\n    "
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and (elem_type.__name__ != 'string_'):
        if elem_type.__name__ == 'ndarray' and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: default_convert(data[key]) for key in data})
        except TypeError:
            return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, tuple):
        return [default_convert(d) for d in data]
    elif isinstance(data, collections.abc.Sequence) and (not isinstance(data, (str, bytes))):
        try:
            return elem_type([default_convert(d) for d in data])
        except TypeError:
            return [default_convert(d) for d in data]
    else:
        return data
default_collate_err_msg_format = 'default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}'

def collate(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]]=None):
    if False:
        print('Hello World!')
    "\n    General collate function that handles collection type of element within each batch.\n\n    The function also opens function registry to deal with specific element types. `default_collate_fn_map`\n    provides default collate functions for tensors, numpy arrays, numbers and strings.\n\n    Args:\n        batch: a single batch to be collated\n        collate_fn_map: Optional dictionary mapping from element type to the corresponding collate function.\n            If the element type isn't present in this dictionary,\n            this function will go through each key of the dictionary in the insertion order to\n            invoke the corresponding collate function if the element type is a subclass of the key.\n\n    Examples:\n        >>> def collate_tensor_fn(batch, *, collate_fn_map):\n        >>> # Extend this function to handle batch of tensors\n        ...     return torch.stack(batch, 0)\n        >>> def custom_collate(batch):\n        ...     collate_map = {torch.Tensor: collate_tensor_fn}\n        ...     return collate(batch, collate_fn_map=collate_map)\n        >>> # Extend `default_collate` by in-place modifying `default_collate_fn_map`\n        >>> default_collate_fn_map.update({torch.Tensor: collate_tensor_fn})\n\n    Note:\n        Each collate function requires a positional argument for batch and a keyword argument\n        for the dictionary of collate functions as `collate_fn_map`.\n    "
    elem = batch[0]
    elem_type = type(elem)
    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)
        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](batch, collate_fn_map=collate_fn_map)
    if isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem})
        except TypeError:
            return {key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        return elem_type(*(collate(samples, collate_fn_map=collate_fn_map) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all((len(elem) == elem_size for elem in it)):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))
        if isinstance(elem, tuple):
            return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]
        else:
            try:
                return elem_type([collate(samples, collate_fn_map=collate_fn_map) for samples in transposed])
            except TypeError:
                return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]
    raise TypeError(default_collate_err_msg_format.format(elem_type))

def collate_tensor_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]]=None):
    if False:
        i = 10
        return i + 15
    elem = batch[0]
    out = None
    if elem.is_nested:
        raise RuntimeError('Batches of nested tensors are not currently supported by the default collate_fn; please provide a custom collate_fn to handle them appropriately.')
    if torch.utils.data.get_worker_info() is not None:
        numel = sum((x.numel() for x in batch))
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    return torch.stack(batch, 0, out=out)

def collate_numpy_array_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]]=None):
    if False:
        while True:
            i = 10
    elem = batch[0]
    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
        raise TypeError(default_collate_err_msg_format.format(elem.dtype))
    return collate([torch.as_tensor(b) for b in batch], collate_fn_map=collate_fn_map)

def collate_numpy_scalar_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]]=None):
    if False:
        return 10
    return torch.as_tensor(batch)

def collate_float_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]]=None):
    if False:
        while True:
            i = 10
    return torch.tensor(batch, dtype=torch.float64)

def collate_int_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]]=None):
    if False:
        return 10
    return torch.tensor(batch)

def collate_str_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]]=None):
    if False:
        print('Hello World!')
    return batch
default_collate_fn_map: Dict[Union[Type, Tuple[Type, ...]], Callable] = {torch.Tensor: collate_tensor_fn}
with contextlib.suppress(ImportError):
    import numpy as np
    default_collate_fn_map[np.ndarray] = collate_numpy_array_fn
    default_collate_fn_map[np.bool_, np.number, np.object_] = collate_numpy_scalar_fn
default_collate_fn_map[float] = collate_float_fn
default_collate_fn_map[int] = collate_int_fn
default_collate_fn_map[str] = collate_str_fn
default_collate_fn_map[bytes] = collate_str_fn

def default_collate(batch):
    if False:
        while True:
            i = 10
    "\n    Take in a batch of data and put the elements within the batch into a tensor with an additional outer dimension - batch size.\n\n    The exact output type can be a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a\n    Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.\n    This is used as the default function for collation when\n    `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.\n\n    Here is the general input type (based on the type of the element within the batch) to output type mapping:\n\n        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)\n        * NumPy Arrays -> :class:`torch.Tensor`\n        * `float` -> :class:`torch.Tensor`\n        * `int` -> :class:`torch.Tensor`\n        * `str` -> `str` (unchanged)\n        * `bytes` -> `bytes` (unchanged)\n        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`\n        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]),\n          default_collate([V2_1, V2_2, ...]), ...]`\n        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]),\n          default_collate([V2_1, V2_2, ...]), ...]`\n\n    Args:\n        batch: a single batch to be collated\n\n    Examples:\n        >>> # xdoctest: +SKIP\n        >>> # Example with a batch of `int`s:\n        >>> default_collate([0, 1, 2, 3])\n        tensor([0, 1, 2, 3])\n        >>> # Example with a batch of `str`s:\n        >>> default_collate(['a', 'b', 'c'])\n        ['a', 'b', 'c']\n        >>> # Example with `Map` inside the batch:\n        >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])\n        {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}\n        >>> # Example with `NamedTuple` inside the batch:\n        >>> Point = namedtuple('Point', ['x', 'y'])\n        >>> default_collate([Point(0, 0), Point(1, 1)])\n        Point(x=tensor([0, 1]), y=tensor([0, 1]))\n        >>> # Example with `Tuple` inside the batch:\n        >>> default_collate([(0, 1), (2, 3)])\n        [tensor([0, 2]), tensor([1, 3])]\n        >>> # Example with `List` inside the batch:\n        >>> default_collate([[0, 1], [2, 3]])\n        [tensor([0, 2]), tensor([1, 3])]\n        >>> # Two options to extend `default_collate` to handle specific type\n        >>> # Option 1: Write custom collate function and invoke `default_collate`\n        >>> def custom_collate(batch):\n        ...     elem = batch[0]\n        ...     if isinstance(elem, CustomType):  # Some custom condition\n        ...         return ...\n        ...     else:  # Fall back to `default_collate`\n        ...         return default_collate(batch)\n        >>> # Option 2: In-place modify `default_collate_fn_map`\n        >>> def collate_customtype_fn(batch, *, collate_fn_map=None):\n        ...     return ...\n        >>> default_collate_fn_map.update(CustoType, collate_customtype_fn)\n        >>> default_collate(batch)  # Handle `CustomType` automatically\n    "
    return collate(batch, collate_fn_map=default_collate_fn_map)