"""Utilities used for collections."""
from abc import ABC
from functools import partial
from typing import Any, Callable, List, Tuple, Union
import numpy as np
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from lightning.fabric.utilities.types import _DEVICE
_BLOCKING_DEVICE_TYPES = ('cpu', 'mps')

def _from_numpy(value: np.ndarray, device: _DEVICE) -> Tensor:
    if False:
        i = 10
        return i + 15
    return torch.from_numpy(value).to(device)
CONVERSION_DTYPES: List[Tuple[Any, Callable[[Any, Any], Tensor]]] = [(bool, partial(torch.tensor, dtype=torch.uint8)), (int, partial(torch.tensor, dtype=torch.int)), (float, partial(torch.tensor, dtype=torch.float)), (np.ndarray, _from_numpy)]

class _TransferableDataType(ABC):
    """A custom type for data that can be moved to a torch device via ``.to(...)``.

    Example:

        >>> isinstance(dict, _TransferableDataType)
        False
        >>> isinstance(torch.rand(2, 3), _TransferableDataType)
        True
        >>> class CustomObject:
        ...     def __init__(self):
        ...         self.x = torch.rand(2, 2)
        ...     def to(self, device):
        ...         self.x = self.x.to(device)
        ...         return self
        >>> isinstance(CustomObject(), _TransferableDataType)
        True

    """

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> Union[bool, Any]:
        if False:
            for i in range(10):
                print('nop')
        if cls is _TransferableDataType:
            to = getattr(subclass, 'to', None)
            return callable(to)
        return NotImplemented

def move_data_to_device(batch: Any, device: _DEVICE) -> Any:
    if False:
        while True:
            i = 10
    'Transfers a collection of data to the given device. Any object that defines a method ``to(device)`` will be\n    moved and all other objects in the collection will be left untouched.\n\n    Args:\n        batch: A tensor or collection of tensors or anything that has a method ``.to(...)``.\n            See :func:`apply_to_collection` for a list of supported collection types.\n        device: The device to which the data should be moved\n\n    Return:\n        the same collection but with all contained tensors residing on the new device.\n\n    See Also:\n        - :meth:`torch.Tensor.to`\n        - :class:`torch.device`\n\n    '
    if isinstance(device, str):
        device = torch.device(device)

    def batch_to(data: Any) -> Any:
        if False:
            while True:
                i = 10
        kwargs = {}
        if isinstance(data, Tensor) and isinstance(device, torch.device) and (device.type not in _BLOCKING_DEVICE_TYPES):
            kwargs['non_blocking'] = True
        data_output = data.to(device, **kwargs)
        if data_output is not None:
            return data_output
        return data
    return apply_to_collection(batch, dtype=_TransferableDataType, function=batch_to)

def convert_to_tensors(data: Any, device: _DEVICE) -> Any:
    if False:
        print('Hello World!')
    for (src_dtype, conversion_func) in CONVERSION_DTYPES:
        data = apply_to_collection(data, src_dtype, conversion_func, device=device)
    return move_data_to_device(data, device)

def convert_tensors_to_scalars(data: Any) -> Any:
    if False:
        print('Hello World!')
    'Recursively walk through a collection and convert single-item tensors to scalar values.\n\n    Raises:\n        ValueError:\n            If tensors inside ``metrics`` contains multiple elements, hence preventing conversion to a scalar.\n\n    '

    def to_item(value: Tensor) -> Union[int, float, bool]:
        if False:
            i = 10
            return i + 15
        if value.numel() != 1:
            raise ValueError(f'The metric `{value}` does not contain a single element, thus it cannot be converted to a scalar.')
        return value.item()
    return apply_to_collection(data, Tensor, to_item)