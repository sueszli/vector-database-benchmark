from typing import Dict, Any, Union, Optional
import torch
import numpy as np
from allennlp.data.fields.field import Field
from allennlp.common.util import JsonDict

class TensorField(Field[torch.Tensor]):
    """
    A class representing a tensor, which could have arbitrary dimensions.
    A batch of these tensors are padded to the max dimension length in the batch
    for each dimension.
    """
    __slots__ = ['tensor', 'padding_value']

    def __init__(self, tensor: Union[torch.Tensor, np.ndarray], padding_value: Any=0.0, dtype: Optional[Union[np.dtype, torch.dtype]]=None) -> None:
        if False:
            while True:
                i = 10
        if dtype is not None:
            if isinstance(tensor, np.ndarray):
                tensor = tensor.astype(dtype)
            elif isinstance(tensor, torch.Tensor):
                tensor = tensor.to(dtype)
            else:
                raise ValueError('Did not recognize the type of `tensor`.')
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        self.tensor = tensor.cpu()
        self.padding_value = padding_value

    def get_padding_lengths(self) -> Dict[str, int]:
        if False:
            while True:
                i = 10
        return {'dimension_' + str(i): shape for (i, shape) in enumerate(self.tensor.size())}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        if False:
            return 10
        tensor = self.tensor
        while len(tensor.size()) < len(padding_lengths):
            tensor = tensor.unsqueeze(-1)
        pad = [padding for (i, dimension_size) in reversed(list(enumerate(tensor.size()))) for padding in [0, padding_lengths['dimension_' + str(i)] - dimension_size]]
        return torch.nn.functional.pad(tensor, pad, value=self.padding_value)

    def empty_field(self):
        if False:
            return 10
        return TensorField(torch.tensor([], dtype=self.tensor.dtype), padding_value=self.padding_value)

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return f'TensorField with shape: {self.tensor.size()} and dtype: {self.tensor.dtype}.'

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return 1 if len(self.tensor.size()) <= 0 else self.tensor.size(0)

    def __eq__(self, other) -> bool:
        if False:
            i = 10
            return i + 15
        if isinstance(self, other.__class__):
            return torch.equal(self.tensor, other.tensor) and self.padding_value == other.padding_value
        return NotImplemented

    @property
    def array(self):
        if False:
            print('Hello World!')
        'This is a compatibility method that returns the underlying tensor as a numpy array.'
        return self.tensor.numpy()

    def human_readable_repr(self) -> JsonDict:
        if False:
            print('Hello World!')
        shape = list(self.tensor.shape)
        std = torch.std(self.tensor.float()).item()
        mean = torch.mean(self.tensor.float()).item()
        return {'shape': shape, 'element_std': std, 'element_mean': mean, 'type': str(self.tensor.dtype).replace('torch.', '')}