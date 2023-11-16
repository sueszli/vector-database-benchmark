"""Manipulation of micro-batches."""
import typing
from typing import Any, Callable, List, Union, cast, Sequence
import torch
from torch import Tensor
import torch.cuda.comm
__all__: List[str] = ['NoChunk', 'Batch', 'check', 'scatter', 'gather']
Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]
Function = Callable[[TensorOrTensors], Union[List[Any], Tensor]]

class NoChunk:
    """
    Wrapper for a Tensor in :meth:`Pipe.forward` indicating that the tensor
    should not be chunked on the batch dimension and instead be replicated
    as-is across all micro-batches. This is useful for tensors which might
    not have any 'batch' semantics for the model.
    """

    def __init__(self, inp: Tensor):
        if False:
            for i in range(10):
                print('nop')
        if not torch.is_tensor(inp):
            raise TypeError(f'NoChunk only supported for tensors, found: {inp}')
        self._tensor = inp

    @property
    def tensor(self):
        if False:
            while True:
                i = 10
        return self._tensor

class Batch:
    """
    An abstraction representing a microbatch in the pipeline.
    """

    def __init__(self, values: Union[List[Any], Tensor]) -> None:
        if False:
            print('Hello World!')
        self._values = values
        self.atomic = torch.is_tensor(values)
        if not self.atomic:
            if not any((torch.is_tensor(value) for value in self._values)):
                raise TypeError(f'No tensors found in batch: {self._values}')

    @property
    def tensor(self) -> Tensor:
        if False:
            i = 10
            return i + 15
        'Retrieves the underlying tensor.'
        if not self.atomic:
            raise AttributeError('not atomic batch')
        return cast(Tensor, self._values)

    @property
    def values(self):
        if False:
            while True:
                i = 10
        'Retrieves the underlying values for the batch'
        return self._values

    def find_tensor_idx(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieves the index of first tensor found.\n        '
        if self.atomic:
            return 0
        for (i, value) in enumerate(self._values):
            if torch.is_tensor(value):
                return i
        raise TypeError('No tensor found!')

    def get_device(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieves the device for this microbatch.\n        '
        if self.atomic:
            return self._values.device
        for value in self._values:
            if torch.is_tensor(value):
                return value.device

    def call(self, function: Function) -> 'Batch':
        if False:
            while True:
                i = 10
        'Calls a function on the microbatch. It also wraps\n        the output with :class:`Batch`.\n        '
        if self.atomic:
            return Batch(function(self._values))
        else:
            return Batch(function(*self._values))

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'Batch[atomic={self.atomic!r}]({self._values!r})'

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.atomic:
            yield self._values
        else:
            yield from self._values

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return 1 if self.atomic else len(self._values)

    def __getitem__(self, index: int):
        if False:
            print('Hello World!')
        if not self.atomic:
            return self._values[index]
        if index != 0:
            raise IndexError('atomic batch allows index 0 only')
        return self._values

    @typing.overload
    def __setitem__(self, index: int, value: Tensor) -> None:
        if False:
            for i in range(10):
                print('nop')
        ...

    @typing.overload
    def __setitem__(self, index: slice, value: Tensors) -> None:
        if False:
            print('Hello World!')
        ...

    def __setitem__(self, index: Union[int, slice], value) -> None:
        if False:
            return 10
        if isinstance(index, int):
            self._setitem_by_index(index, value)
        else:
            self._setitem_by_slice(index, value)

    def _setitem_by_index(self, index: int, value) -> None:
        if False:
            return 10
        if not self.atomic:
            i = index
            self._values = self._values[:i] + (value,) + self._values[i + 1:]
            return
        if index != 0:
            raise IndexError('atomic batch allows index 0 only')
        self._values = value

    def _setitem_by_slice(self, index: slice, value) -> None:
        if False:
            return 10
        if not index.start is index.stop is index.step is None:
            raise NotImplementedError('only slice [:] supported')
        if not self.atomic:
            self._values = value
            return
        if len(value) != 1:
            raise IndexError('atomic batch cannot be replaced with multiple tensors')
        self._values = value[0]

def check(first_device, *inputs) -> None:
    if False:
        print('Hello World!')
    '\n    Checks whether the input contains at least one tensor and each tensor is\n    on the same device as the first partition.\n\n    Raises:\n        ValueError: input does not contain at least one tensor\n\n    '
    if not any((torch.is_tensor(input) for input in inputs)):
        raise TypeError(f'inputs do not have any tensors: {inputs}')
    if any((torch.is_tensor(input) and input.device != first_device for input in inputs)):
        raise ValueError('All inputs should be on the same device as the first partition')

def scatter(*inputs, chunks: int) -> List[Batch]:
    if False:
        for i in range(10):
            print('nop')
    'Splits an input mini-batch into multiple micro-batches.'
    if len(inputs) == 1 and isinstance(inputs[0], Tensor):
        return [Batch(x) for x in inputs[0].chunk(chunks)]
    batches: List[Any] = [[] for _ in range(chunks)]
    num_chunks = -1
    for input in inputs:
        if torch.is_tensor(input):
            tensors = input.chunk(chunks)
            if num_chunks != -1 and num_chunks != len(tensors):
                raise RuntimeError(f'Found different number of chunks produced for inputs: {num_chunks} and {len(tensors)}')
            num_chunks = len(tensors)
            for (i, tensor) in enumerate(tensors):
                batches[i].append(tensor)
        else:
            for i in range(chunks):
                if isinstance(input, NoChunk):
                    batches[i].append(input.tensor)
                else:
                    batches[i].append(input)
    batches = batches[:num_chunks]
    return [Batch(x) for x in batches]

def gather(outputs: List[Batch]):
    if False:
        return 10
    'Concatenates output micro-batches into a mini-batch.'
    output: Any
    if outputs[0].atomic:
        tensors = tuple((b.tensor for b in outputs))
        output = torch.cat(tensors)
    else:
        output_buf: List[Any] = []
        for i in range(len(outputs[0])):
            output_type = type(outputs[0][i])
            current_outputs = []
            for batch in outputs:
                if output_type != type(batch[i]):
                    raise TypeError(f'Types for microbatch outputs do not match, found: {output_type} and {type(batch[i])}')
                current_outputs.append(batch[i])
            if torch.is_tensor(outputs[0][i]):
                output_buf.append(torch.cat(current_outputs))
            else:
                output_buf.append(current_outputs)
        output = tuple(output_buf)
    return output