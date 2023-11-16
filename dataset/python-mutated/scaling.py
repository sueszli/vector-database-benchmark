from __future__ import annotations
from functools import reduce
from typing import Callable, List
from typing_extensions import Literal
import torch
from torch import Tensor

class Scaling:
    """
    In the process of generating masks, a large number of operations like pooling or upsampling are involved.
    This class provides tensor-related scaling functions for a given scaling kernel.

    Similar to the concept of convolutional kernel, the scaling kernel also moves over the tensor and does operations.
    The scaling kernel in this class is defined by two parts, kernel size and scaling function (shrink and expand).

    Parameters
    ----------
    kernel_size
        kernel_size is the scale, which determines how large a range in a tensor should shrink to a value,
        or how large a value in a tensor should expand.
        `-1` can be used to indicate that it is a full step in this dimension,
        and the dimension where -1 is located will be reduced or unsqueezed during scaling.

        Example::

            kernel_size = [2, -1]

            # For a given 2D-tensor with size (4, 3),
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9],
             [10, 11, 12]]

            # shrinking it by shrink function, its size becomes (2,) after shrinking:
            [shrink([[1, 2, 3], [4, 5, 6]]), shrink([[7, 8, 9], [10, 11, 12]])]

            # expanding it by expand function with a given expand size,
            # if the expand function is repeating the values, and the expand size is (4, 6, 2):
            [[[1, 1],
              [1, 1],
              [2, 2],
              [2, 2],
              [3, 3],
              [3, 3]],
                ...
              [9, 9]]]
            # note that the original tensor with size (4, 3) will unsqueeze to size (4, 3, 1) at first
            # for the `-1` in kernel_size, then expand size (4, 3, 1) to size (4, 6, 2).
    kernel_padding_mode
        'front' or 'back', default is 'front'.
        If set 'front', for a given tensor when shrinking,
        padding `1` at front of kernel_size until `len(tensor.shape) == len(kernel_size)`;
        for a given expand size when expanding,
        padding `1` at front of kernel_size until `len(expand_size) == len(kernel_size)`.
        If set 'back', for a given tensor when shrinking,
        padding `-1` at back of kernel_size until `len(tensor.shape) == len(kernel_size)`;
        for a given expand size when expanding,
        padding `-1` at back of kernel_size until `len(expand_size) == len(kernel_size)`.

        The default padding value (1 or -1) can be set by passing ``kernel_padding_val``.
    kernel_padding_val
        If ``kernel_padding_val`` is not None, the padding value in kernel padding will be specifed.
    """

    def __init__(self, kernel_size: List[int], kernel_padding_mode: Literal['front', 'back']='front', kernel_padding_val: int | None=None) -> None:
        if False:
            while True:
                i = 10
        self.kernel_size = kernel_size
        err_msg = f"kernel_padding_mode should be one of ['front', 'back'], but get kernel_padding_mode={kernel_padding_mode}."
        assert kernel_padding_mode in ['front', 'back'], err_msg
        self.kernel_padding_mode = kernel_padding_mode
        self.kernel_padding_val = kernel_padding_val if kernel_padding_val else 1 if kernel_padding_mode == 'front' else -1

    def _padding(self, _list: List[int], length: int, padding_value: int=-1, padding_mode: Literal['front', 'back']='back') -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Padding the `_list` to a specific length with `padding_value`.\n\n        Parameters\n        ----------\n        _list\n            The list of int value to be padding.\n        length\n            The length to pad to.\n        padding_value\n            Padding value, should be a int.\n        padding_mode\n            If `padding_mode` is `'front'`, then the padding applied on the front of the size list.\n            If `padding_mode` is `'back'`, then the padding applied on the back of the size list.\n\n        Returns\n        -------\n        List[int]\n            The padded list.\n        "
        assert len(_list) <= length
        padding = [padding_value for _ in range(length - len(_list))]
        if padding_mode == 'front':
            new_list = padding + list(_list)
        elif padding_mode == 'back':
            new_list = list(_list) + padding
        else:
            raise ValueError(f'Unsupported padding mode: {padding_mode}.')
        return new_list

    def _shrink(self, target: Tensor, kernel_size: List[int], reduce_func: Callable[[Tensor], Tensor] | None=None, keepdim: bool=False) -> Tensor:
        if False:
            while True:
                i = 10
        '\n        Main logic about how to shrink target. Subclass could override this function to customize.\n        Sum all values covered by the kernel as a simple implementation.\n        '
        reshape_size = []
        final_size = []
        reduced_dims = []
        for (dim, step) in enumerate(kernel_size):
            if step == -1:
                step = target.shape[dim]
                reduced_dims.insert(0, dim)
            assert target.shape[dim] % step == 0
            reshape_size.append(target.shape[dim] // step)
            final_size.append(target.shape[dim] // step)
            reshape_size.append(step)
        permute_dims = [2 * _ for _ in range(len(kernel_size))] + [2 * _ + 1 for _ in range(len(kernel_size))]
        converted_target = target.reshape(reshape_size).permute(permute_dims).reshape(final_size + [-1])
        result = reduce_func(converted_target) if reduce_func else converted_target.mean(-1)
        if not keepdim:
            result = reduce(lambda t, dim: t.squeeze(dim), [result] + reduced_dims)
        return result

    def _expand(self, target: Tensor, kernel_size: List[int], expand_size: List[int], keepdim: bool=False, full_expand: bool=True) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Main logic about how to expand target to a specific size. Subclass could override this function to customize.\n        Repeat each value to reach the kernel size as a simple implementation.\n        '
        if not keepdim:
            unsqueezed_dims = [dim for (dim, step) in enumerate(kernel_size) if step == -1]
            new_target: Tensor = reduce(lambda t, dim: t.unsqueeze(dim), [target] + unsqueezed_dims)
        else:
            new_target = target
        expand_size = expand_size if full_expand else [1 if a == -1 else b for (a, b) in zip(kernel_size, expand_size)]
        _expand_size = []
        for (a, b) in zip(kernel_size, expand_size):
            if a == -1:
                _expand_size.append(1)
                _expand_size.append(b)
            else:
                assert b % a == 0, f'Can not expand tensor with {target.shape} to {expand_size} with kernel size {kernel_size}.'
                _expand_size.append(b // a)
                _expand_size.append(a)
        new_target: Tensor = reduce(lambda t, dim: t.unsqueeze(dim), [new_target] + [2 * _ + 1 for _ in range(len(expand_size))])
        result = new_target.expand(_expand_size).reshape(expand_size).clone()
        return result

    def shrink(self, target: Tensor, reduce_func: Callable[[Tensor], Tensor] | None=None, keepdim: bool=False) -> Tensor:
        if False:
            i = 10
            return i + 15
        if self.kernel_padding_mode == 'front':
            kernel_size = self._padding(self.kernel_size, len(target.shape), self.kernel_padding_val, 'front')
        elif self.kernel_padding_mode == 'back':
            kernel_size = self._padding(self.kernel_size, len(target.shape), self.kernel_padding_val, 'back')
        else:
            raise ValueError(f'Unsupported kernel padding mode: {self.kernel_padding_mode}.')
        return self._shrink(target, kernel_size, reduce_func, keepdim)

    def expand(self, target: Tensor, expand_size: List[int], keepdim: bool=False, full_expand: bool=True):
        if False:
            for i in range(10):
                print('nop')
        if self.kernel_padding_mode == 'front':
            kernel_size = self._padding(self.kernel_size, len(expand_size), self.kernel_padding_val, 'front')
        elif self.kernel_padding_mode == 'back':
            kernel_size = self._padding(self.kernel_size, len(expand_size), self.kernel_padding_val, 'back')
        else:
            raise ValueError(f'Unsupported kernel padding mode: {self.kernel_padding_mode}.')
        return self._expand(target, kernel_size, expand_size, keepdim, full_expand)

    def validate(self, target: List[int] | Tensor):
        if False:
            return 10
        '\n        Validate the target tensor can be shape-lossless scaling.\n        That means the shape will not change after `shrink` then `expand`.\n        '
        target = target if isinstance(target, Tensor) else torch.rand(target)
        if self.expand(self.shrink(target), list(target.shape)).shape != target.shape:
            raise ValueError(f'The tensor with shape {target.shape}, can not shape-lossless scaling with ' + f'kernel size is {self.kernel_size} and kernel_padding_mode is {self.kernel_padding_mode}.')