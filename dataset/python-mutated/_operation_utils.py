"""Thie file handles "slice" commonly used in mixed-operation.

The ``slice_type`` we support here, is "slice" or "list of slice".
The reason is that sometimes (e.g., in multi-head attention),
the tensor slice could be from multiple parts. This type is extensible.
We can support arbitrary masks in future if we need them.

To slice a tensor, we need ``multidim_slice``,
which is simply a tuple consists of ``slice_type``.

Usually in python programs, the variable put into slice's start, stop and step
should be integers (or NoneType).
But in our case, it could also be a dict from integer to float,
representing a distribution of integer. When that happens,
we convert a "slice with some weighted values", to a "weighted slice".
To this end, we track the computation with ``MaybeWeighted``,
and replay the computation with each possible value.
Meanwhile, we record their weights.
Note that ``MaybeWeighted`` is also extensible.
We can support more types of objects on slice in future.

The fixed/weighted slice is fed into ``_slice_weight``,
which interprets the slice and apply it on a tensor.
"""
from __future__ import annotations
import operator
from typing import Callable, Iterator, TypeVar, Any, Optional, Tuple, Union, List, Dict, Generic, cast
import numpy as np
import torch
__all__ = ['slice_type', 'multidim_slice', 'scalar_or_scalar_dict', 'int_or_int_dict', 'zeros_like', 'Slicable', 'MaybeWeighted']
T = TypeVar('T')
slice_type = Union[slice, List[slice]]
multidim_slice = Tuple[slice_type, ...]
scalar_or_scalar_dict = Union[T, Dict[T, float]]
int_or_int_dict = scalar_or_scalar_dict[int]
_value_fn_type = Optional[Callable[[int_or_int_dict], int]]

def zeros_like(arr: T) -> T:
    if False:
        while True:
            i = 10
    if isinstance(arr, np.ndarray):
        return np.zeros_like(arr)
    elif isinstance(arr, torch.Tensor):
        return torch.zeros_like(arr)
    else:
        raise TypeError(f'Unsupported type for {arr}: {type(arr)}')

def _eliminate_list_slice(shape: tuple, slice_: multidim_slice) -> multidim_slice:
    if False:
        print('Hello World!')
    result = []
    for i in range(len(slice_)):
        if isinstance(slice_[i], list):
            mask = np.zeros(shape[i], dtype=bool)
            for sl in cast(List[slice], slice_[i]):
                mask[sl] = 1
            result.append(mask)
        else:
            result.append(slice_[i])
    return tuple(result)

def _slice_weight(weight: T, slice_: multidim_slice | list[tuple[multidim_slice, float]]) -> T:
    if False:
        print('Hello World!')
    if isinstance(slice_, list):
        masks = []
        for (sl, wt) in slice_:
            with torch.no_grad():
                mask = zeros_like(weight)
                mask[_eliminate_list_slice(weight.shape, sl)] = 1
            masks.append(mask * wt)
        masks = sum(masks)
        return masks * weight
    else:

        def _do_slice(arr, slice_):
            if False:
                while True:
                    i = 10
            return arr[_eliminate_list_slice(arr.shape, slice_)]
        dummy_arr = np.zeros(weight.shape, dtype=bool)
        no_effect = cast(Any, _do_slice(dummy_arr, slice_)).shape == dummy_arr.shape
        if no_effect:
            return weight
        return _do_slice(weight, slice_)

class Slicable(Generic[T]):
    """Wraps the weight so that in can be sliced with a ``multidim_slice``.
    The value within the slice can be instances of :class:`MaybeWeighted`.

    Examples
    --------
    >>> weight = conv2d.weight
    >>> Slicable(weight)[:MaybeWeighted({32: 0.4, 64: 0.6})]
    Tensor of shape (64, 64, 3, 3)
    """

    def __init__(self, weight: T):
        if False:
            print('Hello World!')
        if not isinstance(weight, np.ndarray) and (not torch.is_tensor(weight)):
            raise TypeError(f'Unsuppoted weight type: {type(weight)}')
        self.weight = weight

    def __getitem__(self, index: slice_type | multidim_slice | Any) -> T:
        if False:
            print('Hello World!')
        if not isinstance(index, tuple):
            index = (index,)
        index = cast(multidim_slice, index)
        leaf_dict: dict[int, float] | None = None
        for maybe_weighted in _iterate_over_multidim_slice(index):
            for d in maybe_weighted.leaf_values():
                if isinstance(d, dict):
                    if leaf_dict is None:
                        leaf_dict = d
                    elif leaf_dict is not d:
                        raise ValueError('There can be at most one distinct dict in leaf values.')
        if leaf_dict is None:
            res_index = _evaluate_multidim_slice(index)
        else:
            res_index = []
            for (val, wt) in leaf_dict.items():
                res_index_item = _evaluate_multidim_slice(index, lambda _: val)
                res_index.append((res_index_item, wt))
        return _slice_weight(self.weight, res_index)

class MaybeWeighted:
    """Wrap a value (int or dict with int keys), so that the computation on it can be replayed.
    It builds a binary tree. If ``value`` is not None, it's a leaf node.
    Otherwise, it has left sub-tree and right sub-tree and an operation.

    Only support basic arithmetic operations: ``+``, ``-``, ``*``, ``//``.
    """

    def __init__(self, value: int_or_int_dict | None=None, *, lhs: 'MaybeWeighted' | int | None=None, rhs: 'MaybeWeighted' | int | None=None, operation: Callable[[int_or_int_dict, int_or_int_dict], int_or_int_dict] | None=None):
        if False:
            return 10
        if operation is None:
            if not isinstance(value, (int, dict)):
                raise TypeError(f'Unsupported value type: {type(value)}')
        self.value = value
        self.lhs = lhs
        self.rhs = rhs
        self.operation = operation

    def leaf_values(self) -> Iterator[int_or_int_dict]:
        if False:
            return 10
        'Iterate over values on leaf nodes.'
        if self.value is not None:
            yield self.value
        else:
            if isinstance(self.lhs, MaybeWeighted):
                yield from self.lhs.leaf_values()
            if isinstance(self.rhs, MaybeWeighted):
                yield from self.rhs.leaf_values()

    def evaluate(self, value_fn: _value_fn_type=None) -> int_or_int_dict:
        if False:
            return 10
        'Evaluate the value on root node, after replacing every value on leaf node with ``value_fn``.\n        If ``value_fn`` is none, no replacement will happen and the raw value will be used.\n        '
        if self.value is not None:
            if value_fn is not None:
                return value_fn(self.value)
            return self.value
        else:
            if isinstance(self.lhs, MaybeWeighted):
                eval_lhs = self.lhs.evaluate(value_fn)
            else:
                eval_lhs = cast(int, self.lhs)
            if isinstance(self.rhs, MaybeWeighted):
                eval_rhs = self.rhs.evaluate(value_fn)
            else:
                eval_rhs = cast(int, self.rhs)
            assert self.operation is not None
            return self.operation(eval_lhs, eval_rhs)

    def __repr__(self):
        if False:
            print('Hello World!')
        if self.value is not None:
            return f'{self.__class__.__name__}({self.value})'
        return f'{self.__class__.__name__}(lhs={self.lhs}, rhs={self.rhs}, op={self.operation})'

    def __add__(self, other: Any) -> 'MaybeWeighted':
        if False:
            for i in range(10):
                print('nop')
        return MaybeWeighted(lhs=self, rhs=other, operation=operator.add)

    def __radd__(self, other: Any) -> 'MaybeWeighted':
        if False:
            i = 10
            return i + 15
        return MaybeWeighted(lhs=other, rhs=self, operation=operator.add)

    def __sub__(self, other: Any) -> 'MaybeWeighted':
        if False:
            print('Hello World!')
        return MaybeWeighted(lhs=self, rhs=other, operation=operator.sub)

    def __rsub__(self, other: Any) -> 'MaybeWeighted':
        if False:
            print('Hello World!')
        return MaybeWeighted(lhs=other, rhs=self, operation=operator.sub)

    def __mul__(self, other: Any) -> 'MaybeWeighted':
        if False:
            while True:
                i = 10
        return MaybeWeighted(lhs=self, rhs=other, operation=operator.mul)

    def __rmul__(self, other: Any) -> 'MaybeWeighted':
        if False:
            print('Hello World!')
        return MaybeWeighted(lhs=other, rhs=self, operation=operator.mul)

    def __floordiv__(self, other: Any) -> 'MaybeWeighted':
        if False:
            print('Hello World!')
        return MaybeWeighted(lhs=self, rhs=other, operation=operator.floordiv)

    def __rfloordiv__(self, other: Any) -> 'MaybeWeighted':
        if False:
            print('Hello World!')
        return MaybeWeighted(lhs=other, rhs=self, operation=operator.floordiv)

def _iterate_over_slice_type(s: slice_type):
    if False:
        i = 10
        return i + 15
    if isinstance(s, list):
        for se in s:
            yield from _iterate_over_slice_type(se)
    else:
        if isinstance(s.start, MaybeWeighted):
            yield s.start
        if isinstance(s.stop, MaybeWeighted):
            yield s.stop
        if isinstance(s.step, MaybeWeighted):
            yield s.step

def _iterate_over_multidim_slice(ms: multidim_slice):
    if False:
        while True:
            i = 10
    'Get :class:`MaybeWeighted` instances in ``ms``.'
    for s in ms:
        if s is not None and s is not Ellipsis:
            yield from _iterate_over_slice_type(s)

def _evaluate_slice_type(s: slice_type, value_fn: _value_fn_type=None):
    if False:
        return 10
    if isinstance(s, list):
        return [_evaluate_slice_type(se, value_fn) for se in s]
    else:
        return slice(s.start.evaluate(value_fn) if isinstance(s.start, MaybeWeighted) else s.start, s.stop.evaluate(value_fn) if isinstance(s.stop, MaybeWeighted) else s.stop, s.step.evaluate(value_fn) if isinstance(s.step, MaybeWeighted) else s.step)

def _evaluate_multidim_slice(ms: multidim_slice, value_fn: _value_fn_type=None):
    if False:
        return 10
    'Wraps :meth:`MaybeWeighted.evaluate` to evaluate the whole ``multidim_slice``.'
    res = []
    for s in ms:
        if s is not None and s is not Ellipsis:
            res.append(_evaluate_slice_type(s, value_fn))
        else:
            res.append(s)
    return tuple(res)