"""Utilities to process the value choice compositions,
in the way that is most convenient to one-shot algorithms."""
from __future__ import annotations
import operator
from typing import Any, TypeVar, List, cast, Mapping, Sequence, Optional, Iterable, overload
import numpy as np
import torch
from nni.mutable import MutableExpression, Categorical
Choice = Any
T = TypeVar('T')
__all__ = ['expression_expectation', 'traverse_all_options', 'weighted_sum', 'evaluate_constant']

def expression_expectation(mutable_expr: MutableExpression[float] | Any, weights: dict[str, list[float]]) -> float:
    if False:
        i = 10
        return i + 15
    'Compute the expectation of a value choice.\n\n    Parameters\n    ----------\n    mutable_expr\n        The value choice to compute expectation.\n    weights\n        The weights of each leaf node.\n\n    Returns\n    -------\n    float\n        The expectation.\n    '
    if not isinstance(mutable_expr, MutableExpression):
        return mutable_expr
    if hasattr(mutable_expr, 'function') and mutable_expr.function == operator.add:
        return sum((expression_expectation(child, weights) for child in mutable_expr.arguments))
    if hasattr(mutable_expr, 'function') and mutable_expr.function == operator.sub:
        return expression_expectation(mutable_expr.arguments[0], weights) - expression_expectation(mutable_expr.arguments[1], weights)
    all_options = traverse_all_options(mutable_expr, weights)
    (options, option_weights) = zip(*all_options)
    return weighted_sum(options, option_weights)

@overload
def traverse_all_options(mutable_expr: MutableExpression[T]) -> list[T]:
    if False:
        while True:
            i = 10
    ...

@overload
def traverse_all_options(mutable_expr: MutableExpression[T], weights: dict[str, Sequence[float]] | dict[str, list[float]] | dict[str, np.ndarray] | dict[str, torch.Tensor]) -> list[tuple[T, float]]:
    if False:
        while True:
            i = 10
    ...

def traverse_all_options(mutable_expr: MutableExpression[T], weights: dict[str, Sequence[float]] | dict[str, list[float]] | dict[str, np.ndarray] | dict[str, torch.Tensor] | None=None) -> list[tuple[T, float]] | list[T]:
    if False:
        while True:
            i = 10
    "Traverse all possible computation outcome of a value choice.\n    If ``weights`` is not None, it will also compute the probability of each possible outcome.\n\n    NOTE: This function is very similar to ``MutableExpression.grid``,\n    but it supports specifying weights for each leaf node.\n\n    Parameters\n    ----------\n    mutable_expr\n        The value choice to traverse.\n    weights\n        If there's a prior on leaf nodes, and we intend to know the (joint) prior on results,\n        weights can be provided. The key is label, value are list of float indicating probability.\n        Normally, they should sum up to 1, but we will not check them in this function.\n\n    Returns\n    -------\n    Results will be sorted and duplicates will be eliminated.\n    If weights is provided, the return value will be a list of tuple, with option and its weight.\n    Otherwise, it will be a list of options.\n    "
    simplified = mutable_expr.simplify()
    for (label, param) in simplified.items():
        if not isinstance(param, Categorical):
            raise TypeError(f'{param!r} is not a categorical distribution')
        if weights is not None:
            if label not in weights:
                raise KeyError(f'{mutable_expr} depends on a weight with key {label}, but not found in {weights}')
            if len(param) != len(weights[label]):
                raise KeyError(f'Expect weights with {label} to be of length {len(param)}, but {len(weights[label])} found')
    result: dict[T, float] = {}
    sample = {}
    for sample_res in mutable_expr.grid(memo=sample):
        probability = 1.0
        if weights is not None:
            for (label, chosen) in sample.items():
                if isinstance(weights[label], dict):
                    probability = probability * weights[label][chosen]
                else:
                    chosen_idx = cast(Categorical, simplified[label]).values.index(chosen)
                    if chosen_idx == -1:
                        raise RuntimeError(f'{chosen} is not a valid value for {label}: {simplified[label]!r}')
                    probability = probability * weights[label][chosen_idx]
        if sample_res in result:
            result[sample_res] = result[sample_res] + cast(float, probability)
        else:
            result[sample_res] = cast(float, probability)
    if weights is None:
        return sorted(result.keys())
    else:
        return sorted(result.items())

def evaluate_constant(expr: Any) -> Any:
    if False:
        print('Hello World!')
    "Evaluate a value choice expression to a constant. Raise ValueError if it's not a constant."
    all_options = traverse_all_options(expr)
    if len(all_options) > 1:
        raise ValueError(f'{expr} is not evaluated to a constant. All possible values are: {all_options}')
    res = all_options[0]
    return res

def weighted_sum(items: Sequence[T], weights: Sequence[float | None]=cast(Sequence[Optional[float]], None)) -> T:
    if False:
        while True:
            i = 10
    'Return a weighted sum of items.\n\n    Items can be list of tensors, numpy arrays, or nested lists / dicts.\n\n    If ``weights`` is None, this is simply an unweighted sum.\n    '
    if weights is None:
        weights = [None] * len(items)
    assert len(items) == len(weights) > 0
    elem = items[0]
    unsupported_msg = 'Unsupported element type in weighted sum: {}. Value is: {}'
    if isinstance(elem, str):
        raise TypeError(unsupported_msg.format(type(elem), elem))
    try:
        if isinstance(elem, (torch.Tensor, np.ndarray, float, int, np.number)):
            if weights[0] is None:
                res = elem
            else:
                res = elem * weights[0]
            for (it, weight) in zip(items[1:], weights[1:]):
                if type(it) != type(elem):
                    raise TypeError(f'Expect type {type(elem)} but found {type(it)}. Can not be summed')
                if weight is None:
                    res = res + it
                else:
                    res = res + it * weight
            return cast(T, res)
        if isinstance(elem, Mapping):
            for item in items:
                if not isinstance(item, Mapping):
                    raise TypeError(f'Expect type {type(elem)} but found {type(item)}')
                if set(item) != set(elem):
                    raise KeyError(f'Expect keys {list(elem)} but found {list(item)}')
            return cast(T, {key: weighted_sum(cast(List[dict], [cast(Mapping, d)[key] for d in items]), weights) for key in elem})
        if isinstance(elem, Sequence):
            for item in items:
                if not isinstance(item, Sequence):
                    raise TypeError(f'Expect type {type(elem)} but found {type(item)}')
                if len(item) != len(elem):
                    raise ValueError(f'Expect length {len(item)} but found {len(elem)}')
            transposed = cast(Iterable[list], zip(*items))
            return cast(T, [weighted_sum(column, weights) for column in transposed])
    except (TypeError, ValueError, RuntimeError, KeyError):
        raise ValueError('Error when summing items. Value format / shape does not match. See full traceback for details.' + ''.join([f'\n  {idx}: {_summarize_elem_format(it)}' for (idx, it) in enumerate(items)]))
    raise TypeError(unsupported_msg)

def _summarize_elem_format(elem: Any) -> Any:
    if False:
        while True:
            i = 10

    class _repr_object:

        def __init__(self, representation):
            if False:
                for i in range(10):
                    print('nop')
            self.representation = representation

        def __repr__(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.representation
    if isinstance(elem, torch.Tensor):
        return _repr_object('torch.Tensor(' + ', '.join(map(str, elem.shape)) + ')')
    if isinstance(elem, np.ndarray):
        return _repr_object('np.array(' + ', '.join(map(str, elem.shape)) + ')')
    if isinstance(elem, Mapping):
        return {key: _summarize_elem_format(value) for (key, value) in elem.items()}
    if isinstance(elem, Sequence):
        return [_summarize_elem_format(value) for value in elem]
    return elem