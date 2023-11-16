"""
This module contains tensor creation utilities.
"""
import collections.abc
import math
import warnings
from typing import cast, List, Optional, Tuple, Union
import torch
_INTEGRAL_TYPES = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
_FLOATING_TYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
_FLOATING_8BIT_TYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
_COMPLEX_TYPES = [torch.complex32, torch.complex64, torch.complex128]
_BOOLEAN_OR_INTEGRAL_TYPES = [torch.bool, *_INTEGRAL_TYPES]
_FLOATING_OR_COMPLEX_TYPES = [*_FLOATING_TYPES, *_COMPLEX_TYPES]

def _uniform_random_(t: torch.Tensor, low: float, high: float) -> torch.Tensor:
    if False:
        return 10
    if high - low >= torch.finfo(t.dtype).max:
        return t.uniform_(low / 2, high / 2).mul_(2)
    else:
        return t.uniform_(low, high)

def make_tensor(*shape: Union[int, torch.Size, List[int], Tuple[int, ...]], dtype: torch.dtype, device: Union[str, torch.device], low: Optional[float]=None, high: Optional[float]=None, requires_grad: bool=False, noncontiguous: bool=False, exclude_zero: bool=False, memory_format: Optional[torch.memory_format]=None) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    'Creates a tensor with the given :attr:`shape`, :attr:`device`, and :attr:`dtype`, and filled with\n    values uniformly drawn from ``[low, high)``.\n\n    If :attr:`low` or :attr:`high` are specified and are outside the range of the :attr:`dtype`\'s representable\n    finite values then they are clamped to the lowest or highest representable finite value, respectively.\n    If ``None``, then the following table describes the default values for :attr:`low` and :attr:`high`,\n    which depend on :attr:`dtype`.\n\n    +---------------------------+------------+----------+\n    | ``dtype``                 | ``low``    | ``high`` |\n    +===========================+============+==========+\n    | boolean type              | ``0``      | ``2``    |\n    +---------------------------+------------+----------+\n    | unsigned integral type    | ``0``      | ``10``   |\n    +---------------------------+------------+----------+\n    | signed integral types     | ``-9``     | ``10``   |\n    +---------------------------+------------+----------+\n    | floating types            | ``-9``     | ``9``    |\n    +---------------------------+------------+----------+\n    | complex types             | ``-9``     | ``9``    |\n    +---------------------------+------------+----------+\n\n    Args:\n        shape (Tuple[int, ...]): Single integer or a sequence of integers defining the shape of the output tensor.\n        dtype (:class:`torch.dtype`): The data type of the returned tensor.\n        device (Union[str, torch.device]): The device of the returned tensor.\n        low (Optional[Number]): Sets the lower limit (inclusive) of the given range. If a number is provided it is\n            clamped to the least representable finite value of the given dtype. When ``None`` (default),\n            this value is determined based on the :attr:`dtype` (see the table above). Default: ``None``.\n        high (Optional[Number]): Sets the upper limit (exclusive) of the given range. If a number is provided it is\n            clamped to the greatest representable finite value of the given dtype. When ``None`` (default) this value\n            is determined based on the :attr:`dtype` (see the table above). Default: ``None``.\n\n            .. deprecated:: 2.1\n\n                Passing ``low==high`` to :func:`~torch.testing.make_tensor` for floating or complex types is deprecated\n                since 2.1 and will be removed in 2.3. Use :func:`torch.full` instead.\n\n        requires_grad (Optional[bool]): If autograd should record operations on the returned tensor. Default: ``False``.\n        noncontiguous (Optional[bool]): If `True`, the returned tensor will be noncontiguous. This argument is\n            ignored if the constructed tensor has fewer than two elements. Mutually exclusive with ``memory_format``.\n        exclude_zero (Optional[bool]): If ``True`` then zeros are replaced with the dtype\'s small positive value\n            depending on the :attr:`dtype`. For bool and integer types zero is replaced with one. For floating\n            point types it is replaced with the dtype\'s smallest positive normal number (the "tiny" value of the\n            :attr:`dtype`\'s :func:`~torch.finfo` object), and for complex types it is replaced with a complex number\n            whose real and imaginary parts are both the smallest positive normal number representable by the complex\n            type. Default ``False``.\n        memory_format (Optional[torch.memory_format]): The memory format of the returned tensor. Mutually exclusive\n            with ``noncontiguous``.\n\n    Raises:\n        ValueError: If ``requires_grad=True`` is passed for integral `dtype`\n        ValueError: If ``low >= high``.\n        ValueError: If either :attr:`low` or :attr:`high` is ``nan``.\n        ValueError: If both :attr:`noncontiguous` and :attr:`memory_format` are passed.\n        TypeError: If :attr:`dtype` isn\'t supported by this function.\n\n    Examples:\n        >>> # xdoctest: +SKIP\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)\n        >>> from torch.testing import make_tensor\n        >>> # Creates a float tensor with values in [-1, 1)\n        >>> make_tensor((3,), device=\'cpu\', dtype=torch.float32, low=-1, high=1)\n        >>> # xdoctest: +SKIP\n        tensor([ 0.1205, 0.2282, -0.6380])\n        >>> # Creates a bool tensor on CUDA\n        >>> make_tensor((2, 2), device=\'cuda\', dtype=torch.bool)\n        tensor([[False, False],\n                [False, True]], device=\'cuda:0\')\n    '

    def modify_low_high(low: Optional[float], high: Optional[float], *, lowest_inclusive: float, highest_exclusive: float, default_low: float, default_high: float) -> Tuple[float, float]:
        if False:
            i = 10
            return i + 15
        '\n        Modifies (and raises ValueError when appropriate) low and high values given by the user (input_low, input_high)\n        if required.\n        '

        def clamp(a: float, l: float, h: float) -> float:
            if False:
                print('Hello World!')
            return min(max(a, l), h)
        low = low if low is not None else default_low
        high = high if high is not None else default_high
        if any((isinstance(value, float) and math.isnan(value) for value in [low, high])):
            raise ValueError(f'`low` and `high` cannot be NaN, but got low={low!r} and high={high!r}')
        elif low == high and dtype in _FLOATING_OR_COMPLEX_TYPES:
            warnings.warn('Passing `low==high` to `torch.testing.make_tensor` for floating or complex types is deprecated since 2.1 and will be removed in 2.3. Use torch.full(...) instead.', FutureWarning)
        elif low >= high:
            raise ValueError(f'`low` must be less than `high`, but got {low} >= {high}')
        elif high < lowest_inclusive or low >= highest_exclusive:
            raise ValueError(f'The value interval specified by `low` and `high` is [{low}, {high}), but {dtype} only supports [{lowest_inclusive}, {highest_exclusive})')
        low = clamp(low, lowest_inclusive, highest_exclusive)
        high = clamp(high, lowest_inclusive, highest_exclusive)
        if dtype in _BOOLEAN_OR_INTEGRAL_TYPES:
            return (math.ceil(low), math.ceil(high))
        return (low, high)
    if len(shape) == 1 and isinstance(shape[0], collections.abc.Sequence):
        shape = shape[0]
    shape = cast(Tuple[int, ...], tuple(shape))
    if noncontiguous and memory_format is not None:
        raise ValueError(f'The parameters `noncontiguous` and `memory_format` are mutually exclusive, but got noncontiguous={noncontiguous!r} and memory_format={memory_format!r}')
    if requires_grad and dtype in _BOOLEAN_OR_INTEGRAL_TYPES:
        raise ValueError(f'`requires_grad=True` is not supported for boolean and integral dtypes, but got dtype={dtype!r}')
    if dtype is torch.bool:
        (low, high) = cast(Tuple[int, int], modify_low_high(low, high, lowest_inclusive=0, highest_exclusive=2, default_low=0, default_high=2))
        result = torch.randint(low, high, shape, device=device, dtype=dtype)
    elif dtype in _BOOLEAN_OR_INTEGRAL_TYPES:
        (low, high) = cast(Tuple[int, int], modify_low_high(low, high, lowest_inclusive=torch.iinfo(dtype).min, highest_exclusive=torch.iinfo(dtype).max + (1 if dtype is not torch.int64 else 0), default_low=-9, default_high=10))
        result = torch.randint(low, high, shape, device=device, dtype=dtype)
    elif dtype in _FLOATING_OR_COMPLEX_TYPES:
        (low, high) = modify_low_high(low, high, lowest_inclusive=torch.finfo(dtype).min, highest_exclusive=torch.finfo(dtype).max, default_low=-9, default_high=9)
        result = torch.empty(shape, device=device, dtype=dtype)
        _uniform_random_(torch.view_as_real(result) if dtype in _COMPLEX_TYPES else result, low, high)
    elif dtype in _FLOATING_8BIT_TYPES:
        (low, high) = modify_low_high(low, high, lowest_inclusive=torch.finfo(dtype).min, highest_exclusive=torch.finfo(dtype).max, default_low=-9, default_high=9)
        result = torch.empty(shape, device=device, dtype=torch.float32)
        _uniform_random_(result, low, high)
        result = result.to(dtype)
    else:
        raise TypeError(f"The requested dtype '{dtype}' is not supported by torch.testing.make_tensor(). To request support, file an issue at: https://github.com/pytorch/pytorch/issues")
    if noncontiguous and result.numel() > 1:
        result = torch.repeat_interleave(result, 2, dim=-1)
        result = result[..., ::2]
    elif memory_format is not None:
        result = result.clone(memory_format=memory_format)
    if exclude_zero:
        result[result == 0] = 1 if dtype in _BOOLEAN_OR_INTEGRAL_TYPES else torch.finfo(dtype).tiny
    if dtype in _FLOATING_OR_COMPLEX_TYPES:
        result.requires_grad = requires_grad
    return result