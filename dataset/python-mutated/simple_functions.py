from __future__ import annotations
from functools import lru_cache
import hashlib
import inspect
import math
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, TypeVar
    from manimlib.typing import FloatArray
    Scalable = TypeVar('Scalable', float, FloatArray)

def sigmoid(x: float | FloatArray):
    if False:
        return 10
    return 1.0 / (1 + np.exp(-x))

@lru_cache(maxsize=10)
def choose(n: int, k: int) -> int:
    if False:
        while True:
            i = 10
    return math.comb(n, k)

def gen_choose(n: int, r: int) -> int:
    if False:
        print('Hello World!')
    return int(np.prod(range(n, n - r, -1)) / math.factorial(r))

def get_num_args(function: Callable) -> int:
    if False:
        while True:
            i = 10
    return len(get_parameters(function))

def get_parameters(function: Callable) -> list:
    if False:
        for i in range(10):
            print('nop')
    return list(inspect.signature(function).parameters.keys())

def clip(a: float, min_a: float, max_a: float) -> float:
    if False:
        for i in range(10):
            print('nop')
    if a < min_a:
        return min_a
    elif a > max_a:
        return max_a
    return a

def arr_clip(arr: np.ndarray, min_a: float, max_a: float) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    arr[arr < min_a] = min_a
    arr[arr > max_a] = max_a
    return arr

def fdiv(a: Scalable, b: Scalable, zero_over_zero_value: Scalable | None=None) -> Scalable:
    if False:
        for i in range(10):
            print('nop')
    if zero_over_zero_value is not None:
        out = np.full_like(a, zero_over_zero_value)
        where = np.logical_or(a != 0, b != 0)
    else:
        out = None
        where = True
    return np.true_divide(a, b, out=out, where=where)

def binary_search(function: Callable[[float], float], target: float, lower_bound: float, upper_bound: float, tolerance: float=0.0001) -> float | None:
    if False:
        i = 10
        return i + 15
    lh = lower_bound
    rh = upper_bound
    mh = (lh + rh) / 2
    while abs(rh - lh) > tolerance:
        (lx, mx, rx) = [function(h) for h in (lh, mh, rh)]
        if lx == target:
            return lx
        if rx == target:
            return rx
        if lx <= target and rx >= target:
            if mx > target:
                rh = mh
            else:
                lh = mh
        elif lx > target and rx < target:
            (lh, rh) = (rh, lh)
        else:
            return None
        mh = (lh + rh) / 2
    return mh

def hash_string(string: str) -> str:
    if False:
        return 10
    hasher = hashlib.sha256(string.encode())
    return hasher.hexdigest()[:16]