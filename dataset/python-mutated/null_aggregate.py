from types import ModuleType
from typing import Any, Callable, Tuple, Union
import numpy as np
from ray.data.block import AggType, Block, KeyType, T, U
WrappedAggType = Tuple[AggType, int]

def _wrap_acc(a: AggType, has_data: bool) -> WrappedAggType:
    if False:
        print('Hello World!')
    "\n    Wrap accumulation with a numeric boolean flag indicating whether or not\n    this accumulation contains real data; if it doesn't, we consider it to be\n    empty.\n\n    Args:\n        a: The accumulation value.\n        has_data: Whether the accumulation contains real data.\n\n    Returns:\n        An AggType list with the last element being a numeric boolean flag indicating\n        whether or not this accumulation contains real data. If the input a has length\n        n, the returned AggType has length n + 1.\n    "
    if not isinstance(a, list):
        a = [a]
    return a + [1 if has_data else 0]

def _unwrap_acc(a: WrappedAggType) -> Tuple[AggType, bool]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Unwrap the accumulation, which we assume has been wrapped (via _wrap_acc) with a\n    numeric boolean flag indicating whether or not this accumulation contains real data.\n\n    Args:\n        a: The wrapped accumulation value that we wish to unwrap.\n\n    Returns:\n        A tuple containing the unwrapped accumulation value and a boolean indicating\n        whether the accumulation contains real data.\n    '
    has_data = a[-1] == 1
    a = a[:-1]
    if len(a) == 1:
        a = a[0]
    return (a, has_data)

def _null_wrap_init(init: Callable[[KeyType], AggType]) -> Callable[[KeyType], WrappedAggType]:
    if False:
        while True:
            i = 10
    '\n    Wraps an accumulation initializer with null handling.\n\n    The returned initializer function adds on a has_data field that the accumulator\n    uses to track whether an aggregation is empty.\n\n    Args:\n        init: The core init function to wrap.\n\n    Returns:\n        A new accumulation initializer function that can handle nulls.\n    '

    def _init(k: KeyType) -> AggType:
        if False:
            i = 10
            return i + 15
        a = init(k)
        return _wrap_acc(a, has_data=False)
    return _init

def _null_wrap_merge(ignore_nulls: bool, merge: Callable[[AggType, AggType], AggType]) -> Callable[[WrappedAggType, WrappedAggType], WrappedAggType]:
    if False:
        i = 10
        return i + 15
    '\n    Wrap merge function with null handling.\n\n    The returned merge function expects a1 and a2 to be either None or of the form:\n    a = [acc_data_1, ..., acc_data_2, has_data].\n\n    This merges two accumulations subject to the following null rules:\n    1. If a1 is empty and a2 is empty, return empty accumulation.\n    2. If a1 (a2) is empty and a2 (a1) is None, return None.\n    3. If a1 (a2) is empty and a2 (a1) is non-None, return a2 (a1).\n    4. If a1 (a2) is None, return a2 (a1) if ignoring nulls, None otherwise.\n    5. If a1 and a2 are both non-null, return merge(a1, a2).\n\n    Args:\n        ignore_nulls: Whether nulls should be ignored or cause a None result.\n        merge: The core merge function to wrap.\n\n    Returns:\n        A new merge function that handles nulls.\n    '

    def _merge(a1: WrappedAggType, a2: WrappedAggType) -> WrappedAggType:
        if False:
            print('Hello World!')
        if a1 is None:
            return a2 if ignore_nulls else None
        (unwrapped_a1, a1_has_data) = _unwrap_acc(a1)
        if not a1_has_data:
            return a2
        if a2 is None:
            return a1 if ignore_nulls else None
        (unwrapped_a2, a2_has_data) = _unwrap_acc(a2)
        if not a2_has_data:
            return a1
        a = merge(unwrapped_a1, unwrapped_a2)
        return _wrap_acc(a, has_data=True)
    return _merge

def _null_wrap_accumulate_row(ignore_nulls: bool, on_fn: Callable[[T], T], accum: Callable[[AggType, T], AggType]) -> Callable[[WrappedAggType, T], WrappedAggType]:
    if False:
        i = 10
        return i + 15
    '\n    Wrap accumulator function with null handling.\n\n    The returned accumulate function expects a to be either None or of the form:\n    a = [acc_data_1, ..., acc_data_n, has_data].\n\n    This performs an accumulation subject to the following null rules:\n    1. If r is null and ignore_nulls=False, return None.\n    2. If r is null and ignore_nulls=True, return a.\n    3. If r is non-null and a is None, return None.\n    4. If r is non-null and a is non-None, return accum(a[:-1], r).\n\n    Args:\n        ignore_nulls: Whether nulls should be ignored or cause a None result.\n        on_fn: Function selecting a subset of the row to apply the aggregation.\n        accum: The core accumulator function to wrap.\n\n    Returns:\n        A new accumulator function that handles nulls.\n    '

    def _accum(a: WrappedAggType, r: T) -> WrappedAggType:
        if False:
            print('Hello World!')
        r = on_fn(r)
        if _is_null(r):
            if ignore_nulls:
                return a
            else:
                return None
        elif a is None:
            return None
        else:
            (a, _) = _unwrap_acc(a)
            a = accum(a, r)
            return _wrap_acc(a, has_data=True)
    return _accum

def _null_wrap_accumulate_block(ignore_nulls: bool, accum_block: Callable[[AggType, Block], AggType], null_merge: Callable[[WrappedAggType, WrappedAggType], WrappedAggType]) -> Callable[[WrappedAggType, Block], WrappedAggType]:
    if False:
        i = 10
        return i + 15
    '\n    Wrap vectorized aggregate function with null handling.\n\n    This performs a block accumulation subject to the following null rules:\n    1. If any row is null and ignore_nulls=False, return None.\n    2. If at least one row is not null and ignore_nulls=True, return the block\n       accumulation.\n    3. If all rows are null and ignore_nulls=True, return the base accumulation.\n    4. If all rows non-null, return the block accumulation.\n\n    Args:\n        ignore_nulls: Whether nulls should be ignored or cause a None result.\n        accum_block: The core vectorized aggregate function to wrap.\n        null_merge: A null-handling merge, as returned from _null_wrap_merge().\n\n    Returns:\n        A new vectorized aggregate function that handles nulls.\n    '

    def _accum_block_null(a: WrappedAggType, block: Block) -> WrappedAggType:
        if False:
            for i in range(10):
                print('nop')
        ret = accum_block(block)
        if ret is not None:
            ret = _wrap_acc(ret, has_data=True)
        elif ignore_nulls:
            ret = a
        return null_merge(a, ret)
    return _accum_block_null

def _null_wrap_finalize(finalize: Callable[[AggType], AggType]) -> Callable[[WrappedAggType], U]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Wrap finalizer with null handling.\n\n    If the accumulation is empty or None, the returned finalizer returns None.\n\n    Args:\n        finalize: The core finalizing function to wrap.\n\n    Returns:\n        A new finalizing function that handles nulls.\n    '

    def _finalize(a: AggType) -> U:
        if False:
            while True:
                i = 10
        if a is None:
            return None
        (a, has_data) = _unwrap_acc(a)
        if not has_data:
            return None
        return finalize(a)
    return _finalize
LazyModule = Union[None, bool, ModuleType]
_pandas: LazyModule = None

def _lazy_import_pandas() -> LazyModule:
    if False:
        while True:
            i = 10
    global _pandas
    if _pandas is None:
        try:
            import pandas as _pandas
        except ModuleNotFoundError:
            _pandas = False
    return _pandas

def _is_null(r: Any):
    if False:
        while True:
            i = 10
    pd = _lazy_import_pandas()
    if pd:
        return pd.isnull(r)
    try:
        return np.isnan(r)
    except TypeError:
        return r is None