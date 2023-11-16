from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
import importlib_metadata
from packaging.version import parse as parse_version
from dask.utils import Dispatch
sizeof = Dispatch(name='sizeof')
logger = logging.getLogger(__name__)

@sizeof.register(object)
def sizeof_default(o):
    if False:
        return 10
    return sys.getsizeof(o)

@sizeof.register(bytes)
@sizeof.register(bytearray)
def sizeof_bytes(o):
    if False:
        while True:
            i = 10
    return len(o)

@sizeof.register(memoryview)
def sizeof_memoryview(o):
    if False:
        return 10
    return o.nbytes

@sizeof.register(array)
def sizeof_array(o):
    if False:
        while True:
            i = 10
    return o.itemsize * len(o)

@sizeof.register(list)
@sizeof.register(tuple)
@sizeof.register(set)
@sizeof.register(frozenset)
def sizeof_python_collection(seq):
    if False:
        return 10
    num_items = len(seq)
    num_samples = 10
    if num_items > num_samples:
        if isinstance(seq, (set, frozenset)):
            samples = itertools.islice(seq, num_samples)
        else:
            samples = random.sample(seq, num_samples)
        return sys.getsizeof(seq) + int(num_items / num_samples * sum(map(sizeof, samples)))
    else:
        return sys.getsizeof(seq) + sum(map(sizeof, seq))

class SimpleSizeof:
    """Sentinel class to mark a class to be skipped by the dispatcher. This only
    works if this sentinel mixin is first in the mro.

    Examples
    --------
    >>> def _get_gc_overhead():
    ...     class _CustomObject:
    ...         def __sizeof__(self):
    ...             return 0
    ...
    ...     return sys.getsizeof(_CustomObject())

    >>> class TheAnswer(SimpleSizeof):
    ...     def __sizeof__(self):
    ...         # Sizeof always add overhead of an object for GC
    ...         return 42 - _get_gc_overhead()

    >>> sizeof(TheAnswer())
    42

    """

@sizeof.register(SimpleSizeof)
def sizeof_blocked(d):
    if False:
        return 10
    return sys.getsizeof(d)

@sizeof.register(dict)
def sizeof_python_dict(d):
    if False:
        print('Hello World!')
    return sys.getsizeof(d) + sizeof(list(d.keys())) + sizeof(list(d.values())) - 2 * sizeof(list())

@sizeof.register_lazy('cupy')
def register_cupy():
    if False:
        for i in range(10):
            print('nop')
    import cupy

    @sizeof.register(cupy.ndarray)
    def sizeof_cupy_ndarray(x):
        if False:
            i = 10
            return i + 15
        return int(x.nbytes)

@sizeof.register_lazy('numba')
def register_numba():
    if False:
        print('Hello World!')
    import numba.cuda

    @sizeof.register(numba.cuda.cudadrv.devicearray.DeviceNDArray)
    def sizeof_numba_devicendarray(x):
        if False:
            while True:
                i = 10
        return int(x.nbytes)

@sizeof.register_lazy('rmm')
def register_rmm():
    if False:
        i = 10
        return i + 15
    import rmm
    if hasattr(rmm, 'DeviceBuffer'):

        @sizeof.register(rmm.DeviceBuffer)
        def sizeof_rmm_devicebuffer(x):
            if False:
                for i in range(10):
                    print('nop')
            return int(x.nbytes)

@sizeof.register_lazy('numpy')
def register_numpy():
    if False:
        while True:
            i = 10
    import numpy as np

    @sizeof.register(np.ndarray)
    def sizeof_numpy_ndarray(x):
        if False:
            while True:
                i = 10
        if 0 in x.strides:
            xs = x[tuple((slice(None) if s != 0 else slice(1) for s in x.strides))]
            return xs.nbytes
        return int(x.nbytes)

@sizeof.register_lazy('pandas')
def register_pandas():
    if False:
        i = 10
        return i + 15
    import numpy as np
    import pandas as pd
    OBJECT_DTYPES = (object, pd.StringDtype('python'))

    def object_size(*xs):
        if False:
            print('Hello World!')
        if not xs:
            return 0
        ncells = sum((len(x) for x in xs))
        if not ncells:
            return 0
        unique_samples = {}
        for x in xs:
            sample = np.random.choice(x, size=100, replace=True)
            for i in sample.tolist():
                unique_samples[id(i)] = i
        nsamples = 100 * len(xs)
        sample_nbytes = sum((sizeof(i) for i in unique_samples.values()))
        if len(unique_samples) / nsamples > 0.5:
            return int(sample_nbytes * ncells / nsamples)
        else:
            return sample_nbytes

    @sizeof.register(pd.DataFrame)
    def sizeof_pandas_dataframe(df):
        if False:
            return 10
        p = sizeof(df.index) + sizeof(df.columns)
        object_cols = []
        prev_dtype = None
        for col in df._series.values():
            if prev_dtype is None or col.dtype != prev_dtype:
                prev_dtype = col.dtype
                p += 1200
            p += col.memory_usage(index=False, deep=False)
            if col.dtype in OBJECT_DTYPES:
                object_cols.append(col._values)
        p += object_size(*object_cols)
        return max(1200, p)

    @sizeof.register(pd.Series)
    def sizeof_pandas_series(s):
        if False:
            print('Hello World!')
        p = 1200 + sizeof(s.index) + s.memory_usage(index=False, deep=False)
        if s.dtype in OBJECT_DTYPES:
            p += object_size(s._values)
        return p

    @sizeof.register(pd.Index)
    def sizeof_pandas_index(i):
        if False:
            while True:
                i = 10
        p = 400 + i.memory_usage(deep=False)
        if i.dtype in OBJECT_DTYPES:
            p += object_size(i)
        return p

    @sizeof.register(pd.MultiIndex)
    def sizeof_pandas_multiindex(i):
        if False:
            print('Hello World!')
        p = sum((sizeof(lev) for lev in i.levels))
        for c in i.codes:
            p += c.nbytes
        return p

@sizeof.register_lazy('scipy')
def register_spmatrix():
    if False:
        for i in range(10):
            print('nop')
    import scipy
    from scipy import sparse
    if parse_version(scipy.__version__) < parse_version('1.12.0.dev0'):

        @sizeof.register(sparse.dok_matrix)
        def sizeof_spmatrix_dok(s):
            if False:
                while True:
                    i = 10
            return s.__sizeof__()

    @sizeof.register(sparse.spmatrix)
    def sizeof_spmatrix(s):
        if False:
            for i in range(10):
                print('nop')
        return sum((sizeof(v) for v in s.__dict__.values()))

@sizeof.register_lazy('pyarrow')
def register_pyarrow():
    if False:
        print('Hello World!')
    import pyarrow as pa

    def _get_col_size(data):
        if False:
            i = 10
            return i + 15
        p = 0
        if not isinstance(data, pa.ChunkedArray):
            data = data.data
        for chunk in data.iterchunks():
            for buffer in chunk.buffers():
                if buffer:
                    p += buffer.size
        return p

    @sizeof.register(pa.Table)
    def sizeof_pyarrow_table(table):
        if False:
            return 10
        p = sizeof(table.schema.metadata)
        for col in table.itercolumns():
            p += _get_col_size(col)
        return int(p) + 1000

    @sizeof.register(pa.ChunkedArray)
    def sizeof_pyarrow_chunked_array(data):
        if False:
            for i in range(10):
                print('nop')
        return int(_get_col_size(data)) + 1000

def _register_entry_point_plugins():
    if False:
        while True:
            i = 10
    'Register sizeof implementations exposed by the entry_point mechanism.'
    for entry_point in importlib_metadata.entry_points(group='dask.sizeof'):
        registrar = entry_point.load()
        try:
            registrar(sizeof)
        except Exception:
            logger.exception(f'Failed to register sizeof entry point {entry_point.name}')
_register_entry_point_plugins()