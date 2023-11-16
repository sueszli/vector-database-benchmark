from __future__ import annotations
import dask.array as da
from dask import config
from dask.array.backends import ArrayBackendEntrypoint, register_cupy
from dask.array.core import Array
from dask.array.dispatch import to_cupy_dispatch

def _cupy(strict=True):
    if False:
        print('Hello World!')
    try:
        import cupy
    except ImportError:
        if strict:
            raise ImportError('Please install `cupy` to use `CupyBackendEntrypoint`')
        return None
    return cupy

def _da_with_cupy_meta(attr, *args, meta=None, **kwargs):
    if False:
        i = 10
        return i + 15
    meta = _cupy().empty(()) if meta is None else meta
    with config.set({'array.backend': 'numpy'}):
        return getattr(da, attr)(*args, meta=meta, **kwargs)

class CupyBackendEntrypoint(ArrayBackendEntrypoint):

    def __init__(self):
        if False:
            while True:
                i = 10
        'Register data-directed dispatch functions'
        if _cupy(strict=False):
            register_cupy()

    @classmethod
    def to_backend_dispatch(cls):
        if False:
            print('Hello World!')
        return to_cupy_dispatch

    @classmethod
    def to_backend(cls, data: Array, **kwargs):
        if False:
            i = 10
            return i + 15
        if isinstance(data._meta, _cupy().ndarray):
            return data
        return data.map_blocks(cls.to_backend_dispatch(), **kwargs)

    @property
    def RandomState(self):
        if False:
            while True:
                i = 10
        return _cupy().random.RandomState

    @property
    def default_bit_generator(self):
        if False:
            for i in range(10):
                print('nop')
        return _cupy().random.XORWOW

    @staticmethod
    def ones(*args, **kwargs):
        if False:
            print('Hello World!')
        return _da_with_cupy_meta('ones', *args, **kwargs)

    @staticmethod
    def zeros(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return _da_with_cupy_meta('zeros', *args, **kwargs)

    @staticmethod
    def empty(*args, **kwargs):
        if False:
            while True:
                i = 10
        return _da_with_cupy_meta('empty', *args, **kwargs)

    @staticmethod
    def full(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return _da_with_cupy_meta('full', *args, **kwargs)

    @staticmethod
    def arange(*args, like=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        like = _cupy().empty(()) if like is None else like
        with config.set({'array.backend': 'numpy'}):
            return da.arange(*args, like=like, **kwargs)