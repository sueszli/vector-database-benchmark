import warnings
from . import _realtransforms
__all__ = ['dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn']

def __dir__():
    if False:
        while True:
            i = 10
    return __all__

def __getattr__(name):
    if False:
        i = 10
        return i + 15
    if name not in __all__:
        raise AttributeError(f'scipy.fftpack.realtransforms is deprecated and has no attribute {name}. Try looking in scipy.fftpack instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.fftpack` namespace, the `scipy.fftpack.realtransforms` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_realtransforms, name)