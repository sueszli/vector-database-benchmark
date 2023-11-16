import warnings
from . import _byteordercodes
__all__ = ['aliases', 'native_code', 'swapped_code', 'sys_is_le', 'to_numpy_code']

def __dir__():
    if False:
        for i in range(10):
            print('nop')
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    if name not in __all__:
        raise AttributeError(f'scipy.io.matlab.byteordercodes is deprecated and has no attribute {name}. Try looking in scipy.io.matlab instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.io.matlab` namespace, the `scipy.io.matlab.byteordercodes` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_byteordercodes, name)