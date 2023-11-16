import warnings
from . import _mio_utils
__all__ = ['squeeze_element', 'chars_to_strings']

def __dir__():
    if False:
        i = 10
        return i + 15
    return __all__

def __getattr__(name):
    if False:
        for i in range(10):
            print('nop')
    if name not in __all__:
        raise AttributeError(f'scipy.io.matlab.mio_utils is deprecated and has no attribute {name}. Try looking in scipy.io.matlab instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.io.matlab` namespace, the `scipy.io.matlab.mio_utils` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_mio_utils, name)