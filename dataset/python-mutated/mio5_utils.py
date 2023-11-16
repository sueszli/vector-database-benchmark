import warnings
from . import _mio5_utils
__all__ = ['VarHeader5', 'VarReader5', 'byteswap_u4', 'chars_to_strings', 'csc_matrix', 'mio5p', 'pycopy', 'swapped_code', 'squeeze_element']

def __dir__():
    if False:
        i = 10
        return i + 15
    return __all__

def __getattr__(name):
    if False:
        i = 10
        return i + 15
    if name not in __all__:
        raise AttributeError(f'scipy.io.matlab.mio5_utils is deprecated and has no attribute {name}. Try looking in scipy.io.matlab instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.io.matlab` namespace, the `scipy.io.matlab.mio5_utils` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_mio5_utils, name)