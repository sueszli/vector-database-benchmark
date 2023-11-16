import warnings
from . import _idl
__all__ = ['readsav', 'DTYPE_DICT', 'RECTYPE_DICT', 'STRUCT_DICT', 'Pointer', 'ObjectPointer', 'AttrDict']

def __dir__():
    if False:
        for i in range(10):
            print('nop')
    return __all__

def __getattr__(name):
    if False:
        i = 10
        return i + 15
    if name not in __all__:
        raise AttributeError(f'scipy.io.idl is deprecated and has no attribute {name}. Try looking in scipy.io instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.io` namespace, the `scipy.io.idl` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_idl, name)