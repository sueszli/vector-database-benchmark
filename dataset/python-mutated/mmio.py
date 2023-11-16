import warnings
from . import _mmio
__all__ = ['mminfo', 'mmread', 'mmwrite', 'MMFile', 'coo_matrix', 'asstr']

def __dir__():
    if False:
        i = 10
        return i + 15
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    if name not in __all__:
        raise AttributeError(f'scipy.io.mmio is deprecated and has no attribute {name}. Try looking in scipy.io instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.io` namespace, the `scipy.io.mmio` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_mmio, name)