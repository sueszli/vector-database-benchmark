import warnings
from . import _harwell_boeing
__all__ = ['MalformedHeader', 'hb_read', 'hb_write', 'HBInfo', 'HBFile', 'HBMatrixType', 'FortranFormatParser', 'IntFormat', 'ExpFormat', 'BadFortranFormat', 'hb']

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
        raise AttributeError(f'scipy.io.harwell_boeing is deprecated and has no attribute {name}. Try looking in scipy.io instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.io` namespace, the `scipy.io.harwell_boeing` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_harwell_boeing, name)