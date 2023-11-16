import warnings
from . import _lti_conversion
__all__ = ['tf2ss', 'abcd_normalize', 'ss2tf', 'zpk2ss', 'ss2zpk', 'cont2discrete', 'eye', 'atleast_2d', 'poly', 'prod', 'array', 'outer', 'linalg', 'tf2zpk', 'zpk2tf', 'normalize']

def __dir__():
    if False:
        while True:
            i = 10
    return __all__

def __getattr__(name):
    if False:
        for i in range(10):
            print('nop')
    if name not in __all__:
        raise AttributeError(f'scipy.signal.lti_conversion is deprecated and has no attribute {name}. Try looking in scipy.signal instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.signal` namespace, the `scipy.signal.lti_conversion` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_lti_conversion, name)