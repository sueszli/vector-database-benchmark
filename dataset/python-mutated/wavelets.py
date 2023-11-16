import warnings
from . import _wavelets
__all__ = ['daub', 'qmf', 'cascade', 'morlet', 'ricker', 'morlet2', 'cwt', 'eig', 'comb', 'convolve']

def __dir__():
    if False:
        i = 10
        return i + 15
    return __all__

def __getattr__(name):
    if False:
        return 10
    if name not in __all__:
        raise AttributeError(f'scipy.signal.wavelets is deprecated and has no attribute {name}. Try looking in scipy.signal instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.signal` namespace, the `scipy.signal.wavelets` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_wavelets, name)