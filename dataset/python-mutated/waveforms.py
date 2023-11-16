import warnings
from . import _waveforms
__all__ = ['sawtooth', 'square', 'gausspulse', 'chirp', 'sweep_poly', 'unit_impulse', 'place', 'nan', 'mod', 'extract', 'log', 'exp', 'polyval', 'polyint']

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
        raise AttributeError(f'scipy.signal.waveforms is deprecated and has no attribute {name}. Try looking in scipy.signal instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.signal` namespace, the `scipy.signal.waveforms` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_waveforms, name)