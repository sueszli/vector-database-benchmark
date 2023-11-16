import warnings
from . import _fir_filter_design
__all__ = ['kaiser_beta', 'kaiser_atten', 'kaiserord', 'firwin', 'firwin2', 'remez', 'firls', 'minimum_phase', 'ceil', 'log', 'irfft', 'fft', 'ifft', 'sinc', 'toeplitz', 'hankel', 'solve', 'LinAlgError', 'LinAlgWarning', 'lstsq']

def __dir__():
    if False:
        return 10
    return __all__

def __getattr__(name):
    if False:
        return 10
    if name not in __all__:
        raise AttributeError(f'scipy.signal.fir_filter_design is deprecated and has no attribute {name}. Try looking in scipy.signal instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.signal` namespace, the `scipy.signal.fir_filter_design` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_fir_filter_design, name)