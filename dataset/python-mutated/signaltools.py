import warnings
from . import _signaltools
__all__ = ['correlate', 'correlation_lags', 'correlate2d', 'convolve', 'convolve2d', 'fftconvolve', 'oaconvolve', 'order_filter', 'medfilt', 'medfilt2d', 'wiener', 'lfilter', 'lfiltic', 'sosfilt', 'deconvolve', 'hilbert', 'hilbert2', 'cmplx_sort', 'unique_roots', 'invres', 'invresz', 'residue', 'residuez', 'resample', 'resample_poly', 'detrend', 'lfilter_zi', 'sosfilt_zi', 'sosfiltfilt', 'choose_conv_method', 'filtfilt', 'decimate', 'vectorstrength', 'timeit', 'cKDTree', 'dlti', 'upfirdn', 'linalg', 'sp_fft', 'lambertw', 'get_window', 'axis_slice', 'axis_reverse', 'odd_ext', 'even_ext', 'const_ext', 'cheby1', 'firwin']

def __dir__():
    if False:
        return 10
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    if name not in __all__:
        raise AttributeError(f'scipy.signal.signaltools is deprecated and has no attribute {name}. Try looking in scipy.signal instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.signal` namespace, the `scipy.signal.signaltools` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_signaltools, name)