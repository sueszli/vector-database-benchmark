import warnings
from . import _pseudo_diffs
__all__ = ['diff', 'tilbert', 'itilbert', 'hilbert', 'ihilbert', 'cs_diff', 'cc_diff', 'sc_diff', 'ss_diff', 'shift', 'iscomplexobj', 'convolve']

def __dir__():
    if False:
        return 10
    return __all__

def __getattr__(name):
    if False:
        print('Hello World!')
    if name not in __all__:
        raise AttributeError(f'scipy.fftpack.pseudo_diffs is deprecated and has no attribute {name}. Try looking in scipy.fftpack instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.fftpack` namespace, the `scipy.fftpack.pseudo_diffs` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_pseudo_diffs, name)