import warnings
from . import _mio
__all__ = ['mat_reader_factory', 'loadmat', 'savemat', 'whosmat', 'contextmanager', 'docfiller', 'MatFile4Reader', 'MatFile4Writer', 'MatFile5Reader', 'MatFile5Writer']

def __dir__():
    if False:
        i = 10
        return i + 15
    return __all__

def __getattr__(name):
    if False:
        print('Hello World!')
    if name not in __all__:
        raise AttributeError(f'scipy.io.matlab.mio is deprecated and has no attribute {name}. Try looking in scipy.io.matlab instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.io.matlab` namespace, the `scipy.io.matlab.mio` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_mio, name)