import warnings
from . import _miobase
__all__ = ['MatFileReader', 'MatReadError', 'MatReadWarning', 'MatVarReader', 'MatWriteError', 'arr_dtype_number', 'arr_to_chars', 'convert_dtypes', 'doc_dict', 'docfiller', 'get_matfile_version', 'matdims', 'read_dtype', 'doccer', 'boc']

def __dir__():
    if False:
        while True:
            i = 10
    return __all__

def __getattr__(name):
    if False:
        return 10
    if name not in __all__:
        raise AttributeError(f'scipy.io.matlab.miobase is deprecated and has no attribute {name}. Try looking in scipy.io.matlab instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.io.matlab` namespace, the `scipy.io.matlab.miobase` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_miobase, name)