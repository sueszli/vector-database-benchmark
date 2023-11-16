from numpy._core import _multiarray_umath
from numpy import ufunc
for item in _multiarray_umath.__dir__():
    attr = getattr(_multiarray_umath, item)
    if isinstance(attr, ufunc):
        globals()[item] = attr
_ARRAY_API = _multiarray_umath._ARRAY_API
_UFUNC_API = _multiarray_umath._UFUNC_API

def __getattr__(attr_name):
    if False:
        while True:
            i = 10
    from numpy._core import _multiarray_umath
    from ._utils import _raise_warning
    ret = getattr(_multiarray_umath, attr_name, None)
    if ret is None:
        raise AttributeError(f"module 'numpy.core._multiarray_umath' has no attribute {attr_name}")
    _raise_warning(attr_name, '_multiarray_umath')
    return ret
del _multiarray_umath, ufunc