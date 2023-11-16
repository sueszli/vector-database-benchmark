import warnings
from . import _flinalg_py
__all__ = ['get_flinalg_funcs', 'has_column_major_storage']

def __dir__():
    if False:
        i = 10
        return i + 15
    return __all__

def __getattr__(name):
    if False:
        return 10
    if name not in __all__:
        raise AttributeError(f'scipy.linalg.flinalg is deprecated and has no attribute {name}.')
    warnings.warn('The `scipy.linalg.flinalg` namespace is deprecated and will be removed in SciPy v2.0.0.', category=DeprecationWarning, stacklevel=2)
    return getattr(_flinalg_py, name)