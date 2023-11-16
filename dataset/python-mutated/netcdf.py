import warnings
from . import _netcdf
__all__ = ['netcdf_file', 'netcdf_variable', 'array', 'LITTLE_ENDIAN', 'IS_PYPY', 'ABSENT', 'ZERO', 'NC_BYTE', 'NC_CHAR', 'NC_SHORT', 'NC_INT', 'NC_FLOAT', 'NC_DOUBLE', 'NC_DIMENSION', 'NC_VARIABLE', 'NC_ATTRIBUTE', 'FILL_BYTE', 'FILL_CHAR', 'FILL_SHORT', 'FILL_INT', 'FILL_FLOAT', 'FILL_DOUBLE', 'TYPEMAP', 'FILLMAP', 'REVERSE', 'NetCDFFile', 'NetCDFVariable']

def __dir__():
    if False:
        for i in range(10):
            print('nop')
    return __all__

def __getattr__(name):
    if False:
        for i in range(10):
            print('nop')
    if name not in __all__:
        raise AttributeError(f'scipy.io.netcdf is deprecated and has no attribute {name}. Try looking in scipy.io instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.io` namespace, the `scipy.io.netcdf` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_netcdf, name)