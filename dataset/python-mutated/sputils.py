from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['asmatrix', 'check_reshape_kwargs', 'check_shape', 'downcast_intp_index', 'get_index_dtype', 'get_sum_dtype', 'getdata', 'getdtype', 'is_pydata_spmatrix', 'isdense', 'isintlike', 'ismatrix', 'isscalarlike', 'issequence', 'isshape', 'matrix', 'operator', 'prod', 'supported_dtypes', 'sys', 'to_native', 'upcast', 'upcast_char', 'upcast_scalar', 'validateaxis']

def __dir__():
    if False:
        print('Hello World!')
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    return _sub_module_deprecation(sub_package='sparse', module='sputils', private_modules=['_sputils'], all=__all__, attribute=name)