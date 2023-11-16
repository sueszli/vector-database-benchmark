from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['check_shape', 'dia_matrix', 'dia_matvec', 'get_sum_dtype', 'getdtype', 'isshape', 'isspmatrix_dia', 'spmatrix', 'upcast_char', 'validateaxis']

def __dir__():
    if False:
        print('Hello World!')
    return __all__

def __getattr__(name):
    if False:
        i = 10
        return i + 15
    return _sub_module_deprecation(sub_package='sparse', module='dia', private_modules=['_dia'], all=__all__, attribute=name)