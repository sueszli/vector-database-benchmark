from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['IndexMixin', 'check_shape', 'dok_matrix', 'getdtype', 'isdense', 'isintlike', 'isscalarlike', 'isshape', 'isspmatrix_dok', 'itertools', 'spmatrix', 'upcast', 'upcast_scalar']

def __dir__():
    if False:
        print('Hello World!')
    return __all__

def __getattr__(name):
    if False:
        i = 10
        return i + 15
    return _sub_module_deprecation(sub_package='sparse', module='dok', private_modules=['_dok'], all=__all__, attribute=name)