from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['INT_TYPES', 'IndexMixin', 'bisect_left', 'check_reshape_kwargs', 'check_shape', 'getdtype', 'isscalarlike', 'isshape', 'isspmatrix_lil', 'lil_matrix', 'spmatrix', 'upcast_scalar']

def __dir__():
    if False:
        for i in range(10):
            print('nop')
    return __all__

def __getattr__(name):
    if False:
        print('Hello World!')
    return _sub_module_deprecation(sub_package='sparse', module='lil', private_modules=['_lil'], all=__all__, attribute=name)