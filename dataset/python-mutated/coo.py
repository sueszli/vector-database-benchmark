from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['SparseEfficiencyWarning', 'check_reshape_kwargs', 'check_shape', 'coo_matrix', 'coo_matvec', 'coo_tocsr', 'coo_todense', 'downcast_intp_index', 'getdata', 'getdtype', 'isshape', 'isspmatrix_coo', 'operator', 'spmatrix', 'to_native', 'upcast', 'upcast_char', 'warn']

def __dir__():
    if False:
        print('Hello World!')
    return __all__

def __getattr__(name):
    if False:
        return 10
    return _sub_module_deprecation(sub_package='sparse', module='coo', private_modules=['_coo'], all=__all__, attribute=name)