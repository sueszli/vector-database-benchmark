from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['bsr_matmat', 'bsr_matrix', 'bsr_matvec', 'bsr_matvecs', 'bsr_sort_indices', 'bsr_tocsr', 'bsr_transpose', 'check_shape', 'csr_matmat_maxnnz', 'getdata', 'getdtype', 'isshape', 'isspmatrix_bsr', 'spmatrix', 'to_native', 'upcast', 'warn']

def __dir__():
    if False:
        i = 10
        return i + 15
    return __all__

def __getattr__(name):
    if False:
        for i in range(10):
            print('nop')
    return _sub_module_deprecation(sub_package='sparse', module='bsr', private_modules=['_bsr'], all=__all__, attribute=name)