from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['csr_count_blocks', 'csr_matrix', 'csr_tobsr', 'csr_tocsc', 'get_csr_submatrix', 'isspmatrix_csr', 'spmatrix', 'upcast']

def __dir__():
    if False:
        i = 10
        return i + 15
    return __all__

def __getattr__(name):
    if False:
        for i in range(10):
            print('nop')
    return _sub_module_deprecation(sub_package='sparse', module='csr', private_modules=['_csr'], all=__all__, attribute=name)