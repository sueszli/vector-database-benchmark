from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['csc_matrix', 'csc_tocsr', 'expandptr', 'isspmatrix_csc', 'spmatrix', 'upcast']

def __dir__():
    if False:
        for i in range(10):
            print('nop')
    return __all__

def __getattr__(name):
    if False:
        print('Hello World!')
    return _sub_module_deprecation(sub_package='sparse', module='csc', private_modules=['_csc'], all=__all__, attribute=name)