from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['schur', 'rsf2csf', 'asarray_chkfinite', 'single', 'array', 'norm', 'LinAlgError', 'get_lapack_funcs', 'eigvals', 'eps', 'feps']

def __dir__():
    if False:
        i = 10
        return i + 15
    return __all__

def __getattr__(name):
    if False:
        print('Hello World!')
    return _sub_module_deprecation(sub_package='linalg', module='decomp_schur', private_modules=['_decomp_schur'], all=__all__, attribute=name)