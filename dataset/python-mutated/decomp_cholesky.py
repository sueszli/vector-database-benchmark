from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['cholesky', 'cho_factor', 'cho_solve', 'cholesky_banded', 'cho_solve_banded', 'asarray_chkfinite', 'atleast_2d', 'LinAlgError', 'get_lapack_funcs']

def __dir__():
    if False:
        for i in range(10):
            print('nop')
    return __all__

def __getattr__(name):
    if False:
        for i in range(10):
            print('nop')
    return _sub_module_deprecation(sub_package='linalg', module='decomp_cholesky', private_modules=['_decomp_cholesky'], all=__all__, attribute=name)