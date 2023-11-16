from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['lu', 'lu_solve', 'lu_factor', 'asarray_chkfinite', 'LinAlgWarning', 'get_lapack_funcs', 'get_flinalg_funcs']

def __dir__():
    if False:
        while True:
            i = 10
    return __all__

def __getattr__(name):
    if False:
        print('Hello World!')
    return _sub_module_deprecation(sub_package='linalg', module='decomp_lu', private_modules=['_decomp_lu'], all=__all__, attribute=name)