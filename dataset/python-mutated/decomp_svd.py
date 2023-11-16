from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['svd', 'svdvals', 'diagsvd', 'orth', 'subspace_angles', 'null_space', 'LinAlgError', 'get_lapack_funcs']

def __dir__():
    if False:
        while True:
            i = 10
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    return _sub_module_deprecation(sub_package='linalg', module='decomp_svd', private_modules=['_decomp_svd'], all=__all__, attribute=name)