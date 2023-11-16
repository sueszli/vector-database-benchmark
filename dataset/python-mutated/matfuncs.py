from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['expm', 'cosm', 'sinm', 'tanm', 'coshm', 'sinhm', 'tanhm', 'logm', 'funm', 'signm', 'sqrtm', 'expm_frechet', 'expm_cond', 'fractional_matrix_power', 'khatri_rao', 'prod', 'logical_not', 'ravel', 'transpose', 'conjugate', 'absolute', 'amax', 'sign', 'isfinite', 'single', 'norm', 'solve', 'inv', 'triu', 'svd', 'schur', 'rsf2csf', 'eps', 'feps']

def __dir__():
    if False:
        i = 10
        return i + 15
    return __all__

def __getattr__(name):
    if False:
        print('Hello World!')
    return _sub_module_deprecation(sub_package='linalg', module='matfuncs', private_modules=['_matfuncs'], all=__all__, attribute=name)