from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['LEASTSQ_FAILURE', 'LEASTSQ_SUCCESS', 'LinAlgError', 'OptimizeResult', 'OptimizeWarning', 'asarray', 'atleast_1d', 'check_gradient', 'cholesky', 'curve_fit', 'dot', 'dtype', 'error', 'eye', 'finfo', 'fixed_point', 'fsolve', 'greater', 'inexact', 'inf', 'inv', 'issubdtype', 'least_squares', 'leastsq', 'prepare_bounds', 'prod', 'shape', 'solve_triangular', 'svd', 'take', 'transpose', 'triu', 'zeros']

def __dir__():
    if False:
        for i in range(10):
            print('nop')
    return __all__

def __getattr__(name):
    if False:
        return 10
    return _sub_module_deprecation(sub_package='optimize', module='minpack', private_modules=['_minpack_py'], all=__all__, attribute=name)