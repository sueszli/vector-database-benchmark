from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['OptimizeResult', 'append', 'approx_derivative', 'approx_jacobian', 'array', 'atleast_1d', 'concatenate', 'exp', 'finfo', 'fmin_slsqp', 'inf', 'isfinite', 'linalg', 'old_bound_to_new', 'slsqp', 'sqrt', 'vstack', 'zeros']

def __dir__():
    if False:
        for i in range(10):
            print('nop')
    return __all__

def __getattr__(name):
    if False:
        print('Hello World!')
    return _sub_module_deprecation(sub_package='optimize', module='slsqp', private_modules=['_slsqp_py'], all=__all__, attribute=name)