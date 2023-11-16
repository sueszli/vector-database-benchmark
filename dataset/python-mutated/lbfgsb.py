from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['LbfgsInvHessProduct', 'LinearOperator', 'MemoizeJac', 'OptimizeResult', 'array', 'asarray', 'float64', 'fmin_l_bfgs_b', 'old_bound_to_new', 'zeros']

def __dir__():
    if False:
        return 10
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    return _sub_module_deprecation(sub_package='optimize', module='lbfgsb', private_modules=['_lbfgsb_py'], all=__all__, attribute=name)