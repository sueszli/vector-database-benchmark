from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['OptimizeResult', 'RLock', 'fmin_cobyla', 'functools', 'izip', 'synchronized']

def __dir__():
    if False:
        return 10
    return __all__

def __getattr__(name):
    if False:
        i = 10
        return i + 15
    return _sub_module_deprecation(sub_package='optimize', module='cobyla', private_modules=['_cobyla_py'], all=__all__, attribute=name)