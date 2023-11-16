from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['multigammaln', 'loggam']

def __dir__():
    if False:
        return 10
    return __all__

def __getattr__(name):
    if False:
        i = 10
        return i + 15
    return _sub_module_deprecation(sub_package='special', module='spfun_stats', private_modules=['_spfun_stats'], all=__all__, attribute=name)