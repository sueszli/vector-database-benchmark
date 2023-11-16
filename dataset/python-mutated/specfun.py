from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['airyzo', 'bernob', 'cerzo', 'clpmn', 'clpn', 'clqmn', 'clqn', 'cpbdn', 'cyzo', 'eulerb', 'fcoef', 'fcszo', 'jdzo', 'jyzo', 'klvnzo', 'lamn', 'lamv', 'lpmn', 'lpn', 'lqmn', 'lqnb', 'pbdv', 'rctj', 'rcty', 'segv']

def __dir__():
    if False:
        while True:
            i = 10
    return __all__

def __getattr__(name):
    if False:
        return 10
    return _sub_module_deprecation(sub_package='special', module='specfun', private_modules=['_specfun'], all=__all__, attribute=name)