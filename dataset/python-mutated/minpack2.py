from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['dcsrch', 'dcstep']

def __dir__():
    if False:
        print('Hello World!')
    return __all__

def __getattr__(name):
    if False:
        return 10
    return _sub_module_deprecation(sub_package='optimize', module='minpack2', private_modules=['_minpack2'], all=__all__, attribute=name)