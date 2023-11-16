from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['lsoda']

def __dir__():
    if False:
        return 10
    return __all__

def __getattr__(name):
    if False:
        for i in range(10):
            print('nop')
    return _sub_module_deprecation(sub_package='integrate', module='lsoda', private_modules=['_lsoda'], all=__all__, attribute=name)