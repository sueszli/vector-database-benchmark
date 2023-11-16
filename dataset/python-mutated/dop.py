from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['dopri5', 'dop853']

def __dir__():
    if False:
        print('Hello World!')
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    return _sub_module_deprecation(sub_package='integrate', module='dop', private_modules=['_dop'], all=__all__, attribute=name)