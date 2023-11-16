from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['dvode', 'zvode']

def __dir__():
    if False:
        i = 10
        return i + 15
    return __all__

def __getattr__(name):
    if False:
        i = 10
        return i + 15
    return _sub_module_deprecation(sub_package='integrate', module='vode', private_modules=['_vode'], all=__all__, attribute=name)