from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['odeint', 'ODEintWarning']

def __dir__():
    if False:
        i = 10
        return i + 15
    return __all__

def __getattr__(name):
    if False:
        print('Hello World!')
    return _sub_module_deprecation(sub_package='integrate', module='odepack', private_modules=['_odepack_py'], all=__all__, attribute=name)