from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['mvnun', 'mvnun_weighted', 'mvndst', 'dkblck']

def __dir__():
    if False:
        for i in range(10):
            print('nop')
    return __all__

def __getattr__(name):
    if False:
        return 10
    return _sub_module_deprecation(sub_package='stats', module='mvn', private_modules=['_mvn'], all=__all__, attribute=name)