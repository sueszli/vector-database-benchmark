from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['quad', 'dblquad', 'tplquad', 'nquad', 'IntegrationWarning', 'error']

def __dir__():
    if False:
        for i in range(10):
            print('nop')
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    return _sub_module_deprecation(sub_package='integrate', module='quadpack', private_modules=['_quadpack_py'], all=__all__, attribute=name)