from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['SpecialFunctionWarning', 'SpecialFunctionError']

def __dir__():
    if False:
        print('Hello World!')
    return __all__

def __getattr__(name):
    if False:
        for i in range(10):
            print('nop')
    return _sub_module_deprecation(sub_package='special', module='sf_error', private_modules=['_sf_error'], all=__all__, attribute=name)