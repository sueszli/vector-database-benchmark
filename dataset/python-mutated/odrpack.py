from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['odr', 'OdrWarning', 'OdrError', 'OdrStop', 'Data', 'RealData', 'Model', 'Output', 'ODR', 'odr_error', 'odr_stop']

def __dir__():
    if False:
        i = 10
        return i + 15
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    return _sub_module_deprecation(sub_package='odr', module='odrpack', private_modules=['_odrpack'], all=__all__, attribute=name)