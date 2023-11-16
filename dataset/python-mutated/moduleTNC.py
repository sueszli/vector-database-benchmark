from scipy._lib.deprecation import _sub_module_deprecation
__all__ = []

def __dir__():
    if False:
        return 10
    return __all__

def __getattr__(name):
    if False:
        return 10
    return _sub_module_deprecation(sub_package='optimize', module='moduleTNC', private_modules=['_moduleTNC'], all=__all__, attribute=name)