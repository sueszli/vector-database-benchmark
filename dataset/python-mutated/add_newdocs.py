from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['get', 'add_newdoc', 'docdict']

def __dir__():
    if False:
        i = 10
        return i + 15
    return __all__

def __getattr__(name):
    if False:
        return 10
    return _sub_module_deprecation(sub_package='special', module='add_newdocs', private_modules=['_add_newdocs'], all=__all__, attribute=name)