from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['physical_constants', 'value', 'unit', 'precision', 'find', 'ConstantWarning', 'txt2002', 'txt2006', 'txt2010', 'txt2014', 'txt2018', 'parse_constants_2002to2014', 'parse_constants_2018toXXXX', 'k', 'c', 'mu0', 'epsilon0', 'exact_values', 'key', 'val', 'v']

def __dir__():
    if False:
        return 10
    return __all__

def __getattr__(name):
    if False:
        return 10
    return _sub_module_deprecation(sub_package='constants', module='codata', private_modules=['_codata'], all=__all__, attribute=name)