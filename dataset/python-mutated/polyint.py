from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['BarycentricInterpolator', 'KroghInterpolator', 'approximate_taylor_polynomial', 'barycentric_interpolate', 'factorial', 'float_factorial', 'krogh_interpolate']

def __dir__():
    if False:
        for i in range(10):
            print('nop')
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    return _sub_module_deprecation(sub_package='interpolate', module='polyint', private_modules=['_polyint'], all=__all__, attribute=name)