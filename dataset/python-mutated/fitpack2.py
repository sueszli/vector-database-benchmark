from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['BivariateSpline', 'InterpolatedUnivariateSpline', 'LSQBivariateSpline', 'LSQSphereBivariateSpline', 'LSQUnivariateSpline', 'RectBivariateSpline', 'RectSphereBivariateSpline', 'SmoothBivariateSpline', 'SmoothSphereBivariateSpline', 'SphereBivariateSpline', 'UnivariateSpline', 'array', 'concatenate', 'dfitpack', 'dfitpack_int', 'diff', 'ones', 'ravel', 'zeros']

def __dir__():
    if False:
        i = 10
        return i + 15
    return __all__

def __getattr__(name):
    if False:
        return 10
    return _sub_module_deprecation(sub_package='interpolate', module='fitpack2', private_modules=['_fitpack2'], all=__all__, attribute=name)