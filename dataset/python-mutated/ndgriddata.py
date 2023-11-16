from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['CloughTocher2DInterpolator', 'LinearNDInterpolator', 'NDInterpolatorBase', 'NearestNDInterpolator', 'cKDTree', 'griddata']

def __dir__():
    if False:
        return 10
    return __all__

def __getattr__(name):
    if False:
        i = 10
        return i + 15
    return _sub_module_deprecation(sub_package='interpolate', module='ndgriddata', private_modules=['_ndgriddata'], all=__all__, attribute=name)