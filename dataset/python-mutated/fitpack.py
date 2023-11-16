from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['BSpline', 'bisplev', 'bisplrep', 'dblint', 'insert', 'spalde', 'splantider', 'splder', 'splev', 'splint', 'splprep', 'splrep', 'sproot']

def __dir__():
    if False:
        print('Hello World!')
    return __all__

def __getattr__(name):
    if False:
        print('Hello World!')
    return _sub_module_deprecation(sub_package='interpolate', module='fitpack', private_modules=['_fitpack_py'], all=__all__, attribute=name)