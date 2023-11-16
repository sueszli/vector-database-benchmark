from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['ConvexHull', 'Delaunay', 'HalfspaceIntersection', 'QhullError', 'Voronoi', 'tsearch']

def __dir__():
    if False:
        return 10
    return __all__

def __getattr__(name):
    if False:
        for i in range(10):
            print('nop')
    return _sub_module_deprecation(sub_package='spatial', module='qhull', private_modules=['_qhull'], all=__all__, attribute=name)