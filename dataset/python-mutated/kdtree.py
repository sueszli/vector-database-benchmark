from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['KDTree', 'Rectangle', 'cKDTree', 'cKDTreeNode', 'distance_matrix', 'minkowski_distance', 'minkowski_distance_p']

def __dir__():
    if False:
        while True:
            i = 10
    return __all__

def __getattr__(name):
    if False:
        print('Hello World!')
    return _sub_module_deprecation(sub_package='spatial', module='kdtree', private_modules=['_kdtree'], all=__all__, attribute=name)