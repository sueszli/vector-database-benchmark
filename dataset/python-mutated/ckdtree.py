from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['cKDTree', 'cKDTreeNode', 'coo_entries', 'operator', 'ordered_pairs', 'os', 'scipy', 'threading']

def __dir__():
    if False:
        return 10
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    return _sub_module_deprecation(sub_package='spatial', module='ckdtree', private_modules=['_ckdtree'], all=__all__, attribute=name)