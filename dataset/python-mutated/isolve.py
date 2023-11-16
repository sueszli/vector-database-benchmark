from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['bicg', 'bicgstab', 'cg', 'cgs', 'gcrotmk', 'gmres', 'lgmres', 'lsmr', 'lsqr', 'minres', 'qmr', 'tfqmr', 'utils', 'iterative', 'test']

def __dir__():
    if False:
        for i in range(10):
            print('nop')
    return __all__

def __getattr__(name):
    if False:
        for i in range(10):
            print('nop')
    return _sub_module_deprecation(sub_package='sparse.linalg', module='isolve', private_modules=['_isolve'], all=__all__, attribute=name)