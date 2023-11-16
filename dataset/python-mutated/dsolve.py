from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['MatrixRankWarning', 'SuperLU', 'factorized', 'spilu', 'splu', 'spsolve', 'spsolve_triangular', 'use_solver', 'linsolve', 'test']
dsolve_modules = ['linsolve']

def __dir__():
    if False:
        return 10
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    return _sub_module_deprecation(sub_package='sparse.linalg', module='dsolve', private_modules=['_dsolve'], all=__all__, attribute=name)