from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['qr', 'qr_multiply', 'rq', 'get_lapack_funcs', 'safecall']

def __dir__():
    if False:
        while True:
            i = 10
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    return _sub_module_deprecation(sub_package='linalg', module='decomp_qr', private_modules=['_decomp_qr'], all=__all__, attribute=name)