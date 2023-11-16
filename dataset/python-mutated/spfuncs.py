from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['csr_count_blocks', 'estimate_blocksize', 'count_blocks']

def __dir__():
    if False:
        return 10
    return __all__

def __getattr__(name):
    if False:
        print('Hello World!')
    return _sub_module_deprecation(sub_package='sparse', module='spfuncs', private_modules=['_spfuncs'], all=__all__, attribute=name)