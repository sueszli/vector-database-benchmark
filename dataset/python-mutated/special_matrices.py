from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['tri', 'tril', 'triu', 'toeplitz', 'circulant', 'hankel', 'hadamard', 'leslie', 'kron', 'block_diag', 'companion', 'helmert', 'hilbert', 'invhilbert', 'pascal', 'invpascal', 'dft', 'fiedler', 'fiedler_companion', 'convolution_matrix', 'as_strided']

def __dir__():
    if False:
        print('Hello World!')
    return __all__

def __getattr__(name):
    if False:
        print('Hello World!')
    return _sub_module_deprecation(sub_package='linalg', module='special_matrices', private_modules=['_special_matrices'], all=__all__, attribute=name)