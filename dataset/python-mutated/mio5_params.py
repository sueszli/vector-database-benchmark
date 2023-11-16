import warnings
from . import _mio5_params
__all__ = ['MDTYPES', 'MatlabFunction', 'MatlabObject', 'MatlabOpaque', 'NP_TO_MTYPES', 'NP_TO_MXTYPES', 'OPAQUE_DTYPE', 'codecs_template', 'mat_struct', 'mclass_dtypes_template', 'mclass_info', 'mdtypes_template', 'miCOMPRESSED', 'miDOUBLE', 'miINT16', 'miINT32', 'miINT64', 'miINT8', 'miMATRIX', 'miSINGLE', 'miUINT16', 'miUINT32', 'miUINT64', 'miUINT8', 'miUTF16', 'miUTF32', 'miUTF8', 'mxCELL_CLASS', 'mxCHAR_CLASS', 'mxDOUBLE_CLASS', 'mxFUNCTION_CLASS', 'mxINT16_CLASS', 'mxINT32_CLASS', 'mxINT64_CLASS', 'mxINT8_CLASS', 'mxOBJECT_CLASS', 'mxOBJECT_CLASS_FROM_MATRIX_H', 'mxOPAQUE_CLASS', 'mxSINGLE_CLASS', 'mxSPARSE_CLASS', 'mxSTRUCT_CLASS', 'mxUINT16_CLASS', 'mxUINT32_CLASS', 'mxUINT64_CLASS', 'mxUINT8_CLASS', 'convert_dtypes']

def __dir__():
    if False:
        while True:
            i = 10
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    if name not in __all__:
        raise AttributeError(f'scipy.io.matlab.mio5_params is deprecated and has no attribute {name}. Try looking in scipy.io.matlab instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.io.matlab` namespace, the `scipy.io.matlab.mio5_params` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_mio5_params, name)