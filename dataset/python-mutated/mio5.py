import warnings
from . import _mio5
__all__ = ['mclass_info', 'mxCHAR_CLASS', 'mxSPARSE_CLASS', 'BytesIO', 'native_code', 'swapped_code', 'MatFileReader', 'docfiller', 'matdims', 'read_dtype', 'arr_to_chars', 'arr_dtype_number', 'MatWriteError', 'MatReadError', 'MatReadWarning', 'VarReader5', 'MatlabObject', 'MatlabFunction', 'MDTYPES', 'NP_TO_MTYPES', 'NP_TO_MXTYPES', 'miCOMPRESSED', 'miMATRIX', 'miINT8', 'miUTF8', 'miUINT32', 'mxCELL_CLASS', 'mxSTRUCT_CLASS', 'mxOBJECT_CLASS', 'mxDOUBLE_CLASS', 'mat_struct', 'ZlibInputStream', 'MatFile5Reader', 'varmats_from_mat', 'EmptyStructMarker', 'to_writeable', 'NDT_FILE_HDR', 'NDT_TAG_FULL', 'NDT_TAG_SMALL', 'NDT_ARRAY_FLAGS', 'VarWriter5', 'MatFile5Writer']

def __dir__():
    if False:
        while True:
            i = 10
    return __all__

def __getattr__(name):
    if False:
        print('Hello World!')
    if name not in __all__:
        raise AttributeError(f'scipy.io.matlab.mio5 is deprecated and has no attribute {name}. Try looking in scipy.io.matlab instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.io.matlab` namespace, the `scipy.io.matlab.mio5` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_mio5, name)