""" Constants and classes for matlab 5 read and write

See also mio5_utils.pyx where these same constants arise as c enums.

If you make changes in this file, don't forget to change mio5_utils.pyx
"""
import numpy as np
from ._miobase import convert_dtypes
__all__ = ['MDTYPES', 'MatlabFunction', 'MatlabObject', 'MatlabOpaque', 'NP_TO_MTYPES', 'NP_TO_MXTYPES', 'OPAQUE_DTYPE', 'codecs_template', 'mat_struct', 'mclass_dtypes_template', 'mclass_info', 'mdtypes_template', 'miCOMPRESSED', 'miDOUBLE', 'miINT16', 'miINT32', 'miINT64', 'miINT8', 'miMATRIX', 'miSINGLE', 'miUINT16', 'miUINT32', 'miUINT64', 'miUINT8', 'miUTF16', 'miUTF32', 'miUTF8', 'mxCELL_CLASS', 'mxCHAR_CLASS', 'mxDOUBLE_CLASS', 'mxFUNCTION_CLASS', 'mxINT16_CLASS', 'mxINT32_CLASS', 'mxINT64_CLASS', 'mxINT8_CLASS', 'mxOBJECT_CLASS', 'mxOBJECT_CLASS_FROM_MATRIX_H', 'mxOPAQUE_CLASS', 'mxSINGLE_CLASS', 'mxSPARSE_CLASS', 'mxSTRUCT_CLASS', 'mxUINT16_CLASS', 'mxUINT32_CLASS', 'mxUINT64_CLASS', 'mxUINT8_CLASS']
miINT8 = 1
miUINT8 = 2
miINT16 = 3
miUINT16 = 4
miINT32 = 5
miUINT32 = 6
miSINGLE = 7
miDOUBLE = 9
miINT64 = 12
miUINT64 = 13
miMATRIX = 14
miCOMPRESSED = 15
miUTF8 = 16
miUTF16 = 17
miUTF32 = 18
mxCELL_CLASS = 1
mxSTRUCT_CLASS = 2
mxOBJECT_CLASS = 3
mxCHAR_CLASS = 4
mxSPARSE_CLASS = 5
mxDOUBLE_CLASS = 6
mxSINGLE_CLASS = 7
mxINT8_CLASS = 8
mxUINT8_CLASS = 9
mxINT16_CLASS = 10
mxUINT16_CLASS = 11
mxINT32_CLASS = 12
mxUINT32_CLASS = 13
mxINT64_CLASS = 14
mxUINT64_CLASS = 15
mxFUNCTION_CLASS = 16
mxOPAQUE_CLASS = 17
mxOBJECT_CLASS_FROM_MATRIX_H = 18
mdtypes_template = {miINT8: 'i1', miUINT8: 'u1', miINT16: 'i2', miUINT16: 'u2', miINT32: 'i4', miUINT32: 'u4', miSINGLE: 'f4', miDOUBLE: 'f8', miINT64: 'i8', miUINT64: 'u8', miUTF8: 'u1', miUTF16: 'u2', miUTF32: 'u4', 'file_header': [('description', 'S116'), ('subsystem_offset', 'i8'), ('version', 'u2'), ('endian_test', 'S2')], 'tag_full': [('mdtype', 'u4'), ('byte_count', 'u4')], 'tag_smalldata': [('byte_count_mdtype', 'u4'), ('data', 'S4')], 'array_flags': [('data_type', 'u4'), ('byte_count', 'u4'), ('flags_class', 'u4'), ('nzmax', 'u4')], 'U1': 'U1'}
mclass_dtypes_template = {mxINT8_CLASS: 'i1', mxUINT8_CLASS: 'u1', mxINT16_CLASS: 'i2', mxUINT16_CLASS: 'u2', mxINT32_CLASS: 'i4', mxUINT32_CLASS: 'u4', mxINT64_CLASS: 'i8', mxUINT64_CLASS: 'u8', mxSINGLE_CLASS: 'f4', mxDOUBLE_CLASS: 'f8'}
mclass_info = {mxINT8_CLASS: 'int8', mxUINT8_CLASS: 'uint8', mxINT16_CLASS: 'int16', mxUINT16_CLASS: 'uint16', mxINT32_CLASS: 'int32', mxUINT32_CLASS: 'uint32', mxINT64_CLASS: 'int64', mxUINT64_CLASS: 'uint64', mxSINGLE_CLASS: 'single', mxDOUBLE_CLASS: 'double', mxCELL_CLASS: 'cell', mxSTRUCT_CLASS: 'struct', mxOBJECT_CLASS: 'object', mxCHAR_CLASS: 'char', mxSPARSE_CLASS: 'sparse', mxFUNCTION_CLASS: 'function', mxOPAQUE_CLASS: 'opaque'}
NP_TO_MTYPES = {'f8': miDOUBLE, 'c32': miDOUBLE, 'c24': miDOUBLE, 'c16': miDOUBLE, 'f4': miSINGLE, 'c8': miSINGLE, 'i8': miINT64, 'i4': miINT32, 'i2': miINT16, 'i1': miINT8, 'u8': miUINT64, 'u4': miUINT32, 'u2': miUINT16, 'u1': miUINT8, 'S1': miUINT8, 'U1': miUTF16, 'b1': miUINT8}
NP_TO_MXTYPES = {'f8': mxDOUBLE_CLASS, 'c32': mxDOUBLE_CLASS, 'c24': mxDOUBLE_CLASS, 'c16': mxDOUBLE_CLASS, 'f4': mxSINGLE_CLASS, 'c8': mxSINGLE_CLASS, 'i8': mxINT64_CLASS, 'i4': mxINT32_CLASS, 'i2': mxINT16_CLASS, 'i1': mxINT8_CLASS, 'u8': mxUINT64_CLASS, 'u4': mxUINT32_CLASS, 'u2': mxUINT16_CLASS, 'u1': mxUINT8_CLASS, 'S1': mxUINT8_CLASS, 'b1': mxUINT8_CLASS}
' Before release v7.1 (release 14) matlab (TM) used the system\ndefault character encoding scheme padded out to 16-bits. Release 14\nand later use Unicode. When saving character data, R14 checks if it\ncan be encoded in 7-bit ascii, and saves in that format if so.'
codecs_template = {miUTF8: {'codec': 'utf_8', 'width': 1}, miUTF16: {'codec': 'utf_16', 'width': 2}, miUTF32: {'codec': 'utf_32', 'width': 4}}

def _convert_codecs(template, byte_order):
    if False:
        i = 10
        return i + 15
    " Convert codec template mapping to byte order\n\n    Set codecs not on this system to None\n\n    Parameters\n    ----------\n    template : mapping\n       key, value are respectively codec name, and root name for codec\n       (without byte order suffix)\n    byte_order : {'<', '>'}\n       code for little or big endian\n\n    Returns\n    -------\n    codecs : dict\n       key, value are name, codec (as in .encode(codec))\n    "
    codecs = {}
    postfix = byte_order == '<' and '_le' or '_be'
    for (k, v) in template.items():
        codec = v['codec']
        try:
            ' '.encode(codec)
        except LookupError:
            codecs[k] = None
            continue
        if v['width'] > 1:
            codec += postfix
        codecs[k] = codec
    return codecs.copy()
MDTYPES = {}
for _bytecode in '<>':
    _def = {'dtypes': convert_dtypes(mdtypes_template, _bytecode), 'classes': convert_dtypes(mclass_dtypes_template, _bytecode), 'codecs': _convert_codecs(codecs_template, _bytecode)}
    MDTYPES[_bytecode] = _def

class mat_struct:
    """Placeholder for holding read data from structs.

    We use instances of this class when the user passes False as a value to the
    ``struct_as_record`` parameter of the :func:`scipy.io.loadmat` function.
    """
    pass

class MatlabObject(np.ndarray):
    """Subclass of ndarray to signal this is a matlab object.

    This is a simple subclass of :class:`numpy.ndarray` meant to be used
    by :func:`scipy.io.loadmat` and should not be instantiated directly.
    """

    def __new__(cls, input_array, classname=None):
        if False:
            print('Hello World!')
        obj = np.asarray(input_array).view(cls)
        obj.classname = classname
        return obj

    def __array_finalize__(self, obj):
        if False:
            return 10
        self.classname = getattr(obj, 'classname', None)

class MatlabFunction(np.ndarray):
    """Subclass for a MATLAB function.

    This is a simple subclass of :class:`numpy.ndarray` meant to be used
    by :func:`scipy.io.loadmat` and should not be directly instantiated.
    """

    def __new__(cls, input_array):
        if False:
            while True:
                i = 10
        obj = np.asarray(input_array).view(cls)
        return obj

class MatlabOpaque(np.ndarray):
    """Subclass for a MATLAB opaque matrix.

    This is a simple subclass of :class:`numpy.ndarray` meant to be used
    by :func:`scipy.io.loadmat` and should not be directly instantiated.
    """

    def __new__(cls, input_array):
        if False:
            return 10
        obj = np.asarray(input_array).view(cls)
        return obj
OPAQUE_DTYPE = np.dtype([('s0', 'O'), ('s1', 'O'), ('s2', 'O'), ('arr', 'O')])