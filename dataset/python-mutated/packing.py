import cupy
from cupy import _core
_packbits_kernel = {'big': _core.ElementwiseKernel('raw T a, raw int32 a_size', 'uint8 packed', 'for (int j = 0; j < 8; ++j) {\n                    int k = i * 8 + j;\n                    int bit = k < a_size && a[k] != 0;\n                    packed |= bit << (7 - j);\n                }', 'cupy_packbits_big'), 'little': _core.ElementwiseKernel('raw T a, raw int32 a_size', 'uint8 packed', 'for (int j = 0; j < 8; ++j) {\n                    int k = i * 8 + j;\n                    int bit = k < a_size && a[k] != 0;\n                    packed |= bit << j;\n                }', 'cupy_packbits_little')}

def packbits(a, axis=None, bitorder='big'):
    if False:
        while True:
            i = 10
    "Packs the elements of a binary-valued array into bits in a uint8 array.\n\n    This function currently does not support ``axis`` option.\n\n    Args:\n        a (cupy.ndarray): Input array.\n        axis (int, optional): Not supported yet.\n        bitorder (str, optional): bit order to use when packing the array,\n            allowed values are `'little'` and `'big'`. Defaults to `'big'`.\n\n    Returns:\n        cupy.ndarray: The packed array.\n\n    .. note::\n        When the input array is empty, this function returns a copy of it,\n        i.e., the type of the output array is not necessarily always uint8.\n        This exactly follows the NumPy's behaviour (as of version 1.11),\n        alghough this is inconsistent to the documentation.\n\n    .. seealso:: :func:`numpy.packbits`\n    "
    if a.dtype.kind not in 'biu':
        raise TypeError('Expected an input array of integer or boolean data type')
    if axis is not None:
        raise NotImplementedError('axis option is not supported yet')
    if bitorder not in ('big', 'little'):
        raise ValueError("bitorder must be either 'big' or 'little'")
    a = a.ravel()
    packed_size = (a.size + 7) // 8
    packed = cupy.zeros((packed_size,), dtype=cupy.uint8)
    return _packbits_kernel[bitorder](a, a.size, packed)
_unpackbits_kernel = {'big': _core.ElementwiseKernel('raw uint8 a', 'T unpacked', 'unpacked = (a[i / 8] >> (7 - i % 8)) & 1;', 'cupy_unpackbits_big'), 'little': _core.ElementwiseKernel('raw uint8 a', 'T unpacked', 'unpacked = (a[i / 8] >> (i % 8)) & 1;', 'cupy_unpackbits_little')}

def unpackbits(a, axis=None, bitorder='big'):
    if False:
        print('Hello World!')
    "Unpacks elements of a uint8 array into a binary-valued output array.\n\n    This function currently does not support ``axis`` option.\n\n    Args:\n        a (cupy.ndarray): Input array.\n        bitorder (str, optional): bit order to use when unpacking the array,\n            allowed values are `'little'` and `'big'`. Defaults to `'big'`.\n\n    Returns:\n        cupy.ndarray: The unpacked array.\n\n    .. seealso:: :func:`numpy.unpackbits`\n    "
    if a.dtype != cupy.uint8:
        raise TypeError('Expected an input array of unsigned byte data type')
    if axis is not None:
        raise NotImplementedError('axis option is not supported yet')
    if bitorder not in ('big', 'little'):
        raise ValueError("bitorder must be either 'big' or 'little'")
    unpacked = cupy.ndarray(a.size * 8, dtype=cupy.uint8)
    return _unpackbits_kernel[bitorder](a, unpacked)