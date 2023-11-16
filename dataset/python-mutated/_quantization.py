"""
This file contains the code for Quantizing / Dequantizing floats.
"""
import numpy as np
from astropy.io.fits.hdu.base import BITPIX2DTYPE
from astropy.io.fits.hdu.compressed._compression import quantize_double_c, quantize_float_c, unquantize_double_c, unquantize_float_c
__all__ = ['Quantize']
DITHER_METHODS = {'NONE': 0, 'NO_DITHER': -1, 'SUBTRACTIVE_DITHER_1': 1, 'SUBTRACTIVE_DITHER_2': 2}

class QuantizationFailedException(Exception):
    pass

class Quantize:
    """
    Quantization of floating-point data following the FITS standard.
    """

    def __init__(self, *, row: int, dither_method: int, quantize_level: int, bitpix: int):
        if False:
            return 10
        super().__init__()
        self.row = row
        self.quantize_level = quantize_level
        self.dither_method = dither_method
        self.bitpix = bitpix

    def decode_quantized(self, buf, scale, zero):
        if False:
            while True:
                i = 10
        '\n        Unquantize data.\n\n        Parameters\n        ----------\n        buf : bytes or array_like\n            The buffer to unquantize.\n\n        Returns\n        -------\n        np.ndarray\n            The unquantized buffer.\n        '
        qbytes = np.asarray(buf)
        qbytes = qbytes.astype(qbytes.dtype.newbyteorder('='))
        if self.dither_method == -1:
            return qbytes * scale + zero
        if self.bitpix == -32:
            ubytes = unquantize_float_c(qbytes.tobytes(), self.row, qbytes.size, scale, zero, self.dither_method, 0, 0, 0.0, qbytes.dtype.itemsize)
        elif self.bitpix == -64:
            ubytes = unquantize_double_c(qbytes.tobytes(), self.row, qbytes.size, scale, zero, self.dither_method, 0, 0, 0.0, qbytes.dtype.itemsize)
        else:
            raise TypeError('bitpix should be one of -32 or -64')
        return np.frombuffer(ubytes, dtype=BITPIX2DTYPE[self.bitpix]).data

    def encode_quantized(self, buf):
        if False:
            i = 10
            return i + 15
        '\n        Quantize data.\n\n        Parameters\n        ----------\n        buf : bytes or array_like\n            The buffer to quantize.\n\n        Returns\n        -------\n        np.ndarray\n            A buffer with quantized data.\n        '
        uarray = np.asarray(buf)
        uarray = uarray.astype(uarray.dtype.newbyteorder('='))
        if uarray.dtype.itemsize == 4:
            (qbytes, status, scale, zero) = quantize_float_c(uarray.tobytes(), self.row, uarray.size, 1, 0, 0, self.quantize_level, self.dither_method)[:4]
        elif uarray.dtype.itemsize == 8:
            (qbytes, status, scale, zero) = quantize_double_c(uarray.tobytes(), self.row, uarray.size, 1, 0, 0, self.quantize_level, self.dither_method)[:4]
        if status == 0:
            raise QuantizationFailedException()
        else:
            return (np.frombuffer(qbytes, dtype=np.int32), scale, zero)