"""Tests for DecodeRaw op from parsing_ops."""
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test

class DecodeRawOpTest(test.TestCase):

    def testShapeInference(self):
        if False:
            return 10
        with ops.Graph().as_default():
            for dtype in [dtypes.bool, dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16, dtypes.int32, dtypes.int64, dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]:
                in_bytes = array_ops.placeholder(dtypes.string, shape=[None])
                decode = parsing_ops.decode_raw(in_bytes, dtype)
                self.assertEqual([None, None], decode.get_shape().as_list())

    def testToUint8(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertAllEqual([[ord('A')], [ord('a')]], parsing_ops.decode_raw(['A', 'a'], dtypes.uint8))
        self.assertAllEqual([[ord('w'), ord('e'), ord('r')], [ord('X'), ord('Y'), ord('Z')]], parsing_ops.decode_raw(['wer', 'XYZ'], dtypes.uint8))
        with self.assertRaisesOpError('DecodeRaw requires input strings to all be the same size, but element 1 has size 5 != 6'):
            self.evaluate(parsing_ops.decode_raw(['short', 'longer'], dtypes.uint8))

    def testToInt16(self):
        if False:
            i = 10
            return i + 15
        self.assertAllEqual([[ord('A') + ord('a') * 256, ord('B') + ord('C') * 256]], parsing_ops.decode_raw(['AaBC'], dtypes.uint16))
        with self.assertRaisesOpError('Input to DecodeRaw has length 3 that is not a multiple of 2, the size of int16'):
            self.evaluate(parsing_ops.decode_raw(['123', '456'], dtypes.int16))

    def testEndianness(self):
        if False:
            return 10
        self.assertAllEqual([[67305985]], parsing_ops.decode_raw(['\x01\x02\x03\x04'], dtypes.int32, little_endian=True))
        self.assertAllEqual([[16909060]], parsing_ops.decode_raw(['\x01\x02\x03\x04'], dtypes.int32, little_endian=False))
        self.assertAllEqual([[1 + 2j]], parsing_ops.decode_raw([b'\x00\x00\x80?\x00\x00\x00@'], dtypes.complex64, little_endian=True))
        self.assertAllEqual([[1 + 2j]], parsing_ops.decode_raw([b'?\x80\x00\x00@\x00\x00\x00'], dtypes.complex64, little_endian=False))

    def testToFloat16(self):
        if False:
            i = 10
            return i + 15
        result = np.matrix([[1, -2, -3, 4]], dtype='<f2')
        self.assertAllEqual(result, parsing_ops.decode_raw([result.tobytes()], dtypes.float16))

    def testToBool(self):
        if False:
            return 10
        result = np.matrix([[True, False, False, True]], dtype='<b1')
        self.assertAllEqual(result, parsing_ops.decode_raw([result.tobytes()], dtypes.bool))

    def testToComplex64(self):
        if False:
            while True:
                i = 10
        result = np.matrix([[1 + 1j, 2 - 2j, -3 + 3j, -4 - 4j]], dtype='<c8')
        self.assertAllEqual(result, parsing_ops.decode_raw([result.tobytes()], dtypes.complex64))

    def testToComplex128(self):
        if False:
            while True:
                i = 10
        result = np.matrix([[1 + 1j, 2 - 2j, -3 + 3j, -4 - 4j]], dtype='<c16')
        self.assertAllEqual(result, parsing_ops.decode_raw([result.tobytes()], dtypes.complex128))

    def testEmptyStringInput(self):
        if False:
            print('Hello World!')
        for num_inputs in range(3):
            result = parsing_ops.decode_raw([''] * num_inputs, dtypes.float16)
            self.assertEqual((num_inputs, 0), self.evaluate(result).shape)

    def testToUInt16(self):
        if False:
            while True:
                i = 10
        self.assertAllEqual([[255 + 238 * 256, 221 + 204 * 256]], parsing_ops.decode_raw([b'\xff\xee\xdd\xcc'], dtypes.uint16))
        with self.assertRaisesOpError('Input to DecodeRaw has length 3 that is not a multiple of 2, the size of uint16'):
            self.evaluate(parsing_ops.decode_raw(['123', '456'], dtypes.uint16))
if __name__ == '__main__':
    test.main()