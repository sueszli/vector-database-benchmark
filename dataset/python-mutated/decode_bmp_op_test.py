"""Tests for DecodeBmpOp."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import test

class DecodeBmpOpTest(test.TestCase):

    def testex1(self):
        if False:
            for i in range(10):
                print('nop')
        img_bytes = [[[0, 0, 255], [0, 255, 0]], [[255, 0, 0], [255, 255, 255]]]
        encoded_bytes = [66, 77, 70, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 16, 0, 0, 0, 19, 11, 0, 0, 19, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0]
        byte_string = bytes(bytearray(encoded_bytes))
        img_in = constant_op.constant(byte_string, dtype=dtypes.string)
        decode = array_ops.squeeze(image_ops.decode_bmp(img_in))
        with self.cached_session():
            decoded = self.evaluate(decode)
            self.assertAllEqual(decoded, img_bytes)

    def testGrayscale(self):
        if False:
            print('Hello World!')
        img_bytes = [[[255], [0]], [[255], [0]]]
        encoded_bytes = [66, 77, 61, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 16, 0, 0, 0, 19, 11, 0, 0, 19, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0]
        byte_string = bytes(bytearray(encoded_bytes))
        img_in = constant_op.constant(byte_string, dtype=dtypes.string)
        decode = image_ops.decode_bmp(img_in)
        with self.cached_session():
            decoded = self.evaluate(decode)
            self.assertAllEqual(decoded, img_bytes)
if __name__ == '__main__':
    test.main()