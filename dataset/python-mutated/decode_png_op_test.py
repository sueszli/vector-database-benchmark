"""Tests for DecodePngOp."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
import tensorflow.python.ops.nn_grad
from tensorflow.python.platform import test

class DecodePngOpTest(test.TestCase):

    def test16bit(self):
        if False:
            return 10
        img_bytes = [[0, 255], [1024, 1024 + 255]]
        encoded_bytes = [137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 2, 16, 0, 0, 0, 0, 7, 77, 142, 187, 0, 0, 0, 21, 73, 68, 65, 84, 120, 156, 98, 98, 96, 96, 248, 207, 194, 2, 36, 1, 1, 0, 0, 255, 255, 6, 60, 1, 10, 68, 160, 26, 131, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130]
        byte_string = bytes(bytearray(encoded_bytes))
        img_in = constant_op.constant(byte_string, dtype=dtypes.string)
        decode = array_ops.squeeze(image_ops.decode_png(img_in, dtype=dtypes.uint16))
        with self.cached_session():
            decoded = self.evaluate(decode)
            self.assertAllEqual(decoded, img_bytes)
if __name__ == '__main__':
    test.main()