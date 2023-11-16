"""Functional tests for SpacetoDepth op."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

class SpaceToDepthTest(test.TestCase):

    def _testOne(self, inputs, block_size, outputs, dtype=dtypes.float32):
        if False:
            i = 10
            return i + 15
        input_nhwc = math_ops.cast(inputs, dtype)
        x_tf = array_ops.space_to_depth(input_nhwc, block_size)
        self.assertAllEqual(self.evaluate(x_tf), outputs)
        if test_util.is_gpu_available():
            with test_util.force_gpu():
                input_nchw = test_util.NHWCToNCHW(input_nhwc)
                output_nchw = array_ops.space_to_depth(input_nchw, block_size, data_format='NCHW')
                output_nhwc = test_util.NCHWToNHWC(output_nchw)
                self.assertAllEqual(self.evaluate(output_nhwc), outputs)

    def testBasic(self):
        if False:
            while True:
                i = 10
        x_np = [[[[1], [2]], [[3], [4]]]]
        block_size = 2
        x_out = [[[[1, 2, 3, 4]]]]
        for dtype in [dtypes.float32, dtypes.float16, dtypes.bfloat16, dtypes.uint8]:
            self._testOne(x_np, block_size, x_out, dtype=dtype)

    def testLargerInput2x2(self):
        if False:
            i = 10
            return i + 15
        x_np = [[[[1], [2], [5], [6]], [[3], [4], [7], [8]], [[9], [10], [13], [14]], [[11], [12], [15], [16]]]]
        block_size = 2
        x_out = [[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]]
        self._testOne(x_np, block_size, x_out)

    def testLargerInput4x4(self):
        if False:
            while True:
                i = 10
        x_np = [[[[1], [2], [5], [6]], [[3], [4], [7], [8]], [[9], [10], [13], [14]], [[11], [12], [15], [16]]]]
        block_size = 4
        x_out = [[[[1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16]]]]
        self._testOne(x_np, block_size, x_out)

    def testDepthInterleaved(self):
        if False:
            print('Hello World!')
        x_np = [[[[1, 10], [2, 20]], [[3, 30], [4, 40]]]]
        block_size = 2
        x_out = [[[[1, 10, 2, 20, 3, 30, 4, 40]]]]
        self._testOne(x_np, block_size, x_out)

    def testDepthInterleavedDepth3(self):
        if False:
            print('Hello World!')
        x_np = [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]
        block_size = 2
        x_out = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
        self._testOne(x_np, block_size, x_out)

    def testDepthInterleavedLarge(self):
        if False:
            return 10
        x_np = [[[[1, 10], [2, 20], [5, 50], [6, 60]], [[3, 30], [4, 40], [7, 70], [8, 80]], [[9, 90], [10, 100], [13, 130], [14, 140]], [[11, 110], [12, 120], [15, 150], [16, 160]]]]
        block_size = 2
        x_out = [[[[1, 10, 2, 20, 3, 30, 4, 40], [5, 50, 6, 60, 7, 70, 8, 80]], [[9, 90, 10, 100, 11, 110, 12, 120], [13, 130, 14, 140, 15, 150, 16, 160]]]]
        self._testOne(x_np, block_size, x_out)

    def testBlockSize2Batch10(self):
        if False:
            i = 10
            return i + 15
        block_size = 2

        def batch_input_elt(i):
            if False:
                while True:
                    i = 10
            return [[[1 * i], [2 * i], [5 * i], [6 * i]], [[3 * i], [4 * i], [7 * i], [8 * i]], [[9 * i], [10 * i], [13 * i], [14 * i]], [[11 * i], [12 * i], [15 * i], [16 * i]]]

        def batch_output_elt(i):
            if False:
                print('Hello World!')
            return [[[1 * i, 2 * i, 3 * i, 4 * i], [5 * i, 6 * i, 7 * i, 8 * i]], [[9 * i, 10 * i, 11 * i, 12 * i], [13 * i, 14 * i, 15 * i, 16 * i]]]
        batch_size = 10
        x_np = [batch_input_elt(i) for i in range(batch_size)]
        x_out = [batch_output_elt(i) for i in range(batch_size)]
        self._testOne(x_np, block_size, x_out)

    def testBatchSize0(self):
        if False:
            i = 10
            return i + 15
        block_size = 2
        batch_size = 0
        x_np = array_ops.ones([batch_size, 4, 6, 3])
        x_out = array_ops.ones([batch_size, 2, 3, 12])
        self._testOne(x_np, block_size, x_out)

    def testNonSquare(self):
        if False:
            while True:
                i = 10
        x_np = [[[[1, 10], [2, 20]], [[3, 30], [4, 40]], [[5, 50], [6, 60]], [[7, 70], [8, 80]], [[9, 90], [10, 100]], [[11, 110], [12, 120]]]]
        block_size = 2
        x_out = [[[[1, 10, 2, 20, 3, 30, 4, 40]], [[5, 50, 6, 60, 7, 70, 8, 80]], [[9, 90, 10, 100, 11, 110, 12, 120]]]]
        self._testOne(x_np, block_size, x_out)

    def testInputWrongDimMissingDepth(self):
        if False:
            for i in range(10):
                print('nop')
        x_np = [[[1, 2], [3, 4]]]
        block_size = 2
        with self.assertRaises((ValueError, errors.InvalidArgumentError)):
            out_tf = array_ops.space_to_depth(x_np, block_size)
            self.evaluate(out_tf)

    def testInputWrongDimMissingBatch(self):
        if False:
            print('Hello World!')
        x_np = [[[1], [2]], [[3], [4]]]
        block_size = 2
        with self.assertRaises((ValueError, errors.InvalidArgumentError)):
            _ = array_ops.space_to_depth(x_np, block_size)

    def testBlockSize0(self):
        if False:
            while True:
                i = 10
        x_np = [[[[1], [2]], [[3], [4]]]]
        block_size = 0
        with self.assertRaises((ValueError, errors.InvalidArgumentError)):
            out_tf = array_ops.space_to_depth(x_np, block_size)
            self.evaluate(out_tf)

    def testBlockSizeOne(self):
        if False:
            for i in range(10):
                print('nop')
        x_np = [[[[1], [2]], [[3], [4]]]]
        block_size = 1
        with self.assertRaises((ValueError, errors.InvalidArgumentError)):
            out_tf = array_ops.space_to_depth(x_np, block_size)
            self.evaluate(out_tf)

    def testBlockSizeLarger(self):
        if False:
            print('Hello World!')
        x_np = [[[[1], [2]], [[3], [4]]]]
        block_size = 10
        with self.assertRaises((ValueError, errors.InvalidArgumentError)):
            out_tf = array_ops.space_to_depth(x_np, block_size)
            self.evaluate(out_tf)

    def testBlockSizeNotDivisibleWidth(self):
        if False:
            while True:
                i = 10
        x_np = [[[[1], [2], [3]], [[3], [4], [7]]]]
        block_size = 3
        with self.assertRaises((ValueError, errors.InvalidArgumentError)):
            _ = array_ops.space_to_depth(x_np, block_size)

    def testBlockSizeNotDivisibleHeight(self):
        if False:
            print('Hello World!')
        x_np = [[[[1], [2]], [[3], [4]], [[5], [6]]]]
        block_size = 3
        with self.assertRaises((ValueError, errors.InvalidArgumentError)):
            _ = array_ops.space_to_depth(x_np, block_size)

    def testBlockSizeNotDivisibleBoth(self):
        if False:
            i = 10
            return i + 15
        x_np = [[[[1], [2]], [[3], [4]]]]
        block_size = 3
        with self.assertRaises((ValueError, errors.InvalidArgumentError)):
            _ = array_ops.space_to_depth(x_np, block_size)

    def testUnknownShape(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            t = array_ops.space_to_depth(array_ops.placeholder(dtypes.float32), block_size=4)
            self.assertEqual(4, t.get_shape().ndims)

    def spaceToDepthUsingTranspose(self, tensor, block_size, data_format):
        if False:
            print('Hello World!')
        block_size_sq = block_size * block_size
        dtype = tensor.dtype
        if dtype == dtypes.qint8:
            tensor = array_ops.bitcast(tensor, dtypes.int8)
        if data_format == 'NHWC':
            (b, ih, iw, ic) = tensor.shape.as_list()
            assert ih % block_size == 0, (ih, block_size)
            assert iw % block_size == 0, (iw, block_size)
            (ow, oh, oc) = (iw // block_size, ih // block_size, ic * block_size_sq)
            tensor = array_ops.reshape(tensor, [b, oh, block_size, ow, block_size, ic])
            tensor = array_ops.transpose(tensor, [0, 1, 3, 2, 4, 5])
            tensor = array_ops.reshape(tensor, [b, oh, ow, oc])
        elif data_format == 'NCHW':
            (b, ic, ih, iw) = tensor.shape.as_list()
            assert ih % block_size == 0, (ih, block_size)
            assert iw % block_size == 0, (iw, block_size)
            (ow, oh, oc) = (iw // block_size, ih // block_size, ic * block_size_sq)
            tensor = array_ops.reshape(tensor, [b, ic, oh, block_size, ow, block_size])
            tensor = array_ops.transpose(tensor, [0, 3, 5, 1, 2, 4])
            tensor = array_ops.reshape(tensor, [b, oc, oh, ow])
        if dtype == dtypes.qint8:
            tensor = array_ops.bitcast(tensor, dtype)
        return tensor

    def compareToTranspose(self, batch_size, out_height, out_width, in_channels, block_size, data_format, data_type, use_gpu):
        if False:
            print('Hello World!')
        in_height = out_height * block_size
        in_width = out_width * block_size
        nhwc_input_shape = [batch_size, in_height, in_width, in_channels]
        nchw_input_shape = [batch_size, in_channels, in_height, in_width]
        total_size = np.prod(nhwc_input_shape)
        with test_util.force_cpu():
            if data_type == dtypes.qint8:
                x = [(f + 128) % 255 - 127 for f in range(total_size)]
                t = constant_op.constant(x, shape=nhwc_input_shape, dtype=dtypes.float32)
                (t, _, _) = gen_array_ops.quantize_v2(t, -128.0, 127.0, dtypes.qint8)
            else:
                assert data_type == dtypes.float32
                x = [f * 1.0 for f in range(total_size)]
                shape = nchw_input_shape if data_format == 'NCHW' else nhwc_input_shape
                t = constant_op.constant(x, shape=shape, dtype=dtypes.float32)
        with test_util.device(use_gpu):
            if data_format == 'NCHW_VECT_C':
                assert data_type == dtypes.qint8
                actual = array_ops.bitcast(t, dtypes.int8)
                actual = test_util.NHWCToNCHW_VECT_C(actual)
                actual = array_ops.bitcast(actual, dtypes.qint8)
                actual = array_ops.space_to_depth(actual, block_size, data_format=data_format)
                actual = array_ops.bitcast(actual, dtypes.int8)
                actual = test_util.NCHW_VECT_CToNHWC(actual)
                actual = array_ops.bitcast(actual, dtypes.qint8)
                expected = array_ops.bitcast(t, dtypes.int8)
                expected = math_ops.cast(expected, dtypes.float32)
                expected = self.spaceToDepthUsingTranspose(expected, block_size, 'NHWC')
                expected = math_ops.cast(expected, dtypes.int8)
                expected = array_ops.bitcast(expected, dtypes.qint8)
            else:
                actual = array_ops.space_to_depth(t, block_size, data_format=data_format)
                expected = self.spaceToDepthUsingTranspose(t, block_size, data_format)
            (actual_vals, expected_vals) = self.evaluate([actual, expected])
            self.assertTrue(np.array_equal(actual_vals, expected_vals))

    @test_util.disable_tfrt('b/169901260')
    def testAgainstTranspose(self):
        if False:
            i = 10
            return i + 15
        self.compareToTranspose(3, 2, 3, 1, 2, 'NHWC', dtypes.float32, False)
        self.compareToTranspose(1, 2, 3, 2, 2, 'NHWC', dtypes.float32, False)
        self.compareToTranspose(1, 2, 3, 2, 3, 'NHWC', dtypes.float32, False)
        self.compareToTranspose(3, 2, 3, 1, 2, 'NHWC', dtypes.qint8, False)
        self.compareToTranspose(1, 2, 3, 2, 2, 'NHWC', dtypes.qint8, False)
        self.compareToTranspose(1, 2, 3, 2, 3, 'NHWC', dtypes.qint8, False)
        if not test.is_gpu_available():
            tf_logging.info('skipping gpu tests since gpu not available')
            return
        self.compareToTranspose(3, 2, 3, 1, 2, 'NHWC', dtypes.float32, True)
        self.compareToTranspose(3, 2, 3, 2, 2, 'NHWC', dtypes.float32, True)
        self.compareToTranspose(3, 2, 3, 1, 2, 'NCHW', dtypes.float32, True)
        self.compareToTranspose(3, 2, 3, 2, 3, 'NCHW', dtypes.float32, True)
        self.compareToTranspose(5, 7, 11, 3, 2, 'NCHW', dtypes.float32, True)
        self.compareToTranspose(3, 2, 3, 4, 2, 'NCHW_VECT_C', dtypes.qint8, True)
        self.compareToTranspose(3, 2, 3, 8, 3, 'NCHW_VECT_C', dtypes.qint8, True)
        self.compareToTranspose(5, 7, 11, 12, 2, 'NCHW_VECT_C', dtypes.qint8, True)

class SpaceToDepthGradientTest(test.TestCase):

    def _checkGrad(self, x, block_size, data_format):
        if False:
            print('Hello World!')
        if data_format == 'NCHW' and (not test.is_gpu_available()):
            return
        assert 4 == x.ndim

        def func(x):
            if False:
                print('Hello World!')
            return array_ops.space_to_depth(x, block_size, data_format=data_format)
        with test_util.use_gpu():
            with self.cached_session():
                (theoretical, numerical) = gradient_checker_v2.compute_gradient(func, [ops.convert_to_tensor(x)])
                self.assertAllClose(theoretical, numerical, rtol=0.01, atol=0.01)

    def _compare(self, b, h, w, d, block_size, data_format):
        if False:
            while True:
                i = 10
        block_size_sq = block_size * block_size
        data = np.random.normal(0, 1, b * h * w * d * block_size_sq).astype(np.float32)
        if data_format == 'NHWC':
            x = data.reshape([b, h * block_size, w * block_size, d])
        else:
            x = data.reshape([b, d, h * block_size, w * block_size])
        self._checkGrad(x, block_size, data_format)

    def testSmall(self):
        if False:
            print('Hello World!')
        block_size = 2
        self._compare(1, 2, 3, 5, block_size, 'NHWC')
        self._compare(1, 2, 3, 5, block_size, 'NCHW')

    @test_util.run_deprecated_v1
    def testSmall2(self):
        if False:
            i = 10
            return i + 15
        block_size = 2
        self._compare(2, 4, 3, 2, block_size, 'NHWC')
        self._compare(2, 4, 3, 2, block_size, 'NCHW')
if __name__ == '__main__':
    test.main()