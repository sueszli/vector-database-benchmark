"""Functional tests for BatchToSpace op.

Additional tests are included in spacetobatch_op_test.py, where the BatchToSpace
op is tested in tandem with its reverse SpaceToBatch op.
"""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test

class PythonOpImpl(object):

    @staticmethod
    def batch_to_space(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return array_ops.batch_to_space(*args, **kwargs)

class CppOpImpl(object):

    @staticmethod
    def batch_to_space(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return gen_array_ops.batch_to_space(*args, **kwargs)

class BatchToSpaceDepthToSpace(test.TestCase, parameterized.TestCase, PythonOpImpl):

    @parameterized.parameters(np.float32, dtypes.bfloat16.as_numpy_dtype)
    @test_util.run_deprecated_v1
    def testDepthToSpaceTranspose(self, dtype):
        if False:
            while True:
                i = 10
        x = np.arange(20 * 5 * 8 * 7, dtype=dtype).reshape([20, 5, 8, 7])
        block_size = 2
        for crops_dtype in [dtypes.int64, dtypes.int32]:
            crops = array_ops.zeros((2, 2), dtype=crops_dtype)
            y1 = self.batch_to_space(x, crops, block_size=block_size)
            y2 = array_ops.transpose(array_ops.depth_to_space(array_ops.transpose(x, [3, 1, 2, 0]), block_size=block_size), [3, 1, 2, 0])
            with self.cached_session():
                self.assertAllEqual(y1, y2)

class BatchToSpaceDepthToSpaceCpp(BatchToSpaceDepthToSpace, CppOpImpl):
    pass

class BatchToSpaceErrorHandlingTest(test.TestCase, PythonOpImpl):

    @test_util.run_deprecated_v1
    def testInputWrongDimMissingBatch(self):
        if False:
            i = 10
            return i + 15
        x_np = [[[1], [2]], [[3], [4]]]
        crops = np.zeros((2, 2), dtype=np.int32)
        block_size = 2
        with self.assertRaises(ValueError):
            _ = self.batch_to_space(x_np, crops, block_size)

    @test_util.run_deprecated_v1
    def testBlockSize0(self):
        if False:
            print('Hello World!')
        x_np = [[[[1], [2]], [[3], [4]]]]
        crops = np.zeros((2, 2), dtype=np.int32)
        block_size = 0
        with self.assertRaises(ValueError):
            out_tf = self.batch_to_space(x_np, crops, block_size)
            self.evaluate(out_tf)

    @test_util.run_deprecated_v1
    def testBlockSizeOne(self):
        if False:
            while True:
                i = 10
        x_np = [[[[1], [2]], [[3], [4]]]]
        crops = np.zeros((2, 2), dtype=np.int32)
        block_size = 1
        with self.assertRaises(ValueError):
            out_tf = self.batch_to_space(x_np, crops, block_size)
            out_tf.eval()

    @test_util.run_deprecated_v1
    def testBlockSizeLarger(self):
        if False:
            while True:
                i = 10
        x_np = [[[[1], [2]], [[3], [4]]]]
        crops = np.zeros((2, 2), dtype=np.int32)
        block_size = 10
        with self.assertRaises(ValueError):
            out_tf = self.batch_to_space(x_np, crops, block_size)
            self.evaluate(out_tf)

    @test_util.run_deprecated_v1
    def testBlockSizeSquaredNotDivisibleBatch(self):
        if False:
            return 10
        x_np = [[[[1], [2], [3]], [[3], [4], [7]]]]
        crops = np.zeros((2, 2), dtype=np.int32)
        block_size = 3
        with self.assertRaises(ValueError):
            _ = self.batch_to_space(x_np, crops, block_size)

    @test_util.run_deprecated_v1
    def testUnknownShape(self):
        if False:
            i = 10
            return i + 15
        t = self.batch_to_space(array_ops.placeholder(dtypes.float32), array_ops.placeholder(dtypes.int32), block_size=4)
        self.assertEqual(4, t.get_shape().ndims)

class BatchToSpaceErrorHandlingCppTest(BatchToSpaceErrorHandlingTest, CppOpImpl):
    pass

class BatchToSpaceNDErrorHandlingTest(test.TestCase):

    def _testStaticShape(self, input_shape, block_shape, paddings, error):
        if False:
            while True:
                i = 10
        block_shape = np.array(block_shape)
        paddings = np.array(paddings)
        with self.assertRaises(error):
            _ = array_ops.batch_to_space_nd(np.zeros(input_shape, np.float32), block_shape, paddings)

    def _testDynamicShape(self, input_shape, block_shape, paddings):
        if False:
            while True:
                i = 10
        block_shape = np.array(block_shape)
        paddings = np.array(paddings)
        input_placeholder = array_ops.placeholder(dtypes.float32)
        block_shape_placeholder = array_ops.placeholder(dtypes.int32, shape=block_shape.shape)
        paddings_placeholder = array_ops.placeholder(dtypes.int32)
        t = array_ops.batch_to_space_nd(input_placeholder, block_shape_placeholder, paddings_placeholder)
        with self.assertRaises(ValueError):
            _ = t.eval({input_placeholder: np.zeros(input_shape, np.float32), block_shape_placeholder: block_shape, paddings_placeholder: paddings})

    def _testShape(self, input_shape, block_shape, paddings, error):
        if False:
            return 10
        self._testStaticShape(input_shape, block_shape, paddings, error)
        self._testDynamicShape(input_shape, block_shape, paddings)

    @test_util.run_deprecated_v1
    def testInputWrongDimMissingBatch(self):
        if False:
            print('Hello World!')
        self._testShape([2, 2], [2, 2], [[0, 0], [0, 0]], ValueError)
        self._testShape([2, 2, 3], [2, 2, 3], [[0, 0], [0, 0]], ValueError)

    @test_util.run_deprecated_v1
    def testBlockSize0(self):
        if False:
            while True:
                i = 10
        self._testShape([1, 2, 2, 1], [0, 1], [[0, 0], [0, 0]], ValueError)

    @test_util.run_deprecated_v1
    def testBlockSizeNegative(self):
        if False:
            for i in range(10):
                print('nop')
        self._testShape([1, 2, 2, 1], [-1, 1], [[0, 0], [0, 0]], ValueError)

    @test_util.run_deprecated_v1
    def testNegativePadding(self):
        if False:
            print('Hello World!')
        self._testShape([1, 2, 2], [1, 1], [[0, -1], [0, 0]], ValueError)

    @test_util.run_deprecated_v1
    def testCropTooLarge(self):
        if False:
            while True:
                i = 10
        self._testShape([1 * 2 * 2, 2, 3, 1], [2, 2], [[3, 2], [0, 0]], ValueError)

    @test_util.run_deprecated_v1
    def testBlockSizeSquaredNotDivisibleBatch(self):
        if False:
            return 10
        self._testShape([3, 1, 1, 1], [2, 3], [[0, 0], [0, 0]], ValueError)

    @test_util.run_deprecated_v1
    def testUnknownShape(self):
        if False:
            while True:
                i = 10
        _ = array_ops.batch_to_space_nd(array_ops.placeholder(dtypes.float32), array_ops.placeholder(dtypes.int32, shape=(2,)), array_ops.placeholder(dtypes.int32))
        t = array_ops.batch_to_space_nd(array_ops.placeholder(dtypes.float32, shape=(None, None, None, None)), array_ops.placeholder(dtypes.int32, shape=(2,)), array_ops.placeholder(dtypes.int32))
        self.assertEqual(4, t.get_shape().ndims)
        t = array_ops.batch_to_space_nd(array_ops.placeholder(dtypes.float32, shape=(None, None, None, 2)), array_ops.placeholder(dtypes.int32, shape=(2,)), array_ops.placeholder(dtypes.int32))
        self.assertEqual([None, None, None, 2], t.get_shape().as_list())
        t = array_ops.batch_to_space_nd(array_ops.placeholder(dtypes.float32, shape=(3 * 2 * 3, None, None, 2)), [2, 3], array_ops.placeholder(dtypes.int32))
        self.assertEqual([3, None, None, 2], t.get_shape().as_list())
        t = array_ops.batch_to_space_nd(array_ops.placeholder(dtypes.float32, shape=(3 * 2 * 3, None, 2, 2)), [2, 3], [[1, 1], [0, 1]])
        self.assertEqual([3, None, 5, 2], t.get_shape().as_list())
        t = array_ops.batch_to_space_nd(array_ops.placeholder(dtypes.float32, shape=(3 * 2 * 3, 2, 1, 2)), [2, 3], [[1, 1], [0, 0]])
        self.assertEqual([3, 2, 3, 2], t.get_shape().as_list())

class BatchToSpaceGradientTest(test.TestCase, PythonOpImpl):

    def _checkGrad(self, x, crops, block_size):
        if False:
            i = 10
            return i + 15
        assert 4 == x.ndim
        with self.cached_session():
            tf_x = ops.convert_to_tensor(x)
            tf_y = self.batch_to_space(tf_x, crops, block_size)
            epsilon = 1e-05
            (x_jacob_t, x_jacob_n) = gradient_checker.compute_gradient(tf_x, x.shape, tf_y, tf_y.get_shape().as_list(), x_init_value=x, delta=epsilon)
        self.assertAllClose(x_jacob_t, x_jacob_n, rtol=0.01, atol=epsilon)

    def _compare(self, b, h, w, d, block_size, crop_beg, crop_end):
        if False:
            return 10
        block_size_sq = block_size * block_size
        x = np.random.normal(0, 1, b * h * w * d * block_size_sq).astype(np.float32).reshape([b * block_size * block_size, h, w, d])
        crops = np.array([[crop_beg, crop_end], [crop_beg, crop_end]], dtype=np.int32)
        self._checkGrad(x, crops, block_size)

    @test_util.run_deprecated_v1
    def testSmall(self):
        if False:
            while True:
                i = 10
        block_size = 2
        crop_beg = 0
        crop_end = 0
        self._compare(1, 2, 3, 5, block_size, crop_beg, crop_end)

    @test_util.run_deprecated_v1
    def testSmall2(self):
        if False:
            i = 10
            return i + 15
        block_size = 2
        crop_beg = 0
        crop_end = 0
        self._compare(2, 4, 3, 2, block_size, crop_beg, crop_end)

    @test_util.run_deprecated_v1
    def testSmallCrop1x1(self):
        if False:
            return 10
        block_size = 2
        crop_beg = 1
        crop_end = 1
        self._compare(1, 2, 3, 5, block_size, crop_beg, crop_end)

class BatchToSpaceGradientCppTest(BatchToSpaceGradientTest, CppOpImpl):
    pass

class BatchToSpaceNDGradientTest(test.TestCase):

    def _checkGrad(self, x, block_shape, crops, crops_dtype):
        if False:
            return 10
        block_shape = np.array(block_shape)
        crops = constant_op.constant(np.array(crops).reshape((len(block_shape), 2)), crops_dtype)
        with self.cached_session():
            tf_x = ops.convert_to_tensor(x)
            tf_y = array_ops.batch_to_space_nd(tf_x, block_shape, crops)
            epsilon = 1e-05
            (x_jacob_t, x_jacob_n) = gradient_checker.compute_gradient(tf_x, x.shape, tf_y, tf_y.get_shape().as_list(), x_init_value=x, delta=epsilon)
        self.assertAllClose(x_jacob_t, x_jacob_n, rtol=0.01, atol=epsilon)

    def _compare(self, input_shape, block_shape, crops, crops_dtype):
        if False:
            return 10
        input_shape = list(input_shape)
        input_shape[0] *= np.prod(block_shape)
        x = np.random.normal(0, 1, np.prod(input_shape)).astype(np.float32).reshape(input_shape)
        self._checkGrad(x, block_shape, crops, crops_dtype)

    @test_util.run_deprecated_v1
    def testSmall(self):
        if False:
            while True:
                i = 10
        for dtype in [dtypes.int64, dtypes.int32]:
            self._compare([1, 2, 3, 5], [2, 2], [[0, 0], [0, 0]], dtype)

    @test_util.run_deprecated_v1
    def testSmall2(self):
        if False:
            return 10
        for dtype in [dtypes.int64, dtypes.int32]:
            self._compare([2, 4, 3, 2], [2, 2], [[0, 0], [0, 0]], dtype)

    @test_util.run_deprecated_v1
    def testSmallCrop1x1(self):
        if False:
            print('Hello World!')
        for dtype in [dtypes.int64, dtypes.int32]:
            self._compare([1, 2, 3, 5], [2, 2], [[1, 1], [1, 1]], dtype)
if __name__ == '__main__':
    test.main()