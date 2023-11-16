"""Functional tests for morphological filtering operations."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad
from tensorflow.python.platform import test

class DilationTest(test.TestCase, parameterized.TestCase):

    def _VerifyValues(self, image, kernel, strides, rates, padding, out, use_gpu, dtype):
        if False:
            while True:
                i = 10
        'Verifies the output values of the dilation function.\n\n    Args:\n      image: Input tensor with shape: [batch, in_height, in_width, channels].\n      kernel: Filter tensor with shape: [filter_height, filter_width, channels].\n      strides: Output strides, specified as [stride_height, stride_width].\n      rates: Atrous rates, specified as [rate_height, rate_width].\n      padding: Padding type.\n      out: Expected output.\n      use_gpu: Whether we are running on GPU.\n    '
        strides = [1] + strides + [1]
        rates = [1] + rates + [1]
        with self.cached_session(use_gpu=use_gpu):
            out_tensor = nn_ops.dilation2d(constant_op.constant(image, dtype=dtype), constant_op.constant(kernel, dtype=dtype), strides=strides, rates=rates, padding=padding, name='dilation2d')
            self.assertAllCloseAccordingToType(out, self.evaluate(out_tensor))

    def _testDilationValidPadding(self, use_gpu, dtype):
        if False:
            return 10
        image = [[[[0.1], [0.2]], [[0.3], [0.4]]]]
        kernel = [[[0.4], [0.3]], [[0.1], [0.0]]]
        out = [[[[0.5]]]]
        self._VerifyValues(image, kernel, strides=[1, 1], rates=[1, 1], padding='VALID', out=out, use_gpu=use_gpu, dtype=dtype)

    def _testDilationSamePadding(self, use_gpu, dtype):
        if False:
            print('Hello World!')
        image = [[[[0.1], [0.2]], [[0.3], [0.4]]]]
        kernel = [[[0.4], [0.3]], [[0.1], [0.0]]]
        out = [[[[0.5], [0.6]], [[0.7], [0.8]]]]
        self._VerifyValues(image, kernel, strides=[1, 1], rates=[1, 1], padding='SAME', out=out, use_gpu=use_gpu, dtype=dtype)

    def _testDilationSamePaddingDepth(self, use_gpu, dtype):
        if False:
            while True:
                i = 10
        image = [[[[0.1, 0.2, 0.0], [0.2, 0.3, 0.1]], [[0.3, 0.4, 0.2], [0.4, 0.5, 0.3]]]]
        kernel = [[[0.4, 0.5, 0.3], [0.3, 0.4, 0.2]], [[0.1, 0.2, 0.0], [0.0, 0.1, -0.1]]]
        out = [[[[0.5, 0.7, 0.3], [0.6, 0.8, 0.4]], [[0.7, 0.9, 0.5], [0.8, 1.0, 0.6]]]]
        self._VerifyValues(image, kernel, strides=[1, 1], rates=[1, 1], padding='SAME', out=out, use_gpu=use_gpu, dtype=dtype)

    def _testDilationSamePaddingBatch(self, use_gpu, dtype):
        if False:
            i = 10
            return i + 15
        image = [[[[0.1], [0.2]], [[0.3], [0.4]]], [[[0.2], [0.3]], [[0.4], [0.5]]]]
        kernel = [[[0.4], [0.3]], [[0.1], [0.0]]]
        out = [[[[0.5], [0.6]], [[0.7], [0.8]]], [[[0.6], [0.7]], [[0.8], [0.9]]]]
        self._VerifyValues(image, kernel, strides=[1, 1], rates=[1, 1], padding='SAME', out=out, use_gpu=use_gpu, dtype=dtype)

    def _testDilationValidPaddingNonSquareWindow(self, use_gpu, dtype):
        if False:
            while True:
                i = 10
        image = [[[[0.1], [0.2]], [[0.3], [0.4]]]]
        kernel = [[[0.4], [0.3]]]
        out = [[[[0.5]], [[0.7]]]]
        self._VerifyValues(image, kernel, strides=[1, 1], rates=[1, 1], padding='VALID', out=out, use_gpu=use_gpu, dtype=dtype)

    def _testDilationSamePaddingRate(self, use_gpu, dtype):
        if False:
            i = 10
            return i + 15
        image = [[[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]], [[0.7], [0.8], [0.9]]]]
        kernel = [[[0.4], [0.3]], [[0.1], [0.2]]]
        out = [[[[0.7], [0.8], [0.6]], [[1.0], [1.1], [0.9]], [[0.8], [0.9], [0.9]]]]
        self._VerifyValues(image, kernel, strides=[1, 1], rates=[2, 2], padding='SAME', out=out, use_gpu=use_gpu, dtype=dtype)

    def _testDilationValidPaddingUnevenStride(self, use_gpu, dtype):
        if False:
            for i in range(10):
                print('nop')
        image = [[[[0.1], [0.2], [0.3], [0.4]], [[0.5], [0.6], [0.7], [0.8]], [[0.9], [1.0], [1.1], [1.2]]]]
        kernel = [[[0.4], [0.3]], [[0.1], [0.2]]]
        out = [[[[0.8], [1.0]], [[1.2], [1.4]]]]
        self._VerifyValues(image, kernel, strides=[1, 2], rates=[1, 1], padding='VALID', out=out, use_gpu=use_gpu, dtype=dtype)

    @parameterized.parameters(dtypes.float32, dtypes.bfloat16)
    def testDilation(self, dtype):
        if False:
            i = 10
            return i + 15
        for use_gpu in (True, False):
            self._testDilationValidPadding(use_gpu, dtype)
            self._testDilationSamePadding(use_gpu, dtype)
            self._testDilationSamePaddingDepth(use_gpu, dtype)
            self._testDilationSamePaddingBatch(use_gpu, dtype)
            self._testDilationValidPaddingNonSquareWindow(use_gpu, dtype)
            self._testDilationSamePaddingRate(use_gpu, dtype)
            self._testDilationValidPaddingUnevenStride(use_gpu, dtype)

    def _ConstructAndTestGradient(self, image_shape, kernel_shape, strides, rates, padding, use_gpu, dtype=dtypes.float32):
        if False:
            print('Hello World!')
        'Verifies the gradients of the dilation function.\n\n    Args:\n      image_shape: Input shape, [batch, in_height, in_width, channels].\n      kernel_shape: Filter shape, [filter_height, filter_width, channels].\n      strides: Output strides, specified as [stride_height, stride_width].\n      rates: Atrous rates, specified as [rate_height, rate_width].\n      padding: Padding type.\n      use_gpu: Whether we are running on GPU.\n    '
        assert image_shape[3] == kernel_shape[2]
        np.random.seed(1)
        image = np.random.random_sample(image_shape).astype(np.float32)
        kernel = np.random.random_sample(kernel_shape).astype(np.float32)
        strides = [1] + strides + [1]
        rates = [1] + rates + [1]
        image_tensor = constant_op.constant(image, shape=image_shape, name='input', dtype=dtype)
        kernel_tensor = constant_op.constant(kernel, shape=kernel_shape, name='filter', dtype=dtype)

        def compute_dilation2d(image_tensor, kernel_tensor):
            if False:
                i = 10
                return i + 15
            return nn_ops.dilation2d(image_tensor, kernel_tensor, strides=strides, rates=rates, padding=padding, name='dilation2d')
        with test_util.device(use_gpu=use_gpu):
            with self.cached_session():
                err1 = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(lambda x: compute_dilation2d(x, kernel_tensor), [image_tensor]))
                err2 = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(lambda x: compute_dilation2d(image_tensor, x), [kernel_tensor]))
                err = max(err1, err2)
        print('Dilation gradient error = %f' % err)
        if dtype == dtypes.bfloat16:
            self.assertLess(err, 4.0)
        else:
            self.assertLess(err, 0.0001)

    def _testDilationGradValidPadding_1x1x1(self, use_gpu, dtype):
        if False:
            while True:
                i = 10
        self._ConstructAndTestGradient(image_shape=[1, 3, 3, 1], kernel_shape=[1, 1, 1], strides=[1, 1], rates=[1, 1], padding='VALID', use_gpu=use_gpu, dtype=dtype)

    def _testDilationGradDeterminismError(self, use_gpu, dtype):
        if False:
            while True:
                i = 10
        if use_gpu and test.is_gpu_available(cuda_only=True):
            try:
                config.enable_op_determinism()
                with self.assertRaisesRegexp(errors_impl.UnimplementedError, 'Determinism is not yet supported for Dilation2DBackpropInput.'):
                    self._ConstructAndTestGradient(image_shape=[1, 3, 3, 1], kernel_shape=[1, 1, 1], strides=[1, 1], rates=[1, 1], padding='VALID', use_gpu=use_gpu, dtype=dtype)
            finally:
                config.disable_op_determinism()
        else:
            try:
                config.enable_op_determinism()
                self._ConstructAndTestGradient(image_shape=[1, 3, 3, 1], kernel_shape=[1, 1, 1], strides=[1, 1], rates=[1, 1], padding='VALID', use_gpu=use_gpu, dtype=dtype)
            finally:
                config.disable_op_determinism()

    def _testDilationGradSamePadding_1x1x1(self, use_gpu, dtype):
        if False:
            while True:
                i = 10
        self._ConstructAndTestGradient(image_shape=[1, 3, 3, 1], kernel_shape=[1, 1, 1], strides=[1, 1], rates=[1, 1], padding='SAME', use_gpu=use_gpu, dtype=dtype)

    def _testDilationGradSamePadding_1x1x2(self, use_gpu, dtype):
        if False:
            for i in range(10):
                print('nop')
        self._ConstructAndTestGradient(image_shape=[1, 3, 3, 2], kernel_shape=[1, 1, 2], strides=[1, 1], rates=[1, 1], padding='SAME', use_gpu=use_gpu, dtype=dtype)

    def _testDilationGradValidPadding_2x2x1(self, use_gpu, dtype):
        if False:
            for i in range(10):
                print('nop')
        self._ConstructAndTestGradient(image_shape=[1, 3, 3, 1], kernel_shape=[2, 2, 1], strides=[1, 1], rates=[1, 1], padding='VALID', use_gpu=use_gpu, dtype=dtype)

    def _testDilationGradSamePadding_2x2x1(self, use_gpu, dtype):
        if False:
            while True:
                i = 10
        self._ConstructAndTestGradient(image_shape=[1, 3, 3, 1], kernel_shape=[2, 2, 1], strides=[1, 1], rates=[1, 1], padding='SAME', use_gpu=use_gpu, dtype=dtype)

    def _testDilationGradSamePaddingBatch_2x2x1(self, use_gpu, dtype):
        if False:
            for i in range(10):
                print('nop')
        self._ConstructAndTestGradient(image_shape=[4, 3, 3, 1], kernel_shape=[2, 2, 1], strides=[1, 1], rates=[1, 1], padding='SAME', use_gpu=use_gpu, dtype=dtype)

    def _testDilationGradSamePadding_2x2x4(self, use_gpu, dtype):
        if False:
            return 10
        self._ConstructAndTestGradient(image_shape=[1, 3, 3, 4], kernel_shape=[2, 2, 4], strides=[1, 1], rates=[1, 1], padding='SAME', use_gpu=use_gpu, dtype=dtype)

    @parameterized.parameters(dtypes.float32, dtypes.bfloat16)
    def testDilationGrad(self, dtype):
        if False:
            return 10
        for use_gpu in (True, False):
            self._testDilationGradDeterminismError(use_gpu, dtype)
            self._testDilationGradValidPadding_1x1x1(use_gpu, dtype)
            self._testDilationGradSamePadding_1x1x1(use_gpu, dtype)
            self._testDilationGradSamePadding_1x1x2(use_gpu, dtype)
            self._testDilationGradValidPadding_2x2x1(use_gpu, dtype)
            self._testDilationGradSamePadding_2x2x1(use_gpu, dtype)
            self._testDilationGradSamePaddingBatch_2x2x1(use_gpu, dtype)
            self._testDilationGradSamePadding_2x2x4(use_gpu, dtype)

class ErosionTest(test.TestCase):

    def _VerifyValues(self, image, kernel, strides, rates, padding, out, use_gpu):
        if False:
            print('Hello World!')
        'Verifies the output values of the erosion function.\n\n    Args:\n      image: Input tensor with shape: [batch, in_height, in_width, channels].\n      kernel: Filter tensor with shape: [filter_height, filter_width, channels].\n      strides: Output strides, specified as [stride_height, stride_width].\n      rates: Atrous rates, specified as [rate_height, rate_width].\n      padding: Padding type.\n      out: Expected output.\n      use_gpu: Whether we are running on GPU.\n    '
        strides = [1] + strides + [1]
        rates = [1] + rates + [1]
        with self.cached_session(use_gpu=use_gpu):
            out_tensor = nn_ops.erosion2d(constant_op.constant(image), constant_op.constant(kernel), strides=strides, rates=rates, padding=padding, name='erosion2d')
            self.assertAllClose(out, self.evaluate(out_tensor))

    def _testErosionValidPadding(self, use_gpu):
        if False:
            print('Hello World!')
        image = [[[[0.1], [0.2]], [[0.3], [0.4]]]]
        kernel = [[[0.4], [0.3]], [[0.1], [0.0]]]
        out = [[[[0.0]]]]
        self._VerifyValues(image, kernel, strides=[1, 1], rates=[1, 1], padding='VALID', out=out, use_gpu=use_gpu)

    def _testErosionSamePadding(self, use_gpu):
        if False:
            for i in range(10):
                print('nop')
        image = [[[[0.1], [0.2]], [[0.3], [0.4]]]]
        kernel = [[[0.4], [0.3]], [[0.1], [0.0]]]
        out = [[[[0.0], [0.1]], [[0.3], [0.4]]]]
        self._VerifyValues(image, kernel, strides=[1, 1], rates=[1, 1], padding='SAME', out=out, use_gpu=use_gpu)

    def _testErosionSamePaddingDepth(self, use_gpu):
        if False:
            print('Hello World!')
        image = [[[[0.1, 0.2, 0.0], [0.2, 0.3, 0.1]], [[0.3, 0.4, 0.2], [0.4, 0.5, 0.3]]]]
        kernel = [[[0.4, 0.5, 0.3], [0.3, 0.4, 0.2]], [[0.1, 0.2, 0.0], [0.0, 0.1, -0.1]]]
        out = [[[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]], [[0.3, 0.3, 0.3], [0.4, 0.4, 0.4]]]]
        self._VerifyValues(image, kernel, strides=[1, 1], rates=[1, 1], padding='SAME', out=out, use_gpu=use_gpu)

    def _testErosionSamePaddingBatch(self, use_gpu):
        if False:
            print('Hello World!')
        image = [[[[0.1], [0.2]], [[0.3], [0.4]]], [[[0.2], [0.3]], [[0.4], [0.5]]]]
        kernel = [[[0.4], [0.3]], [[0.1], [0.0]]]
        out = [[[[0.0], [0.1]], [[0.3], [0.4]]], [[[0.1], [0.2]], [[0.4], [0.5]]]]
        self._VerifyValues(image, kernel, strides=[1, 1], rates=[1, 1], padding='SAME', out=out, use_gpu=use_gpu)

    def _testErosionValidPaddingNonSquareWindow(self, use_gpu):
        if False:
            print('Hello World!')
        image = [[[[0.1], [0.2]], [[0.3], [0.4]]]]
        kernel = [[[0.4], [0.3]]]
        out = [[[[-0.2]], [[0.0]]]]
        self._VerifyValues(image, kernel, strides=[1, 1], rates=[1, 1], padding='VALID', out=out, use_gpu=use_gpu)

    def _testErosionSamePaddingRate(self, use_gpu):
        if False:
            print('Hello World!')
        image = [[[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]], [[0.7], [0.8], [0.9]]]]
        kernel = [[[0.4], [0.3]], [[0.1], [0.2]]]
        out = [[[[0.1], [0.1], [0.2]], [[0.1], [-0.1], [0.0]], [[0.4], [0.2], [0.3]]]]
        self._VerifyValues(image, kernel, strides=[1, 1], rates=[2, 2], padding='SAME', out=out, use_gpu=use_gpu)

    def _testErosionValidPaddingUnevenStride(self, use_gpu):
        if False:
            i = 10
            return i + 15
        image = [[[[0.1], [0.2], [0.3], [0.4]], [[0.5], [0.6], [0.7], [0.8]], [[0.9], [1.0], [1.1], [1.2]]]]
        kernel = [[[0.4], [0.3]], [[0.1], [0.2]]]
        out = [[[[-0.1], [0.1]], [[0.3], [0.5]]]]
        self._VerifyValues(image, kernel, strides=[1, 2], rates=[1, 1], padding='VALID', out=out, use_gpu=use_gpu)

    def testErosion(self):
        if False:
            print('Hello World!')
        for use_gpu in (True, False):
            self._testErosionValidPadding(use_gpu)
            self._testErosionSamePadding(use_gpu)
            self._testErosionSamePaddingDepth(use_gpu)
            self._testErosionSamePaddingBatch(use_gpu)
            self._testErosionValidPaddingNonSquareWindow(use_gpu)
            self._testErosionSamePaddingRate(use_gpu)
            self._testErosionValidPaddingUnevenStride(use_gpu)

    def _ConstructAndTestGradient(self, image_shape, kernel_shape, strides, rates, padding, use_gpu):
        if False:
            i = 10
            return i + 15
        'Verifies the gradients of the erosion function.\n\n    Args:\n      image_shape: Input shape, [batch, in_height, in_width, channels].\n      kernel_shape: Filter shape, [filter_height, filter_width, channels].\n      strides: Output strides, specified as [stride_height, stride_width].\n      rates: Atrous rates, specified as [rate_height, rate_width].\n      padding: Padding type.\n      use_gpu: Whether we are running on GPU.\n    '
        assert image_shape[3] == kernel_shape[2]
        np.random.seed(1)
        image = np.random.random_sample(image_shape).astype(np.float32)
        kernel = np.random.random_sample(kernel_shape).astype(np.float32)
        strides = [1] + strides + [1]
        rates = [1] + rates + [1]
        image_tensor = constant_op.constant(image, shape=image_shape, name='input')
        kernel_tensor = constant_op.constant(kernel, shape=kernel_shape, name='filter')

        def compute_erosion2d(image_tensor, kernel_tensor):
            if False:
                for i in range(10):
                    print('nop')
            return nn_ops.erosion2d(image_tensor, kernel_tensor, strides=strides, rates=rates, padding=padding, name='erosion2d')
        with test_util.device(use_gpu=use_gpu):
            with self.cached_session():
                err1 = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(lambda x: compute_erosion2d(x, kernel_tensor), [image_tensor]))
                err2 = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(lambda x: compute_erosion2d(image_tensor, x), [kernel_tensor]))
                err = max(err1, err2)
        print('Erosion gradient error = %f' % err)
        self.assertLess(err, 0.0001)

    def _testErosionGradValidPadding_1x1x1(self, use_gpu):
        if False:
            return 10
        self._ConstructAndTestGradient(image_shape=[1, 3, 3, 1], kernel_shape=[1, 1, 1], strides=[1, 1], rates=[1, 1], padding='VALID', use_gpu=use_gpu)

    def _testErosionGradSamePadding_1x1x1(self, use_gpu):
        if False:
            print('Hello World!')
        self._ConstructAndTestGradient(image_shape=[1, 3, 3, 1], kernel_shape=[1, 1, 1], strides=[1, 1], rates=[1, 1], padding='SAME', use_gpu=use_gpu)

    def _testErosionGradSamePadding_1x1x2(self, use_gpu):
        if False:
            while True:
                i = 10
        self._ConstructAndTestGradient(image_shape=[1, 3, 3, 2], kernel_shape=[1, 1, 2], strides=[1, 1], rates=[1, 1], padding='SAME', use_gpu=use_gpu)

    def _testErosionGradValidPadding_2x2x1(self, use_gpu):
        if False:
            for i in range(10):
                print('nop')
        self._ConstructAndTestGradient(image_shape=[1, 3, 3, 1], kernel_shape=[2, 2, 1], strides=[1, 1], rates=[1, 1], padding='VALID', use_gpu=use_gpu)

    def _testErosionGradSamePadding_2x2x1(self, use_gpu):
        if False:
            for i in range(10):
                print('nop')
        self._ConstructAndTestGradient(image_shape=[1, 3, 3, 1], kernel_shape=[2, 2, 1], strides=[1, 1], rates=[1, 1], padding='SAME', use_gpu=use_gpu)

    def _testErosionGradSamePaddingBatch_2x2x1(self, use_gpu):
        if False:
            while True:
                i = 10
        self._ConstructAndTestGradient(image_shape=[4, 3, 3, 1], kernel_shape=[2, 2, 1], strides=[1, 1], rates=[1, 1], padding='SAME', use_gpu=use_gpu)

    def _testErosionGradSamePadding_2x2x4(self, use_gpu):
        if False:
            print('Hello World!')
        self._ConstructAndTestGradient(image_shape=[1, 3, 3, 4], kernel_shape=[2, 2, 4], strides=[1, 1], rates=[1, 1], padding='SAME', use_gpu=use_gpu)

    def testErosionGrad(self):
        if False:
            while True:
                i = 10
        for use_gpu in (True, False):
            self._testErosionGradValidPadding_1x1x1(use_gpu)
            self._testErosionGradSamePadding_1x1x1(use_gpu)
            self._testErosionGradSamePadding_1x1x2(use_gpu)
            self._testErosionGradValidPadding_2x2x1(use_gpu)
            self._testErosionGradSamePadding_2x2x1(use_gpu)
            self._testErosionGradSamePaddingBatch_2x2x1(use_gpu)
            self._testErosionGradSamePadding_2x2x4(use_gpu)
if __name__ == '__main__':
    test.main()