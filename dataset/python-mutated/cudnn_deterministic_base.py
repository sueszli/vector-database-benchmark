"""Tests for deterministic cuDNN functionality."""
import collections
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
LayerShapeNHWC = collections.namedtuple('LayerShapeNHWC', 'batch, height, width, channels')
FilterShape2D = collections.namedtuple('FilterShape2D', 'height, width, in_channels, out_channels')
FilterShape2DTranspose = collections.namedtuple('FilterShape2DTranspose', 'height, width, out_channels, in_channels')
LayerShapeNCDHW = collections.namedtuple('LayerShapeNCDHW', 'batch, channels, depth, height, width')
FilterShape3D = collections.namedtuple('FilterShape3D', 'depth, height, width, in_channels, out_channels')

class ConvolutionTest(test.TestCase):
    """Tests for deterministic cuDNN functionality."""

    def _random_data_op(self, shape):
        if False:
            return 10
        return constant_op.constant(2 * np.random.random_sample(shape) - 1, dtype=dtypes.float32)

    def _random_out_op(self, in_shape, filter_shape, strides, padding, dilations):
        if False:
            for i in range(10):
                print('nop')
        in_op = self._random_data_op(in_shape)
        filter_op = self._random_data_op(filter_shape)
        conv_op = nn_ops.conv2d(in_op, filter_op, strides=strides, padding=padding, dilations=dilations)
        out_shape = conv_op.get_shape()
        out_op = self._random_data_op(out_shape)
        return out_op

    def _assert_reproducible(self, operation):
        if False:
            for i in range(10):
                print('nop')
        with test_util.force_gpu():
            result_1 = operation()
            result_2 = operation()
        self.assertAllEqual(result_1, result_2)

    @test_util.run_cuda_only
    def testConvForwardDefaultAlgorithmChoice(self):
        if False:
            i = 10
            return i + 15
        in_shape = LayerShapeNCDHW(batch=2, channels=3, depth=5, height=7, width=6)
        filter_shape = FilterShape3D(depth=3, height=3, width=3, in_channels=3, out_channels=2)
        in_op = self._random_data_op(in_shape)
        filter_op = self._random_data_op(filter_shape)
        self._assert_reproducible(lambda : nn_ops.conv3d(in_op, filter_op, strides=[1, 1, 1, 1, 1], padding='VALID', data_format='NCDHW', dilations=[1, 1, 2, 2, 2]))

    @test_util.run_cuda_only
    def testConvForwardXLA(self):
        if False:
            return 10
        in_shape = LayerShapeNCDHW(batch=2, channels=8, depth=5, height=12, width=15)
        filter_shape = FilterShape3D(depth=3, height=3, width=3, in_channels=8, out_channels=1)
        in_op = self._random_data_op(in_shape)
        filter_op = self._random_data_op(filter_shape)
        self._assert_reproducible(lambda : nn_ops.conv3d(in_op, filter_op, strides=[1, 1, 1, 1, 1], padding='VALID', data_format='NCDHW', dilations=[1, 1, 2, 2, 2]))

    @test_util.run_cuda_only
    def testConvBackwardFilterGradient(self, rate=1):
        if False:
            return 10
        in_shape = LayerShapeNHWC(batch=8, height=64, width=64, channels=8)
        filter_shape = FilterShape2D(height=3, width=3, in_channels=8, out_channels=8)
        in_op = self._random_data_op(in_shape)
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        dilations = [1, rate, rate, 1]
        out_op = self._random_out_op(in_shape, filter_shape, strides, padding, dilations)
        self._assert_reproducible(lambda : nn_ops.conv2d_backprop_filter(in_op, filter_shape, out_op, strides=strides, padding=padding, dilations=dilations))

    @test_util.run_cuda_only
    def testConvBackwardFilterGradientWithDilations(self):
        if False:
            return 10
        self.testConvBackwardFilterGradient(rate=2)

    @test_util.run_cuda_only
    def testConvBackwardInputGradient(self, rate=1):
        if False:
            print('Hello World!')
        in_shape = LayerShapeNHWC(batch=1, height=16, width=16, channels=1)
        filter_shape = FilterShape2D(height=7, width=7, in_channels=1, out_channels=3)
        filter_op = self._random_data_op(filter_shape)
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        dilations = [1, rate, rate, 1]
        out_op = self._random_out_op(in_shape, filter_shape, strides, padding, dilations)
        self._assert_reproducible(lambda : nn_ops.conv2d_backprop_input(in_shape, filter_op, out_op, strides=strides, padding=padding, dilations=dilations))

    @test_util.run_cuda_only
    def testConvBackwardInputGradientWithDilations(self):
        if False:
            i = 10
            return i + 15
        self.testConvBackwardInputGradient(rate=2)

    @test_util.run_cuda_only
    def testConvTransposeForward(self, rate=1):
        if False:
            return 10
        in_channels = 3
        out_channels = 1
        in_shape = LayerShapeNHWC(batch=1, height=16, width=16, channels=in_channels)
        filter_shape = FilterShape2DTranspose(height=7, width=7, out_channels=out_channels, in_channels=in_channels)
        in_op = self._random_data_op(in_shape)
        filter_op = self._random_data_op(filter_shape)
        out_shape = LayerShapeNHWC(batch=in_shape.batch, height=in_shape.height, width=in_shape.width, channels=out_channels)
        self._assert_reproducible(lambda : nn_ops.conv2d_transpose_v2(in_op, filter_op, out_shape, strides=1, padding='SAME', data_format='NHWC', dilations=[1, rate, rate, 1]))

    @test_util.run_cuda_only
    def testConvTransposeForwardWithDilations(self):
        if False:
            print('Hello World!')
        self.testConvTransposeForward(rate=2)

    @test_util.run_cuda_only
    def testConvTransposeBackwardFilterGradient(self, rate=1):
        if False:
            for i in range(10):
                print('nop')
        in_channels = 8
        out_channels = 8
        in_shape = LayerShapeNHWC(batch=8, height=64, width=64, channels=in_channels)
        filter_shape = FilterShape2DTranspose(height=3, width=3, out_channels=out_channels, in_channels=in_channels)
        in_op = self._random_data_op(in_shape)
        filter_op = self._random_data_op(filter_shape)
        out_shape = LayerShapeNHWC(batch=in_shape.batch, height=in_shape.height, width=in_shape.width, channels=out_channels)
        upstream_gradients = self._random_data_op(out_shape)

        def gradient():
            if False:
                for i in range(10):
                    print('nop')
            with backprop.GradientTape() as tape:
                tape.watch(filter_op)
                op_output = nn_ops.conv2d_transpose_v2(in_op, filter_op, out_shape, strides=1, padding='SAME', data_format='NHWC', dilations=[1, rate, rate, 1])
                gradient_injector_output = op_output * upstream_gradients
            return tape.gradient(gradient_injector_output, [filter_op])[0]
        self._assert_reproducible(gradient)

    @test_util.run_cuda_only
    def testConvTransposeBackwardFilterGradientWithDilations(self):
        if False:
            return 10
        self.testConvTransposeBackwardFilterGradient(rate=2)

    @test_util.run_cuda_only
    def testConvTransposeBackwardInputGradient(self, rate=1):
        if False:
            return 10
        in_channels = 1
        out_channels = 3
        in_shape = LayerShapeNHWC(batch=1, height=16, width=16, channels=in_channels)
        filter_shape = FilterShape2DTranspose(height=7, width=7, out_channels=out_channels, in_channels=in_channels)
        in_op = self._random_data_op(in_shape)
        filter_op = self._random_data_op(filter_shape)
        out_shape = LayerShapeNHWC(batch=in_shape.batch, height=in_shape.height, width=in_shape.width, channels=out_channels)
        upstream_gradients = self._random_data_op(out_shape)

        def gradient():
            if False:
                return 10
            with backprop.GradientTape() as tape:
                tape.watch(in_op)
                op_output = nn_ops.conv2d_transpose_v2(in_op, filter_op, out_shape, strides=1, padding='SAME', data_format='NHWC', dilations=[1, rate, rate, 1])
                gradient_injector_output = op_output * upstream_gradients
            return tape.gradient(gradient_injector_output, [in_op])[0]
        self._assert_reproducible(gradient)

    @test_util.run_cuda_only
    def testConvTransposeBackwardInputGradientWithDilations(self):
        if False:
            print('Hello World!')
        self.testConvTransposeBackwardInputGradient(rate=2)