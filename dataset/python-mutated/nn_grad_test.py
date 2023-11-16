"""Tests for Python ops defined in nn_grad.py."""
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_grad
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test

class SoftmaxOpTest(test.TestCase):

    def testSoftmaxGradGradExtendType(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():

            def f(x):
                if False:
                    print('Hello World!')
                assert x.dtype == dtypes.float32
                with backprop.GradientTape() as tape:
                    tape.watch(x)
                    y = nn_ops.softmax(x)
                return tape.gradient(y, x)
            x = constant_op.constant([[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
            error = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(f, [x]))
            self.assertLess(error, 0.0001)

class Relu6OpTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testRelu6GradGrad(self):
        if False:
            i = 10
            return i + 15
        inputs = constant_op.constant([[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
        x_init_value = np.array([[-3.5, -1.5, 2, 4], [4.5, 7.5, 8.5, 11]])
        r = nn_ops.relu6(inputs)
        r_g = gradients_impl.gradients(r, inputs)[0]
        with self.cached_session():
            error = gradient_checker.compute_gradient_error(inputs, inputs.get_shape().as_list(), r_g, r_g.get_shape().as_list(), x_init_value=x_init_value)
            self.assertLess(error, 0.0001)

class Conv2dOpTest(test.TestCase):

    def run_test(self, x, y):
        if False:
            return 10
        with self.test_session():
            error = gradient_checker.compute_gradient_error(x, x.get_shape().as_list(), y, y.get_shape().as_list())
            self.assertLess(error, 0.001)

    @test_util.run_deprecated_v1
    def testConv2dGradWRTInput(self):
        if False:
            i = 10
            return i + 15
        x = array_ops.placeholder(dtype=dtypes.float32, shape=[1, 4, 4, 3], name='input')
        f = constant_op.constant([0.5], dtype=dtypes.float32, shape=[2, 2, 3, 2], name='filter')
        y = nn_ops.conv2d(x, f, [1, 1, 1, 1], 'SAME')
        self.run_test(x, y)

    @test_util.run_deprecated_v1
    def testConv2dGradWRTFilter(self):
        if False:
            return 10
        x = constant_op.constant([0.5], dtype=dtypes.float32, shape=[1, 4, 4, 3], name='input')
        f = array_ops.placeholder(dtype=dtypes.float32, shape=[2, 2, 3, 2], name='filter')
        y = nn_ops.conv2d(x, f, [1, 1, 1, 1], 'SAME')
        self.run_test(f, y)

    @test_util.run_deprecated_v1
    def testConv2dBackpropFilterGrad(self):
        if False:
            i = 10
            return i + 15
        x = array_ops.placeholder(dtype=dtypes.float32, shape=[1, 4, 4, 3], name='input')
        f = constant_op.constant([0.5], dtype=dtypes.float32, shape=[2, 2, 3, 2], name='filter')
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        out = nn_impl.depthwise_conv2d(x, f, strides, padding)
        grad_wrt_input = gradients_impl.gradients(out, x)[0]
        self.run_test(f, grad_wrt_input)
        grad_wrt_filter = gradients_impl.gradients(out, f)[0]
        self.run_test(x, grad_wrt_filter)

class DepthwiseConv2dTest(test.TestCase):

    def run_test(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        with self.test_session():
            error = gradient_checker.compute_gradient_error(x, x.get_shape().as_list(), y, y.get_shape().as_list())
            self.assertLess(error, 0.001)

    @test_util.run_deprecated_v1
    def testDepthwiseConv2dGradWRTInput(self):
        if False:
            print('Hello World!')
        x = array_ops.placeholder(dtype=dtypes.float32, shape=[1, 4, 4, 3], name='input')
        f = constant_op.constant([0.5], dtype=dtypes.float32, shape=[2, 2, 3, 2], name='filter')
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        y = nn_impl.depthwise_conv2d(x, f, strides, padding)
        self.run_test(x, y)

    @test_util.run_deprecated_v1
    def testDepthwiseConv2dGradWRTFilter(self):
        if False:
            i = 10
            return i + 15
        x = constant_op.constant([0.5], dtype=dtypes.float32, shape=[1, 4, 4, 3], name='input')
        f = array_ops.placeholder(dtype=dtypes.float32, shape=[2, 2, 3, 2], name='filter')
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        y = nn_impl.depthwise_conv2d(x, f, strides, padding)
        self.run_test(f, y)

    @test_util.run_deprecated_v1
    def testDepthwiseConv2dBackpropFilterGrad(self):
        if False:
            for i in range(10):
                print('nop')
        x = array_ops.placeholder(dtype=dtypes.float32, shape=[1, 4, 4, 3], name='input')
        f = constant_op.constant([0.5], dtype=dtypes.float32, shape=[2, 2, 3, 2], name='filter')
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        out = nn_impl.depthwise_conv2d(x, f, strides, padding)
        grad_wrt_input = gradients_impl.gradients(out, x)[0]
        self.run_test(f, grad_wrt_input)
        grad_wrt_filter = gradients_impl.gradients(out, f)[0]
        self.run_test(x, grad_wrt_filter)

class EluGradOpTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testEluGradGradWRTgrad_ys(self):
        if False:
            for i in range(10):
                print('nop')
        inputs = constant_op.constant([[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
        dummy = constant_op.constant([[3, 1, -1, -2], [9, 8, 7, 6]], dtype=dtypes.float32)
        elu = gen_nn_ops.elu(inputs)
        elu_grad = gradients_impl.gradients(elu, inputs, grad_ys=dummy)[0]
        with self.cached_session():
            error = gradient_checker.compute_gradient_error(dummy, dummy.shape, elu_grad, elu_grad.shape)
            self.assertLess(error, 0.0001)

    @test_util.run_deprecated_v1
    def testEluGradGradWRTinputs(self):
        if False:
            return 10
        inputs = constant_op.constant([[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
        dummy = constant_op.constant([[3, 1, -1, -2], [9, 8, 7, 6]], dtype=dtypes.float32)
        elu = gen_nn_ops.elu(inputs)
        elu_grad = gradients_impl.gradients(elu, inputs, grad_ys=dummy)[0]
        with self.cached_session():
            error = gradient_checker.compute_gradient_error(inputs, inputs.shape, elu_grad, elu_grad.shape)
            self.assertLess(error, 0.0001)

class SeluGradOpTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testSeluGradGradWRTgrad_ys(self):
        if False:
            for i in range(10):
                print('nop')
        inputs = constant_op.constant([[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
        dummy = constant_op.constant([[3, 1, -1, -2], [9, 8, 7, 6]], dtype=dtypes.float32)
        selu = gen_nn_ops.selu(inputs)
        selu_grad = gradients_impl.gradients(selu, inputs, grad_ys=dummy)[0]
        with self.cached_session():
            error = gradient_checker.compute_gradient_error(dummy, dummy.shape, selu_grad, selu_grad.shape)
            self.assertLess(error, 0.0001)

    @test_util.run_deprecated_v1
    def testSeluGradGradWRTinputs(self):
        if False:
            return 10
        inputs = constant_op.constant([[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
        dummy = constant_op.constant([[3, 1, -1, -2], [9, 8, 7, 6]], dtype=dtypes.float32)
        selu = gen_nn_ops.selu(inputs)
        selu_grad = gradients_impl.gradients(selu, inputs, grad_ys=dummy)[0]
        with self.cached_session():
            error = gradient_checker.compute_gradient_error(inputs, inputs.shape, selu_grad, selu_grad.shape)
            self.assertLess(error, 0.0001)

class SwishGradOpTest(test.TestCase):

    def testSwishGrad(self):
        if False:
            print('Hello World!')
        features = constant_op.constant([[-2, -1, 1, 3]], dtype=dtypes.float32)
        beta = constant_op.constant(0.25, dtype=dtypes.float32)
        with self.cached_session():
            (theoretical, numerical) = gradient_checker_v2.compute_gradient(nn_impl.swish, [features, beta])
            error = gradient_checker_v2.max_error(theoretical, numerical)
            self.assertLess(error, 0.0001)
if __name__ == '__main__':
    test.main()