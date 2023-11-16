"""Tests for local response normalization."""
import copy
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
import tensorflow.python.ops.nn_grad
from tensorflow.python.platform import test

class LRNOpTest(test.TestCase):

    def _LRN(self, input_image, lrn_depth_radius=5, bias=1.0, alpha=1.0, beta=0.5):
        if False:
            i = 10
            return i + 15
        'Compute expected result.'
        output = copy.deepcopy(input_image)
        batch_size = input_image.shape[0]
        rows = input_image.shape[1]
        cols = input_image.shape[2]
        depth = input_image.shape[3]
        for b in range(batch_size):
            for r in range(rows):
                for c in range(cols):
                    for d in range(depth):
                        begin = max(0, d - lrn_depth_radius)
                        end = min(depth, d + lrn_depth_radius + 1)
                        patch = input_image[b, r, c, begin:end]
                        output[b, r, c, d] /= np.power(bias + alpha * np.sum(patch * patch), beta)
        return output

    def _RunAndVerify(self, dtype):
        if False:
            while True:
                i = 10
        with self.cached_session():
            shape = np.random.randint(1, 16, size=4)
            shape[3] += 1
            p = array_ops.placeholder(dtype, shape=shape)
            lrn_depth_radius = np.random.randint(1, min(8, shape[3]))
            bias = 1.0 + np.random.rand()
            alpha = 2.0 * np.random.rand()
            beta = 0.01 + 2.0 * np.random.rand()
            lrn_t = nn.local_response_normalization(p, name='lrn', depth_radius=lrn_depth_radius, bias=bias, alpha=alpha, beta=beta)
            params = {p: np.random.rand(*shape).astype('f')}
            result = lrn_t.eval(feed_dict=params)
        expected = self._LRN(params[p], lrn_depth_radius=lrn_depth_radius, bias=bias, alpha=alpha, beta=beta)
        err = np.amax(np.abs(result - expected))
        print('LRN error for bias ', bias, 'alpha ', alpha, ' beta ', beta, ' is ', err)
        if dtype == dtypes.float32:
            self.assertTrue(err < 0.0001)
        else:
            self.assertTrue(err < 0.01)
        self.assertShapeEqual(expected, lrn_t)

    @test_util.run_deprecated_v1
    def testCompute(self):
        if False:
            print('Hello World!')
        for _ in range(2):
            self._RunAndVerify(dtypes.float32)
            if not test.is_gpu_available():
                self._RunAndVerify(dtypes.float16)

    @test_util.run_deprecated_v1
    def testGradientsZeroInput(self):
        if False:
            return 10
        with self.session():
            shape = [4, 4, 4, 4]
            p = array_ops.placeholder(dtypes.float32, shape=shape)
            inp_array = np.zeros(shape).astype('f')
            lrn_op = nn.local_response_normalization(p, 2, 1.0, 0.0, 1.0, name='lrn')
            grad = gradients_impl.gradients([lrn_op], [p])[0]
            params = {p: inp_array}
            r = grad.eval(feed_dict=params)
        expected = np.ones(shape).astype('f')
        self.assertAllClose(r, expected)
        self.assertShapeEqual(expected, grad)

    @test_util.run_in_graph_and_eager_modes
    def testIncompatibleInputAndOutputImageShapes(self):
        if False:
            print('Hello World!')
        depth_radius = 1
        bias = 1.59018219
        alpha = 0.117728651
        beta = 0.404427052
        input_grads = random_ops.random_uniform(shape=[4, 4, 4, 4], minval=-10000, maxval=10000, dtype=dtypes.float32, seed=-2033)
        input_image = random_ops.random_uniform(shape=[4, 4, 4, 4], minval=-10000, maxval=10000, dtype=dtypes.float32, seed=-2033)
        invalid_output_image = random_ops.random_uniform(shape=[4, 4, 4, 4, 4, 4], minval=-10000, maxval=10000, dtype=dtypes.float32, seed=-2033)
        with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
            self.evaluate(nn.lrn_grad(input_grads=input_grads, input_image=input_image, output_image=invalid_output_image, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta))

    def _RunAndVerifyGradients(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            shape = np.random.randint(1, 5, size=4)
            shape[3] += 1
            lrn_depth_radius = np.random.randint(1, min(8, shape[3]))
            bias = 1.0 + np.random.rand()
            alpha = 1.0 * np.random.rand()
            beta = 0.01 + 1.0 * np.random.rand()
            if dtype == dtypes.float32:
                inp_array = np.random.rand(*shape).astype(np.float32)
            else:
                inp_array = np.random.rand(*shape).astype(np.float16)
            inp = constant_op.constant(list(inp_array.ravel(order='C')), shape=shape, dtype=dtype)
            lrn_op = nn.local_response_normalization(inp, name='lrn', depth_radius=lrn_depth_radius, bias=bias, alpha=alpha, beta=beta)
            err = gradient_checker.compute_gradient_error(inp, shape, lrn_op, shape)
        print('LRN Gradient error for bias ', bias, 'alpha ', alpha, ' beta ', beta, ' is ', err)
        if dtype == dtypes.float32:
            self.assertLess(err, 0.0001)
        else:
            self.assertLess(err, 1.0)

    @test_util.run_deprecated_v1
    def testGradients(self):
        if False:
            return 10
        for _ in range(2):
            self._RunAndVerifyGradients(dtypes.float32)
            if not test.is_gpu_available():
                self._RunAndVerifyGradients(dtypes.float16)
if __name__ == '__main__':
    test.main()