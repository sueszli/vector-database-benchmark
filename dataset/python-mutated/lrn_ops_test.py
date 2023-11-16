"""Tests for Local Response Normalization ops."""
import copy
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import googletest
CPU_DEVICE = '/job:localhost/replica:0/task:0/cpu:0'

class LRNTest(xla_test.XLATestCase):

    def _LRN(self, input_image, lrn_depth_radius=5, bias=1.0, alpha=1.0, beta=0.5):
        if False:
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
        with self.session():
            shape = np.random.randint(1, 16, size=4)
            shape[3] += 1
            p = array_ops.placeholder(dtype, shape=shape)
            lrn_depth_radius = np.random.randint(1, shape[3])
            bias = 1.0 + np.random.rand()
            alpha = 2.0 * np.random.rand()
            beta = 2.0 * np.random.rand()
            with self.test_scope():
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

    def testCompute(self):
        if False:
            print('Hello World!')
        for _ in range(2):
            self._RunAndVerify(dtypes.float32)

    def testLrnGrad(self):
        if False:
            for i in range(10):
                print('nop')
        shape = [1, 2, 3, 4]
        total_size = np.prod(shape)
        in_image_vals = np.arange(1, total_size + 1, dtype=np.float32)
        out_image_vals = np.arange(1, total_size + 1, dtype=np.float32)
        out_grads_vals = np.arange(1, total_size + 1, dtype=np.float32)
        depth_radius = np.random.randint(1, shape[3])
        bias = 1.0 + np.random.rand()
        alpha = 1.0 * np.random.rand()
        beta = 1.0 * np.random.rand()
        with self.session():
            in_image = constant_op.constant(in_image_vals, shape=shape)
            out_image = constant_op.constant(out_image_vals, shape=shape)
            out_grads = constant_op.constant(out_grads_vals, shape=shape)
            with ops.device(CPU_DEVICE):
                expected = gen_nn_ops.lrn_grad(out_grads, in_image, out_image, depth_radius, bias, alpha, beta)
            with self.test_scope():
                actual = gen_nn_ops.lrn_grad(out_grads, in_image, out_image, depth_radius, bias, alpha, beta)
            expected_val = self.evaluate(expected)
            actual_val = self.evaluate(actual)
        self.assertAllClose(actual_val, expected_val, rtol=0.001)
if __name__ == '__main__':
    googletest.main()