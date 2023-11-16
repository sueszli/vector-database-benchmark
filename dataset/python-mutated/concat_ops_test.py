"""Functional tests for XLA Concat Op."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python import pywrap_sanitizers
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

class ConcatTest(xla_test.XLATestCase):

    def testHStack(self):
        if False:
            return 10
        with self.session():
            p1 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
            p2 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
            with self.test_scope():
                c = array_ops.concat([p1, p2], 0)
            params = {p1: np.random.rand(4, 4).astype('f'), p2: np.random.rand(4, 4).astype('f')}
            result = c.eval(feed_dict=params)
        self.assertEqual(result.shape, c.get_shape())
        self.assertAllEqual(result[:4, :], params[p1])
        self.assertAllEqual(result[4:, :], params[p2])

    def testVStack(self):
        if False:
            return 10
        with self.session():
            p1 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
            p2 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
            with self.test_scope():
                c = array_ops.concat([p1, p2], 1)
            params = {p1: np.random.rand(4, 4).astype('f'), p2: np.random.rand(4, 4).astype('f')}
            result = c.eval(feed_dict=params)
        self.assertEqual(result.shape, c.get_shape())
        self.assertAllEqual(result[:, :4], params[p1])
        self.assertAllEqual(result[:, 4:], params[p2])

    def testInt32(self):
        if False:
            print('Hello World!')
        with self.session():
            p1 = np.random.rand(2, 3).astype('i')
            p2 = np.random.rand(2, 3).astype('i')
            x1 = constant_op.constant(p1)
            x2 = constant_op.constant(p2)
            with self.test_scope():
                c = array_ops.concat([x1, x2], 0)
            result = self.evaluate(c)
        self.assertAllEqual(result[:2, :], p1)
        self.assertAllEqual(result[2:, :], p2)

    def testAxisInt64(self):
        if False:
            print('Hello World!')
        with self.session():
            p1 = np.random.rand(2, 3).astype('i')
            p2 = np.random.rand(2, 3).astype('i')
            x1 = constant_op.constant(p1)
            x2 = constant_op.constant(p2)
            axis = constant_op.constant(0, dtype=dtypes.int64)
            with self.test_scope():
                c = array_ops.concat([x1, x2], axis)
            result = self.evaluate(c)
        self.assertAllEqual(result[:2, :], p1)
        self.assertAllEqual(result[2:, :], p2)

    def _testRandom(self, dtype):
        if False:
            return 10
        shape = np.random.randint(1, 5, size=5)
        num_tensors = np.random.randint(2, 10)
        concat_dim = np.random.randint(5)
        params = {}
        if dtype == dtypes.bfloat16:
            dtype_feed = dtypes.float32
        else:
            dtype_feed = dtype
        with self.session():
            p = []
            for i in np.arange(num_tensors):
                input_shape = shape
                input_shape[concat_dim] = np.random.randint(1, 5)
                placeholder = array_ops.placeholder(dtype_feed, shape=input_shape)
                p.append(placeholder)
                t = dtype_feed.as_numpy_dtype
                params[placeholder] = np.random.rand(*input_shape).astype(t)
            if dtype != dtype_feed:
                concat_inputs = [math_ops.cast(p_i, dtype) for p_i in p]
            else:
                concat_inputs = p
            with self.test_scope():
                c = array_ops.concat(concat_inputs, concat_dim)
                if dtype != dtype_feed:
                    c = math_ops.cast(c, dtype_feed)
            result = c.eval(feed_dict=params)
        self.assertEqual(result.shape, c.get_shape())
        cur_offset = 0
        for i in np.arange(num_tensors):
            ind = [slice(0, params[p[i]].shape[j]) for j in np.arange(5)]
            ind[concat_dim] = slice(cur_offset, cur_offset + params[p[i]].shape[concat_dim])
            cur_offset += params[p[i]].shape[concat_dim]
            if dtype == dtype_feed:
                self.assertAllEqual(result[tuple(ind)], params[p[i]])
            else:
                self.assertAllClose(result[tuple(ind)], params[p[i]], 0.01)

    def testRandom(self):
        if False:
            while True:
                i = 10
        self._testRandom(dtypes.float32)
        self._testRandom(dtypes.int32)

    def _testGradientsSimple(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session():
            inp = []
            inp_tensors = []
            with self.test_scope():
                for x in [1, 2, 6]:
                    shape = [10, x, 2]
                    t = np.random.rand(*shape).astype('f')
                    inp.append(t)
                    inp_tensors.append(constant_op.constant([float(y) for y in t.flatten()], shape=shape, dtype=dtypes.float32))
                c = array_ops.concat(inp_tensors, 1)
                output_shape = [10, 9, 2]
                grad_inp = np.random.rand(*output_shape).astype('f')
                grad_tensor = constant_op.constant([float(x) for x in grad_inp.flatten()], shape=output_shape)
                grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
                concated_grad = array_ops.concat(grad, 1)
            result = self.evaluate(concated_grad)
        self.assertAllEqual(result, grad_inp)

    def testGradientsSimpleAll(self):
        if False:
            while True:
                i = 10
        self._testGradientsSimple()

    def _testGradientsFirstDim(self):
        if False:
            print('Hello World!')
        with self.session():
            inp = []
            inp_tensors = []
            with self.test_scope():
                for x in [1, 2, 6]:
                    shape = [x, 10, 2]
                    t = np.random.rand(*shape).astype('f')
                    inp.append(t)
                    inp_tensors.append(constant_op.constant([float(y) for y in t.flatten()], shape=shape, dtype=dtypes.float32))
                c = array_ops.concat(inp_tensors, 0)
                output_shape = [9, 10, 2]
                grad_inp = np.random.rand(*output_shape).astype('f')
                grad_tensor = constant_op.constant([float(x) for x in grad_inp.flatten()], shape=output_shape)
                grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
                concated_grad = array_ops.concat(grad, 0)
                result = self.evaluate(concated_grad)
        self.assertAllEqual(result, grad_inp)

    def testGradientsFirstDimAll(self):
        if False:
            while True:
                i = 10
        self._testGradientsFirstDim()

    def _testGradientsLastDim(self):
        if False:
            print('Hello World!')
        with self.session():
            inp = []
            inp_tensors = []
            with self.test_scope():
                for x in [1, 2, 6]:
                    shape = [10, 2, x]
                    t = np.random.rand(*shape).astype('f')
                    inp.append(t)
                    inp_tensors.append(constant_op.constant([float(y) for y in t.flatten()], shape=shape, dtype=dtypes.float32))
                c = array_ops.concat(inp_tensors, 2)
                output_shape = [10, 2, 9]
                grad_inp = np.random.rand(*output_shape).astype('f')
                grad_tensor = constant_op.constant([float(x) for x in grad_inp.flatten()], shape=output_shape)
                grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
                concated_grad = array_ops.concat(grad, 2)
                result = self.evaluate(concated_grad)
        self.assertAllEqual(result, grad_inp)

    def testGradientsLastDimAll(self):
        if False:
            for i in range(10):
                print('nop')
        self._testGradientsLastDim()

    def _RunAndVerifyGradientsRandom(self):
        if False:
            i = 10
            return i + 15
        input_shape = np.random.randint(1, 5, size=5)
        num_tensors = np.random.randint(1, 10)
        concat_dim = np.random.randint(5)
        concat_dim_sizes = np.random.randint(1, 5, size=num_tensors)
        with self.session():
            inp = []
            inp_tensors = []
            with self.test_scope():
                for x in concat_dim_sizes:
                    shape = input_shape
                    shape[concat_dim] = x
                    t = np.random.rand(*shape).astype('f')
                    inp.append(t)
                    inp_tensors.append(constant_op.constant([float(y) for y in t.flatten()], shape=shape, dtype=dtypes.float32))
                c = array_ops.concat(inp_tensors, concat_dim)
                output_shape = input_shape
                output_shape[concat_dim] = concat_dim_sizes.sum()
                grad_inp = np.random.rand(*output_shape).astype('f')
                grad_tensor = constant_op.constant([float(x) for x in grad_inp.flatten()], shape=output_shape)
                grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
                concated_grad = array_ops.concat(grad, concat_dim)
                result = self.evaluate(concated_grad)
        self.assertAllEqual(result, grad_inp)

    def testGradientsRandom(self):
        if False:
            print('Hello World!')
        for _ in range(5):
            self._RunAndVerifyGradientsRandom()

    def DISABLED_testZeroSize(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(7)
        with self.session():
            with self.test_scope():
                for shape0 in ((), (2,)):
                    axis = len(shape0)
                    for shape1 in ((), (3,)):
                        for n0 in (0, 1, 2):
                            for n1 in (0, 1, 2):
                                x0 = np.random.randn(*shape0 + (n0,) + shape1)
                                x1 = np.random.randn(*shape0 + (n1,) + shape1)
                                correct = np.concatenate([x0, x1], axis=axis)
                                xs = list(map(constant_op.constant, [x0, x1]))
                                c = array_ops.concat(xs, axis)
                                self.assertAllEqual(c, correct)
                                dc = np.random.randn(*c.get_shape().as_list())
                                dxs = self.evaluate(gradients_impl.gradients(c, xs, dc))
                                self.assertAllEqual(dc, np.concatenate(dxs, axis=axis))

    def testConcatTuple(self):
        if False:
            i = 10
            return i + 15
        c1 = np.random.rand(4, 4).astype(np.float32)
        c2 = np.random.rand(4, 4).astype(np.float32)
        with self.session():
            with self.test_scope():
                concat_list_t = array_ops.concat([c1, c2], 0)
                concat_tuple_t = array_ops.concat((c1, c2), 0)
            self.assertAllEqual(concat_list_t, self.evaluate(concat_tuple_t))

    def testConcatNoScalars(self):
        if False:
            return 10
        with self.session():
            with self.test_scope():
                scalar = constant_op.constant(7)
                dim = array_ops.placeholder(dtypes.int32)
                with self.assertRaisesRegex(ValueError, "Can't concatenate scalars \\(use tf\\.stack instead\\)"):
                    array_ops.concat([scalar, scalar, scalar], dim)

    def testConcatLargeNumberOfTensors(self):
        if False:
            return 10
        if 'CPU' in self.device:
            self.skipTest('This test can time out on CPU, so we will just allow other backends to catch this specific error.')
        if pywrap_sanitizers.is_asan_enabled() or pywrap_sanitizers.is_tsan_enabled() or pywrap_sanitizers.is_msan_enabled() or pywrap_sanitizers.is_ubsan_enabled():
            self.skipTest('This test can time out on *SAN.')
        with self.session():
            with self.test_scope():
                for concat_dim in range(2):
                    params = {}
                    p = []
                    shape = np.array([7, 13])
                    num_tensors = 1001
                    for i in np.arange(num_tensors):
                        input_shape = shape
                        placeholder = array_ops.placeholder(dtypes.float32, shape=input_shape)
                        p.append(placeholder)
                        params[placeholder] = np.random.rand(*input_shape).astype(np.float32)
                    concat_inputs = p
                    c = array_ops.concat(concat_inputs, concat_dim)
                    result = c.eval(feed_dict=params)
                    self.assertEqual(result.shape, c.get_shape())
                    cur_offset = 0
                    for i in np.arange(num_tensors):
                        index = [slice(0, params[p[i]].shape[j]) for j in np.arange(2)]
                        index[concat_dim] = slice(cur_offset, cur_offset + params[p[i]].shape[concat_dim])
                        cur_offset += params[p[i]].shape[concat_dim]
                        self.assertAllEqual(result[tuple(index)], params[p[i]])

class ConcatOffsetTest(xla_test.XLATestCase):

    def testBasic(self):
        if False:
            print('Hello World!')
        with self.session():
            with self.test_scope():
                cdim = constant_op.constant(1, dtypes.int32)
                s0 = constant_op.constant([2, 3, 5], dtypes.int32)
                s1 = constant_op.constant([2, 7, 5], dtypes.int32)
                s2 = constant_op.constant([2, 20, 5], dtypes.int32)
                off = gen_array_ops.concat_offset(cdim, [s0, s1, s2])
                ans = self.evaluate(off)
                self.assertAllEqual(ans, [[0, 0, 0], [0, 3, 0], [0, 10, 0]])

class PackTest(xla_test.XLATestCase):

    def testBasic(self):
        if False:
            while True:
                i = 10
        with self.session():
            with self.test_scope():
                s0 = constant_op.constant([2, 3, 5], dtypes.int32)
                s1 = constant_op.constant([2, 7, 5], dtypes.int32)
                s2 = constant_op.constant([2, 20, 5], dtypes.int32)
                packed = array_ops_stack.stack([s0, s1, s2])
                ans = self.evaluate(packed)
                self.assertAllEqual(ans, [[2, 3, 5], [2, 7, 5], [2, 20, 5]])

    def testScalars(self):
        if False:
            print('Hello World!')
        with self.session():
            with self.test_scope():
                s0 = constant_op.constant(2, dtypes.int32)
                s1 = constant_op.constant(3, dtypes.int32)
                s2 = constant_op.constant(5, dtypes.int32)
                packed = array_ops_stack.stack([s0, s1, s2])
                ans = self.evaluate(packed)
                self.assertAllEqual(ans, [2, 3, 5])

    def testEmpty(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session():
            with self.test_scope():
                s0 = constant_op.constant([[]], dtypes.int32)
                s1 = constant_op.constant([[]], dtypes.int32)
                s2 = constant_op.constant([[]], dtypes.int32)
                packed = array_ops_stack.stack([s0, s1, s2])
                ans = self.evaluate(packed)
                self.assertAllEqual(ans, [[[]], [[]], [[]]])
if __name__ == '__main__':
    googletest.main()