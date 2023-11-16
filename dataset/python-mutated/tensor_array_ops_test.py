"""Functional tests for XLA TensorArray Ops."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_grad
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

def _make_converter(dtype):
    if False:
        print('Hello World!')

    def _converter(x):
        if False:
            for i in range(10):
                print('nop')
        return np.asarray(x).astype(dtype.as_numpy_dtype)
    return _converter

@test_util.run_v1_only('b/')
@test_util.with_control_flow_v2
class TensorArrayTest(xla_test.XLATestCase):

    @test_util.disable_control_flow_v2('Tries to evaluate flow')
    def testTensorArrayWriteRead(self):
        if False:
            return 10
        with self.session() as session, self.test_scope():

            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=3)
                w0 = ta.write(0, [[4.0, 5.0]])
                w1 = w0.write(1, [[1.0, 3.0]])
                w2 = w1.write(2, [[7.0, -8.5]])
                r0 = w2.read(0)
                r1 = w2.read(1)
                r2 = w2.read(2)
                flow = w2.flow
                return [r0, r1, r2, flow]
            (d0, d1, d2, flow_val) = self.evaluate(xla.compile(fn))
            self.assertAllEqual([[4.0, 5.0]], d0)
            self.assertAllEqual([[1.0, 3.0]], d1)
            self.assertAllEqual([[7.0, -8.5]], d2)
            self.assertAllEqual([], flow_val.shape)

    def _testTensorArrayWritePack(self, tf_dtype):
        if False:
            while True:
                i = 10
        with self.session(), self.test_scope():
            convert = _make_converter(tf_dtype)

            def fn():
                if False:
                    while True:
                        i = 10
                ta = tensor_array_ops.TensorArray(dtype=tf_dtype, tensor_array_name='foo', size=3)
                w0 = ta.write(0, convert([[4.0, 5.0]]))
                w1 = w0.write(1, convert([[6.0, 7.0]]))
                w2 = w1.write(2, convert([[8.0, 9.0]]))
                return w2.stack()
            self.assertAllEqual(convert([[[4.0, 5.0]], [[6.0, 7.0]], [[8.0, 9.0]]]), self.evaluate(xla.compile(fn)[0]))

    def testTensorArrayWritePack(self):
        if False:
            while True:
                i = 10
        for dtype in self.numeric_tf_types:
            self._testTensorArrayWritePack(dtype)

    def testEmptyTensorArrayPack(self):
        if False:
            return 10
        with self.session(), self.test_scope():

            def fn():
                if False:
                    while True:
                        i = 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=3)
                empty_element = np.zeros((0, 1), dtype=np.float32)
                w0 = ta.write(0, empty_element)
                w1 = w0.write(1, empty_element)
                w2 = w1.write(2, empty_element)
                return w2.stack()
            self.assertAllEqual([3, 0, 1], self.evaluate(xla.compile(fn)[0]).shape)

    def _testTensorArrayWriteConcat(self, tf_dtype):
        if False:
            print('Hello World!')
        with self.session(), self.test_scope():
            convert = _make_converter(tf_dtype)

            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                ta = tensor_array_ops.TensorArray(dtype=tf_dtype, tensor_array_name='foo', size=3)
                w0 = ta.write(0, convert([[4.0, 5.0], [104.0, 105.0]]))
                w1 = w0.write(1, convert([[6.0, 7.0], [106.0, 107.0]]))
                w2 = w1.write(2, convert([[8.0, 9.0], [124.0, 125.0]]))
                return w2.concat()
            self.assertAllEqual(convert([[4.0, 5.0], [104.0, 105.0], [6.0, 7.0], [106.0, 107.0], [8.0, 9.0], [124.0, 125.0]]), self.evaluate(xla.compile(fn)[0]))

    @test_util.disable_control_flow_v2('b/122315751 (concat)')
    def testTensorArrayWriteConcat(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self.numeric_tf_types:
            self._testTensorArrayWriteConcat(dtype)

    def _testTensorArrayUnpackRead(self, tf_dtype):
        if False:
            i = 10
            return i + 15
        with self.session() as session, self.test_scope():
            convert = _make_converter(tf_dtype)

            def fn():
                if False:
                    return 10
                ta = tensor_array_ops.TensorArray(dtype=tf_dtype, tensor_array_name='foo', size=3)
                w0 = ta.unstack(convert([1.0, 2.0, 3.0]))
                r0 = w0.read(0)
                r1 = w0.read(1)
                r2 = w0.read(2)
                return [r0, r1, r2]
            (d0, d1, d2) = self.evaluate(xla.compile(fn))
            self.assertAllEqual(convert(1.0), d0)
            self.assertAllEqual(convert(2.0), d1)
            self.assertAllEqual(convert(3.0), d2)

            def fn():
                if False:
                    i = 10
                    return i + 15
                ta = tensor_array_ops.TensorArray(dtype=tf_dtype, tensor_array_name='foo', size=3)
                w1 = ta.unstack(convert([[1.0, 1.03125], [2.0, 2.03125], [3.0, 3.03125]]))
                r0 = w1.read(0)
                r1 = w1.read(1)
                r2 = w1.read(2)
                return [r0, r1, r2]
            (d0, d1, d2) = self.evaluate(xla.compile(fn))
            self.assertAllEqual(convert([1.0, 1.03125]), d0)
            self.assertAllEqual(convert([2.0, 2.03125]), d1)
            self.assertAllEqual(convert([3.0, 3.03125]), d2)

            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                ta = tensor_array_ops.TensorArray(dtype=tf_dtype, tensor_array_name='foo', size=3)
                w2 = ta.unstack(convert([[], [], []]))
                r0 = w2.read(0)
                r1 = w2.read(1)
                r2 = w2.read(2)
                return [r0, r1, r2]
            (d0, d1, d2) = self.evaluate(xla.compile(fn))
            self.assertAllEqual(convert([]), d0)
            self.assertAllEqual(convert([]), d1)
            self.assertAllEqual(convert([]), d2)

    def _testTensorArrayUnpackReadMaybeLegacy(self):
        if False:
            return 10
        for dtype in self.numeric_tf_types:
            self._testTensorArrayUnpackRead(dtype)

    def testTensorArrayUnpackRead(self):
        if False:
            for i in range(10):
                print('nop')
        self._testTensorArrayUnpackReadMaybeLegacy()

    def _testTensorArraySplitRead(self, tf_dtype):
        if False:
            print('Hello World!')
        with self.session() as session, self.test_scope():
            convert = _make_converter(tf_dtype)

            def fn():
                if False:
                    return 10
                ta = tensor_array_ops.TensorArray(dtype=tf_dtype, tensor_array_name='foo', size=3)
                lengths = constant_op.constant([0, 0, 0])
                w0 = ta.split(convert([]), lengths=lengths)
                r0 = w0.read(0)
                r1 = w0.read(1)
                r2 = w0.read(2)
                return [r0, r1, r2]
            (d0, d1, d2) = self.evaluate(xla.compile(fn))
            self.assertAllEqual(convert([]), d0)
            self.assertAllEqual(convert([]), d1)
            self.assertAllEqual(convert([]), d2)

            def fn():
                if False:
                    print('Hello World!')
                ta = tensor_array_ops.TensorArray(dtype=tf_dtype, tensor_array_name='foo', size=3)
                lengths = constant_op.constant([1, 1, 1])
                w0 = ta.split(convert([1.0, 2.0, 3.0]), lengths=lengths)
                r0 = w0.read(0)
                r1 = w0.read(1)
                r2 = w0.read(2)
                return [r0, r1, r2]
            (d0, d1, d2) = self.evaluate(xla.compile(fn))
            self.assertAllEqual(convert([1.0]), d0)
            self.assertAllEqual(convert([2.0]), d1)
            self.assertAllEqual(convert([3.0]), d2)

            def fn():
                if False:
                    while True:
                        i = 10
                ta = tensor_array_ops.TensorArray(dtype=tf_dtype, tensor_array_name='foo', size=3)
                lengths = constant_op.constant([1, 1, 1])
                w0 = ta.split(convert([[1.0, 101.0], [2.0, 121.0], [3.0, 127.0]]), lengths=lengths)
                r0 = w0.read(0)
                r1 = w0.read(1)
                r2 = w0.read(2)
                return [r0, r1, r2]
            (d0, d1, d2) = self.evaluate(xla.compile(fn))
            self.assertAllEqual(convert([[1.0, 101.0]]), d0)
            self.assertAllEqual(convert([[2.0, 121.0]]), d1)
            self.assertAllEqual(convert([[3.0, 127.0]]), d2)

    @test_util.disable_control_flow_v2('b/122315872 (split)')
    def testTensorArraySplitRead(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self.numeric_tf_types:
            self._testTensorArraySplitRead(dtype)

    @test_util.disable_control_flow_v2('TensorArray.grad is not supported in v2')
    def testTensorGradArrayWriteRead(self):
        if False:
            return 10
        with self.session() as session, self.test_scope():

            def fn():
                if False:
                    while True:
                        i = 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=3)
                w0 = ta.write(0, [[4.0]])
                w1 = w0.write(1, [[1.0]])
                w2 = w1.write(2, [[-3.0]])
                g_ta = w2.grad('grad')
                g_w0 = g_ta.write(0, [[5.0]])
                g_w1 = g_w0.write(1, [[2.0]])
                g_w2 = g_w1.write(2, [[-2.0]])
                r0 = w2.read(0)
                r1 = w2.read(1)
                r2 = w2.read(2)
                g_r0 = g_w2.read(0)
                g_r1 = g_w2.read(1)
                g_r2 = g_w2.read(2)
                return [r0, r1, r2, g_r0, g_r1, g_r2]
            (d0, d1, d2, g_d0, g_d1, g_d2) = self.evaluate(xla.compile(fn))
            self.assertAllEqual([[4.0]], d0)
            self.assertAllEqual([[1.0]], d1)
            self.assertAllEqual([[-3.0]], d2)
            self.assertAllEqual([[5.0]], g_d0)
            self.assertAllEqual([[2.0]], g_d1)
            self.assertAllEqual([[-2.0]], g_d2)

    @test_util.disable_control_flow_v2('TensorArray.grad is not supported in v2')
    def testTensorGradArrayDynamicWriteRead(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session() as session, self.test_scope():

            def fn():
                if False:
                    print('Hello World!')
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=3)
                w0 = ta.write(0, [[4.0]])
                w1 = w0.write(1, [[1.0]])
                w2 = w1.write(2, [[-3.0]])
                g_ta = w2.grad('grad')
                s = w2.size()
                g_s = g_ta.size()
                g_w0 = g_ta.write(0, [[5.0]])
                g_w1 = g_w0.write(1, [[2.0]])
                g_w2 = g_w1.write(2, [[-2.0]])
                r0 = w2.read(0)
                r1 = w2.read(1)
                r2 = w2.read(2)
                g_r0 = g_w2.read(0)
                g_r1 = g_w2.read(1)
                g_r2 = g_w2.read(2)
                return [r0, r1, r2, g_r0, g_r1, g_r2, s, g_s]
            (d0, d1, d2, g_d0, g_d1, g_d2, vs, g_vs) = self.evaluate(xla.compile(fn))
            self.assertAllEqual([[4.0]], d0)
            self.assertAllEqual([[1.0]], d1)
            self.assertAllEqual([[-3.0]], d2)
            self.assertAllEqual([[5.0]], g_d0)
            self.assertAllEqual([[2.0]], g_d1)
            self.assertAllEqual([[-2.0]], g_d2)
            self.assertAllEqual(3, vs)
            self.assertAllEqual(3, g_vs)

    @test_util.disable_control_flow_v2('TensorArray.grad is not supported in v2')
    def testTensorGradAccessTwiceReceiveSameObject(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session() as session, self.test_scope():
            ta_out = {}

            def fn():
                if False:
                    while True:
                        i = 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=3, element_shape=[1, 2])
                g_ta_0 = ta.grad('grad')
                g_ta_1 = ta.grad('grad')
                ta_out[0] = g_ta_0.handle
                ta_out[1] = g_ta_1.handle
                with ops.control_dependencies([g_ta_0.write(0, [[4.0, 5.0]]).flow]):
                    r1_0 = g_ta_1.read(0)
                with ops.control_dependencies([g_ta_0.handle.op, g_ta_1.handle.op]):
                    return [r1_0]
            [d_r1_0] = self.evaluate(xla.compile(fn))
            self.assertAllEqual([[4.0, 5.0]], d_r1_0)

    @test_util.disable_control_flow_v2('b/124334470')
    def testTensorArrayWriteWrongIndexOrDataTypeFails(self):
        if False:
            while True:
                i = 10
        with self.session(), self.test_scope():

            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=3)
                return ta.write(-1, constant_op.constant(7)).flow
            with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), '(conversion requested dtype float32 for Tensor with dtype int32|TensorArray dtype is float but op has dtype int32)'):
                xla.compile(fn)[0].eval()

    @test_util.disable_control_flow_v2('b/124334096 verify dtype')
    def testTensorArrayReadWrongIndexOrDataTypeFails(self):
        if False:
            i = 10
            return i + 15
        if len(self.float_types) > 1:
            (dtype1, dtype2) = list(self.float_types)[:2]
            with self.session(), self.test_scope():

                def fn():
                    if False:
                        print('Hello World!')
                    ta = tensor_array_ops.TensorArray(dtype=dtype1, tensor_array_name='foo', size=3)
                    w0 = ta.write(0, math_ops.cast([[4.0, 5.0]], dtype1))
                    return gen_data_flow_ops.tensor_array_read_v3(handle=w0.handle, index=0, dtype=dtype2, flow_in=w0.flow)
                with self.assertRaisesOpError('TensorArray dtype is '):
                    self.evaluate(xla.compile(fn))

                def fn():
                    if False:
                        print('Hello World!')
                    ta = tensor_array_ops.TensorArray(dtype=dtype1, tensor_array_name='foo', size=3)
                    w0 = ta.write(0, math_ops.cast([[4.0, 5.0]], dtype1))
                    with ops.control_dependencies([w0.read(1)]):
                        return 1.0
                xla.compile(fn)[0].eval()

    @test_util.disable_control_flow_v2('b/122315872 (split)')
    def testTensorArraySplitIncompatibleShapesFails(self):
        if False:
            return 10
        with self.session(), self.test_scope():

            def fn():
                if False:
                    return 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=3, infer_shape=False)
                return ta.split([1.0, 2.0, 3.0], 1).flow
            with self.assertRaisesWithPredicateMatch(ValueError, 'Shape must be rank 1 but is rank 0'):
                xla.compile(fn)[0].eval()

            def fn():
                if False:
                    i = 10
                    return i + 15
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=3, infer_shape=False)
                return ta.split([1.0, 2.0, 3.0], [1, 2, 3]).flow
            with self.assertRaisesOpError('lengths must be equal: 1 vs. 2'):
                xla.compile(fn)[0].eval()

            def fn():
                if False:
                    while True:
                        i = 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=3, infer_shape=False)
                return ta.split(1.0, [1]).flow
            with self.assertRaisesOpError('value must have rank >= 1'):
                xla.compile(fn)[0].eval()

            def fn():
                if False:
                    print('Hello World!')
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=2, infer_shape=False)
                return ta.split([1.0], [1]).flow
            with self.assertRaisesOpError("TensorArray's size is not equal to the size of lengths \\(1 vs. 2\\)"):
                xla.compile(fn)[0].eval()

    def _testTensorArrayWriteGradientAddMultipleAdds(self, dtype):
        if False:
            i = 10
            return i + 15
        with self.session(), self.test_scope():
            c = lambda x: np.asarray(x, dtype=dtype.as_numpy_dtype)

            def fn():
                if False:
                    while True:
                        i = 10
                ta = tensor_array_ops.TensorArray(dtype=dtype, tensor_array_name='foo', size=3, infer_shape=False)
                w0 = ta.write(2, c(3.0))
                w1 = w0.write(2, c(4.0))
                ta_grad = w1.grad('grad')
                w0_grad = ta_grad.write(2, c(3.0))
                w1_grad = w0_grad.write(2, c(4.0))
                w2_grad = w1_grad.write(2, c(5.0))
                return w2_grad.read(2)
            self.assertAllEqual(c(12.0), xla.compile(fn)[0])

            def fn():
                if False:
                    i = 10
                    return i + 15
                ta = tensor_array_ops.TensorArray(dtype=dtype, tensor_array_name='foo', size=3, infer_shape=False)
                w0 = ta.write(2, c(3.0))
                w1 = w0.write(2, c(4.0))
                ta_grad = w1.grad('grad')
                wb0_grad = ta_grad.write(1, c(1.0))
                wb1_grad = wb0_grad.write(1, c([1.0]))
                return wb1_grad.flow
            with self.assertRaisesOpError('Mismatched TensorArray sizes'):
                xla.compile(fn)[0].eval()

    @test_util.disable_control_flow_v2('TensorArray.grad is not supported in v2')
    def testTensorArrayWriteGradientAddMultipleAdds(self):
        if False:
            print('Hello World!')
        for dtype in self.numeric_tf_types:
            self._testTensorArrayWriteGradientAddMultipleAdds(dtype)

    def testMultiTensorArray(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session(), self.test_scope():

            def fn():
                if False:
                    while True:
                        i = 10
                h1 = tensor_array_ops.TensorArray(size=1, dtype=dtypes.float32, tensor_array_name='foo')
                w1 = h1.write(0, 4.0)
                r1 = w1.read(0)
                h2 = tensor_array_ops.TensorArray(size=1, dtype=dtypes.float32, tensor_array_name='bar')
                w2 = h2.write(0, 5.0)
                r2 = w2.read(0)
                return r1 + r2
            self.assertAllClose(9.0, self.evaluate(xla.compile(fn)[0]))

    def _testTensorArrayGradientWriteReadType(self, dtype):
        if False:
            i = 10
            return i + 15
        with self.session() as session, self.test_scope():
            c = lambda x: np.array(x, dtype=dtype)

            def fn():
                if False:
                    print('Hello World!')
                ta = tensor_array_ops.TensorArray(dtype=dtypes.as_dtype(dtype), tensor_array_name='foo', size=3, infer_shape=False)
                value_0 = constant_op.constant(c([[4.0, 5.0]]))
                value_1 = constant_op.constant(c([[3.0, 3.5]]))
                w0 = ta.write(0, value_0)
                w1 = w0.write(1, value_1)
                r0 = w1.read(0)
                r1 = w1.read(1)
                r0_2 = w1.read(0)
                grad_just_r0 = gradients_impl.gradients(ys=[r0], xs=[value_0], grad_ys=[c([[2.0, 3.0]])])
                grad_r0_r0_2 = gradients_impl.gradients(ys=[r0, r0_2], xs=[value_0], grad_ys=[c([[2.0, 3.0]]), c([[1.0, -1.0]])])
                grad_just_r1 = gradients_impl.gradients(ys=[r1], xs=[value_1], grad_ys=[c([[-2.0, -4.0]])])
                grad = gradients_impl.gradients(ys=[r0, r0_2, r1], xs=[value_0, value_1], grad_ys=[c([[2.0, 3.0]]), c([[1.0, -1.0]]), c([[-2.0, -10.0]])])
                return [grad_just_r0, grad_r0_r0_2, grad_just_r1, grad]
            [grad_just_r0_vals, grad_r0_r0_2_vals, grad_just_r1_vals, grad_vals] = self.evaluate(xla.compile(fn))
            self.assertAllEqual(c([[2.0, 3.0]]), grad_just_r0_vals[0])
            self.assertAllEqual(c([[3.0, 2.0]]), grad_r0_r0_2_vals[0])
            self.assertAllEqual(c([[-2.0, -4.0]]), grad_just_r1_vals[0])
            self.assertEqual(len(grad_vals), 2)
            self.assertAllEqual(c([[3.0, 2.0]]), grad_vals[0])
            self.assertAllEqual(c([[-2.0, -10.0]]), grad_vals[1])

    def testTensorArrayGradientWriteRead(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.float_types:
            self._testTensorArrayGradientWriteReadType(dtype)
        for dtype in self.complex_types:
            self._testTensorArrayGradientWriteReadType(dtype)

    def _testTensorArrayGradientWritePackConcatAndRead(self):
        if False:
            i = 10
            return i + 15
        with self.session() as sess, self.test_scope():

            def fn():
                if False:
                    while True:
                        i = 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=2, clear_after_read=False)
                value_0 = constant_op.constant([-1.0, 1.0])
                value_1 = constant_op.constant([-10.0, 10.0])
                w0 = ta.write(0, value_0)
                w1 = w0.write(1, value_1)
                p0 = w1.stack()
                r0 = w1.read(0)
                s0 = w1.concat()
                with ops.control_dependencies([p0, r0, s0]):
                    return gradients_impl.gradients(ys=[p0, r0, s0], xs=[value_0, value_1], grad_ys=[[[2.0, 3.0], [4.0, 5.0]], [-0.5, 1.5], [20.0, 30.0, 40.0, 50.0]])
            grad_vals = self.evaluate(xla.compile(fn))
            self.assertAllClose([2.0 - 0.5 + 20.0, 3.0 + 1.5 + 30.0], grad_vals[0])
            self.assertAllEqual([4.0 + 40.0, 5.0 + 50.0], grad_vals[1])

    @test_util.disable_control_flow_v2('b/122315751 (concat)')
    def testTensorArrayGradientWritePackConcatAndRead(self):
        if False:
            return 10
        self._testTensorArrayGradientWritePackConcatAndRead()

    def testTensorArrayReadTwice(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session(), self.test_scope():

            def fn():
                if False:
                    print('Hello World!')
                value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])
                ta_readtwice = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=2, clear_after_read=False)
                w_readtwice = ta_readtwice.unstack(value)
                r0_readtwice = w_readtwice.read(0)
                with ops.control_dependencies([r0_readtwice]):
                    r1_readtwice = w_readtwice.read(0)
                return [r0_readtwice, r1_readtwice]
            self.assertAllEqual([1.0, -1.0], self.evaluate(xla.compile(fn))[0])

    def _testTensorArrayGradientUnpackRead(self):
        if False:
            print('Hello World!')
        with self.session() as session, self.test_scope():

            def fn():
                if False:
                    return 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=2, clear_after_read=False)
                value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])
                w = ta.unstack(value)
                r0 = w.read(0)
                r0_1 = w.read(0)
                r1 = w.read(1)
                return gradients_impl.gradients(ys=[r0, r0_1, r1], xs=[value], grad_ys=[[2.0, 3.0], [-1.5, 1.5], [4.0, 5.0]])
            grad_vals = self.evaluate(xla.compile(fn))
            self.assertEqual(len(grad_vals), 1)
            self.assertAllEqual([[2.0 - 1.5, 3.0 + 1.5], [4.0, 5.0]], grad_vals[0])

    def testTensorArrayGradientUnpackRead(self):
        if False:
            while True:
                i = 10
        self._testTensorArrayGradientUnpackRead()

    @test_util.disable_control_flow_v2('b/122315751(concat), b/122315872(split)')
    def testTensorArrayGradientSplitConcat(self):
        if False:
            return 10
        with self.session() as session, self.test_scope():

            def fn():
                if False:
                    print('Hello World!')
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=2)
                value = constant_op.constant([[1.0, -1.0], [10.0, -10.0], [100.0, -100.0], [1000.0, -1000.0]])
                w = ta.split(value, [2, 2])
                r = w.concat()
                return gradients_impl.gradients(ys=[r], xs=[value], grad_ys=[[[2.0, -2.0], [20.0, -20.0], [200.0, -200.0], [2000.0, -2000.0]]])
            grad_vals = self.evaluate(xla.compile(fn))
            self.assertEqual(len(grad_vals), 1)
            self.assertAllEqual([[2.0, -2.0], [20.0, -20.0], [200.0, -200.0], [2000.0, -2000.0]], grad_vals[0])

    def testCloseTensorArray(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session() as session, self.test_scope():

            def fn():
                if False:
                    return 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=3)
                with ops.control_dependencies([ta.close()]):
                    return 1.0
            self.evaluate(xla.compile(fn)[0])

    def testSizeTensorArray(self):
        if False:
            return 10
        with self.session(), self.test_scope():

            def fn():
                if False:
                    while True:
                        i = 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=3)
                return ta.size()
            self.assertAllEqual(3, self.evaluate(xla.compile(fn))[0])

    def testWriteCloseTensorArray(self):
        if False:
            i = 10
            return i + 15
        with self.session(), self.test_scope():

            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=3, infer_shape=False)
                w0 = ta.write(0, [[4.0, 5.0]])
                w1 = w0.write(1, [[3.0, 1.0]])
                with ops.control_dependencies([w1.close()]):
                    return 1.0
            self.evaluate(xla.compile(fn))

    def testSumOfTwoReadVariablesWithoutRepeatGrad(self):
        if False:
            print('Hello World!')
        with self.session() as session, self.test_scope():
            g0 = -(np.arange(3 * 5, dtype=np.float32).reshape(3, 5) + 1)

            def fn():
                if False:
                    print('Hello World!')
                a = array_ops.identity(np.arange(3 * 5, dtype=np.float32).reshape(3, 5) + 1)
                b = array_ops.identity(np.arange(3 * 5, dtype=np.float32).reshape(3, 5) + 1 + 3 * 5)
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
                ta = ta.write(0, a, name='write_a')
                ta = ta.write(1, b, name='write_b')
                c = ta.read(0, name='read_a_0') + ta.read(1, name='read_b_0')
                grad_a = gradients_impl.gradients([c], [a], [g0])[0]
                grad_b = gradients_impl.gradients([c], [b], [g0])[0]
                return [grad_a, grad_b]
            (grad_a, grad_b) = xla.compile(fn)
            (grad_a_t,) = self.evaluate([grad_a])
            self.assertAllEqual(grad_a_t, g0)
            (grad_b_t,) = self.evaluate([grad_b])
            self.assertAllEqual(grad_b_t, g0)
            (joint_grad_a_t, joint_grad_b_t) = self.evaluate([grad_a, grad_b])
            self.assertAllEqual(joint_grad_a_t, g0)
            self.assertAllEqual(joint_grad_b_t, g0)

    def testWriteShape(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session(), self.test_scope():

            def fn():
                if False:
                    return 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=3)
                c0 = constant_op.constant([4.0, 5.0])
                w0 = ta.write(0, c0)
                r0 = w0.read(0)
                return [c0, r0]
            (c0, r0) = xla.compile(fn)
            self.assertAllEqual(c0.get_shape(), r0.get_shape())

            def fn():
                if False:
                    return 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=3)
                c1 = constant_op.constant([6.0, 7.0])
                w0 = ta.write(0, c0)
                w1 = w0.write(1, c1)
                r0 = w1.read(0)
                r1 = w1.read(1)
                return [r0, c1, r1]
            [r0, c1, r1] = xla.compile(fn)
            self.assertAllEqual(c0.get_shape(), r0.get_shape())
            self.assertAllEqual(c1.get_shape(), r1.get_shape())

            def fn():
                if False:
                    return 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=3)
                w0 = ta.write(0, c0)
                c2 = constant_op.constant([4.0, 5.0, 6.0])
                return w0.write(0, c2).flow
            with self.assertRaises(ValueError):
                self.evaluate(xla.compile(fn))

    def _testGradientWhenNotAllComponentsRead(self):
        if False:
            return 10
        with self.session() as session, self.test_scope():

            def fn():
                if False:
                    while True:
                        i = 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
                x = constant_op.constant([2.0, 3.0])
                w = ta.unstack(x)
                r0 = w.read(0)
                return gradients_impl.gradients(ys=[r0], xs=[x], grad_ys=[1.0])
            grad_r0_vals = self.evaluate(xla.compile(fn))[0]
            self.assertAllEqual(grad_r0_vals, [1.0, 0.0])

    def testGradientWhenNotAllComponentsRead(self):
        if False:
            i = 10
            return i + 15
        self._testGradientWhenNotAllComponentsRead()

    def _testTensorArrayEvalEmpty(self):
        if False:
            while True:
                i = 10
        with self.session(), self.test_scope():

            def fn():
                if False:
                    i = 10
                    return i + 15
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=0, infer_shape=False)
                return ta.stack()
            with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, 'Uninitialized TensorArray passed to TensorArrayStack/TensorArrayGatherV3'):
                xla.compile(fn)[0].eval()

    @test_util.disable_control_flow_v2('b/124335246')
    def testTensorArrayEvalEmpty(self):
        if False:
            return 10
        self._testTensorArrayEvalEmpty()

    def _testTensorArrayEvalEmptyWithDefault(self):
        if False:
            return 10
        with self.session(), self.test_scope():

            def fn():
                if False:
                    while True:
                        i = 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=0, infer_shape=True)
                size = ta.size()
                ta = ta.unstack(array_ops.zeros([0, 3, 5]))
                return [size, ta.stack()]
            [size, stack] = self.evaluate(xla.compile(fn))
            self.assertEqual(0, size)
            self.assertAllEqual([0, 3, 5], stack.shape)
            if not control_flow_util.ENABLE_CONTROL_FLOW_V2:

                def fn():
                    if False:
                        for i in range(10):
                            print('nop')
                    ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=0, infer_shape=True)
                    ta = ta.unstack(array_ops.zeros([0, 3, 5]))
                    return ta.concat()
                self.assertAllEqual([0, 5], self.evaluate(xla.compile(fn))[0].shape)

    def testTensorArrayEvalEmptyWithDefault(self):
        if False:
            while True:
                i = 10
        self._testTensorArrayEvalEmptyWithDefault()

    def _testTensorArrayScatterRead(self, tf_dtype):
        if False:
            print('Hello World!')
        with self.session() as session, self.test_scope():
            convert = _make_converter(tf_dtype)
            id0 = array_ops.placeholder(dtypes.int32)
            id1 = array_ops.placeholder(dtypes.int32)

            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                ta = tensor_array_ops.TensorArray(dtype=tf_dtype, tensor_array_name='foo', size=10)
                indices = constant_op.constant([1, 8])
                value = constant_op.constant(convert([[1.0, 5.0], [10.0, 20.0]]))
                w = ta.scatter(indices, value)
                r0 = w.read(id0)
                r1 = w.read(id1)
                return [r0, r1]
            read_vals = session.run(xla.compile(fn), feed_dict={id0: 1, id1: 8})
            self.assertAllEqual(convert([1.0, 5.0]), read_vals[0])
            self.assertAllEqual(convert([10.0, 20.0]), read_vals[1])

    @test_util.disable_control_flow_v2('b/122315734 (scatter)')
    def testTensorArrayScatterRead(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.numeric_tf_types:
            self._testTensorArrayScatterRead(dtype)
        self._testTensorArrayScatterRead(dtypes.bool)

    @test_util.disable_control_flow_v2('b/122315734 (scatter)')
    def testTensorArrayScatterReadAndGradients(self):
        if False:
            while True:
                i = 10
        with self.session() as session, self.test_scope():
            id0 = array_ops.placeholder(dtypes.int32)
            id1 = array_ops.placeholder(dtypes.int32)

            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=10)
                indices = constant_op.constant([1, 8])
                value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])
                w = ta.scatter(indices, value)
                r0 = w.read(id0)
                r1 = w.read(id1)
                grad = gradients_impl.gradients(ys=[r0, r1], xs=[value], grad_ys=[[2.0, 3.0], [4.0, 5.0]])
                return [[r0, r1], grad]
            (read_vals, grad_vals) = session.run(xla.compile(fn), feed_dict={id0: 1, id1: 8})
            self.assertEqual(len(read_vals), 2)
            self.assertEqual(len(grad_vals), 1)
            self.assertAllEqual([1.0, -1.0], read_vals[0])
            self.assertAllEqual([10.0, -10.0], read_vals[1])
            self.assertAllEqual([[2.0, 3.0], [4.0, 5.0]], grad_vals[0])

    @test_util.disable_control_flow_v2('b/122315378 (gather)')
    def testTensorArrayWriteGatherAndGradients(self):
        if False:
            return 10
        with self.session() as session, self.test_scope():

            def fn():
                if False:
                    while True:
                        i = 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='foo', size=10)
                values = constant_op.constant([[1.0 * x, -1.0 * x] for x in range(10)])
                indices = constant_op.constant([1, 8])
                w = ta.unstack(values)
                g = w.gather(indices)
                grad = gradients_impl.gradients(ys=[g], xs=[values], grad_ys=[[[2.0, 3.0], [4.0, 5.0]]])
                return [[g], grad]
            (g_vals, grad_vals) = self.evaluate(xla.compile(fn))
            expected_grad = np.zeros((10, 2))
            expected_grad[1] = [2.0, 3.0]
            expected_grad[8] = [4.0, 5.0]
            self.assertEqual(len(g_vals), 1)
            self.assertEqual(len(grad_vals), 1)
            self.assertAllEqual([[1.0, -1.0], [8.0, -8.0]], g_vals[0])
            self.assertAllEqual(expected_grad, grad_vals[0])

    def testTensorArrayIdentity(self):
        if False:
            while True:
                i = 10
        with self.session() as session, self.test_scope():
            tensor_arrays = {}
            v0 = resource_variable_ops.ResourceVariable(0.0)
            v1 = resource_variable_ops.ResourceVariable(0.0)

            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                ta0 = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2, infer_shape=False)
                ta1 = tensor_array_ops.TensorArray(dtype=dtypes.int32, size=4, infer_shape=True)
                ta0 = ta0.write(0, 0.0)
                ta1 = ta1.write(0, 1)
                with ops.control_dependencies([v0.assign_add(1.0)]):
                    ta0 = ta0.identity()
                with ops.control_dependencies([v1.assign_add(1.0)]):
                    ta1 = ta1.identity()
                read0 = ta0.read(0)
                read1 = ta1.read(0)
                size0 = ta0.size()
                size1 = ta1.size()
                tensor_arrays[0] = ta0
                tensor_arrays[1] = ta1
                return [read0, read1, size0, size1, v0, v1]
            self.evaluate(variables.global_variables_initializer())
            (read0_v, read1_v, size0_v, size1_v, v0, v1) = self.evaluate(xla.compile(fn))
            self.assertEqual(dtypes.float32, tensor_arrays[0].dtype)
            self.assertEqual(dtypes.int32, tensor_arrays[1].dtype)
            self.assertEqual(1.0, v0)
            self.assertEqual(1.0, v1)
            self.assertEqual(read0_v, 0)
            self.assertEqual(read1_v, 1)
            self.assertEqual(size0_v, 2)
            self.assertEqual(size1_v, 4)
if __name__ == '__main__':
    test.main()