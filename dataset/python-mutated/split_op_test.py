"""Functional tests for Split Op."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
_TEST_DTYPES = (dtypes.int8, dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128, dtypes.bfloat16, dtypes.float8_e5m2, dtypes.float8_e4m3fn)
if not test_util.is_xla_enabled():
    _TEST_DTYPES += (dtypes.int4,)

class SplitOpTest(test.TestCase):

    def _makeData(self, shape, dtype):
        if False:
            return 10
        data = np.random.rand(*shape).astype(dtype.as_numpy_dtype)
        if dtype.is_complex:
            data -= 1j * data
        return data

    @test_util.run_deprecated_v1
    def testShapeInference(self):
        if False:
            print('Hello World!')
        model_input = array_ops.placeholder(dtypes.float32, shape=(1, 10))
        with self.assertRaises(ValueError):
            array_ops.split(model_input, [4], axis=1)[0]
        model_input = array_ops.placeholder(dtypes.float32)
        inp = np.zeros((1, 10))
        with self.cached_session() as sess:
            with self.assertRaises(errors_impl.InvalidArgumentError):
                sess.run(array_ops.split(model_input, [4]), {model_input: inp})
        for axis in [0, -2]:
            with self.cached_session() as sess:
                with self.assertRaises(ValueError):
                    sess.run(array_ops.split(array_ops.ones([4, 4]), num_or_size_splits=constant_op.constant(2), axis=axis))
        result = array_ops.split(array_ops.ones([5, 2]), array_ops.constant([2, 1, 2]) * 1, axis=0)
        self.assertEqual(result[0].shape[1], 2)
        self.assertEqual(result[1].shape[1], 2)
        self.assertEqual(result[2].shape[1], 2)
        model_input2 = array_ops.placeholder(dtypes.float32, shape=[None, 2])
        result = array_ops.split(model_input2, [2, 2], axis=0)[0]
        with self.cached_session() as sess:
            sess.run(result, feed_dict={model_input2: np.ones([4, 2])})

    @test_util.run_deprecated_v1
    def testFailWithoutExplicitNum(self):
        if False:
            print('Hello World!')
        size_splits = array_ops.placeholder(dtype=dtypes.int32, shape=[None])
        value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        with self.session() as sess:
            with self.assertRaises(ValueError) as context:
                sess.run(array_ops.split(value, size_splits), {size_splits: [2, 2, 6]})
            self.assertIn('Cannot infer argument `num` from shape', str(context.exception))

    @test_util.run_in_graph_and_eager_modes
    def testExplicitNum(self):
        if False:
            for i in range(10):
                print('nop')
        size_splits = array_ops.constant([2, 2, 6], dtype=dtypes.int32)
        value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        with self.assertRaises((errors_impl.InvalidArgumentError, ValueError)):
            array_ops.split(value, size_splits, num=4)
        r = self.evaluate(array_ops.split(value, size_splits, num=3))
        self.assertAllEqual(r[0], value[0:2])
        self.assertAllEqual(r[1], value[2:4])
        self.assertAllEqual(r[2], value[4:])

    @test_util.run_in_graph_and_eager_modes
    def testListOfScalarTensors(self):
        if False:
            while True:
                i = 10
        a = math_ops.cast(5, dtypes.int32)
        b = math_ops.cast(6, dtypes.int32)
        value = np.random.rand(11, 11)
        with test_util.device(use_gpu=True):
            result = self.evaluate(array_ops.split(value, [a, b]))
        self.assertAllEqual(result[0], value[0:5, :])
        self.assertAllEqual(result[1], value[5:, :])

    def _RunAndVerifyVariable(self, dtype, large_num_splits=False):
        if False:
            return 10
        shape = np.random.randint(1, 5, size=5)
        split_dim = np.random.randint(-5, 5)
        if large_num_splits:
            num_split = np.random.randint(16, 25)
        else:
            num_split = np.random.randint(2, 8)
        size_splits = np.random.randint(2, 8, num_split, dtype=np.int32)
        shape[split_dim] = np.sum(size_splits)
        inp = self._makeData(shape, dtype)
        with test_util.device(use_gpu=True):
            result = self.evaluate(array_ops.split(inp, size_splits, split_dim))
        slices = [slice(0, x) for x in shape]
        offset = 0
        for i in range(num_split):
            slices[split_dim] = slice(offset, offset + size_splits[i])
            offset += size_splits[i]
            self.assertAllEqual(result[i], inp[tuple(slices)])

    def _testSpecialCasesVariable(self):
        if False:
            i = 10
            return i + 15
        inp = np.random.rand(4, 4).astype('f')
        with test_util.device(use_gpu=True):
            result = self.evaluate(array_ops.split(inp, [4], 0))
            self.assertAllEqual(result[0], inp)
            result = self.evaluate(array_ops.split(inp, [-1, 3], 0))
            self.assertAllEqual(result[0], inp[0:1, :])
            self.assertAllEqual(result[1], inp[1:4, :])

    def _testHugeNumberOfTensorsVariable(self, dtype):
        if False:
            i = 10
            return i + 15
        num_split = 1000
        size_splits = np.random.randint(1, 3, num_split, dtype=np.int32)
        shape = [3, np.sum(size_splits)]
        split_dim = 1
        inp = self._makeData(shape, dtype)
        with test_util.device(use_gpu=True):
            result = self.evaluate(array_ops.split(inp, size_splits, split_dim))
        slices = [slice(0, x) for x in shape]
        offset = 0
        for i in range(num_split):
            slices[split_dim] = slice(offset, offset + size_splits[i])
            offset += size_splits[i]
            self.assertAllEqual(result[i], inp[tuple(slices)])

    @test_util.run_in_graph_and_eager_modes
    def testSpecialCasesVariable(self):
        if False:
            print('Hello World!')
        self._testSpecialCasesVariable()
        for dtype in _TEST_DTYPES:
            self._testHugeNumberOfTensorsVariable(dtype)

    @test_util.run_in_graph_and_eager_modes
    def testDegenerateVariable(self):
        if False:
            print('Hello World!')
        inp = np.random.rand(4, 4).astype('f')
        with test_util.device(use_gpu=True):
            result = self.evaluate(array_ops.split(inp, [-1, 4], 0))
            self.assertAllEqual(result[0], inp[0:0, :])
            self.assertAllEqual(result[1], inp[0:4, :])
            result = self.evaluate(array_ops.split(inp, [4, -1], 0))
            self.assertAllEqual(result[0], inp[0:4, :])
            self.assertAllEqual(result[1], inp[4:4, :])
            result = self.evaluate(array_ops.split(inp, [-1, 4], 1))
            self.assertAllEqual(result[0], inp[:, 0:0])
            self.assertAllEqual(result[1], inp[:, 0:4])
            result = self.evaluate(array_ops.split(inp, [4, -1], 1))
            self.assertAllEqual(result[0], inp[:, 0:4])
            self.assertAllEqual(result[1], inp[:, 4:4])

    def _testGradientsSimpleVariable(self, dtype):
        if False:
            return 10
        inp = self._makeData((4, 4), dtype)
        with test_util.device(use_gpu=True):
            inp_tensor = ops.convert_to_tensor(inp)
            s = array_ops.split(inp_tensor, [1, 3], 1)
            inp_grads = [self._makeData((4, 1), dtype), self._makeData((4, 3), dtype)]
            grad_tensors = [constant_op.constant(x) for x in inp_grads]
            grad = gradients_impl.gradients(s, [inp_tensor], grad_tensors)[-1]
            result = self.evaluate(grad)
        self.assertAllEqual(result[:, 0:1], inp_grads[0])
        self.assertAllEqual(result[:, 1:4], inp_grads[1])

    @test_util.run_deprecated_v1
    def testOutputShape(self):
        if False:
            for i in range(10):
                print('nop')
        for axis in [1, -1]:
            with self.cached_session():
                tensor = array_ops.placeholder(dtypes.float32, shape=[None, 12])
                size_splits = [3, 7, 2]
                outputs = array_ops.split(tensor, size_splits, axis)
                for (i, output) in enumerate(outputs):
                    self.assertEqual(output.get_shape().as_list(), [None, size_splits[i]])

    def _compare(self, x, dim, num):
        if False:
            print('Hello World!')
        np_ans = np.split(x, num, dim)
        with test_util.device(use_gpu=True):
            tf_ans = array_ops.split(value=x, num_or_size_splits=num, axis=dim)
            out = self.evaluate(tf_ans)
        self.assertEqual(num, len(np_ans))
        self.assertEqual(num, len(tf_ans))
        self.assertEqual(num, len(out))
        for i in range(num):
            self.assertAllEqual(np_ans[i], out[i])
            self.assertShapeEqual(np_ans[i], tf_ans[i])

    @test_util.run_in_graph_and_eager_modes
    def testSplitRows(self):
        if False:
            while True:
                i = 10
        for dtype in _TEST_DTYPES:
            inp = self._makeData((4, 4), dtype)
            self._compare(inp, 0, 4)

    @test_util.run_in_graph_and_eager_modes
    def testSplitCols(self):
        if False:
            while True:
                i = 10
        for dtype in _TEST_DTYPES:
            inp = self._makeData((4, 4), dtype)
            self._compare(inp, 1, 4)

    def _testEmpty(self, x, dim, num, expected_shape):
        if False:
            print('Hello World!')
        with test_util.device(use_gpu=True):
            tf_ans = array_ops.split(value=x, num_or_size_splits=num, axis=dim)
            out = self.evaluate(tf_ans)
        self.assertEqual(x.size, 0)
        self.assertEqual(len(out), num)
        for i in range(num):
            self.assertEqual(out[i].shape, expected_shape)
            self.assertEqual(expected_shape, tf_ans[i].get_shape())

    @test_util.run_in_graph_and_eager_modes
    def testEmpty(self):
        if False:
            print('Hello World!')
        for dtype in _TEST_DTYPES:
            inp = self._makeData((8, 0, 21), dtype)
            self._testEmpty(inp, 0, 2, (4, 0, 21))
            self._testEmpty(inp, 0, 4, (2, 0, 21))
            self._testEmpty(inp, 1, 4, (8, 0, 21))
            self._testEmpty(inp, 2, 3, (8, 0, 7))
            self._testEmpty(inp, 2, 7, (8, 0, 3))

    @test_util.run_in_graph_and_eager_modes
    def testIdentity(self):
        if False:
            print('Hello World!')
        for dtype in _TEST_DTYPES:
            inp = self._makeData((2, 2, 2), dtype)
            self._compare(inp, 0, 1)
            self._compare(inp, 1, 1)
            self._compare(inp, 2, 1)

    @test_util.run_in_graph_and_eager_modes
    def testSplitDim0(self):
        if False:
            while True:
                i = 10
        for dtype in _TEST_DTYPES:
            self._compare(self._makeData((6, 10, 18), dtype), 0, 3)
            self._compare(self._makeData((6, 7, 18), dtype), 0, 3)
            self._compare(self._makeData((6, 7, 9), dtype), 0, 3)

    def _RunAndVerify(self, dtype, large_num_splits=False):
        if False:
            i = 10
            return i + 15
        shape = np.random.randint(0, 5, size=5)
        split_dim = np.random.randint(-5, 5)
        if large_num_splits:
            num_split = np.random.randint(9, 15)
        else:
            num_split = np.random.randint(2, 8)
        shape[split_dim] = np.random.randint(2, 5) * num_split
        inp = self._makeData(shape, dtype)
        with test_util.device(use_gpu=True):
            result = self.evaluate(array_ops.split(value=inp, num_or_size_splits=num_split, axis=split_dim))
        slices = [slice(0, x) for x in shape]
        offset = 0
        length = shape[split_dim] // num_split
        for i in range(num_split):
            slices[split_dim] = slice(offset, offset + length)
            offset += length
            self.assertAllEqual(result[i], inp[tuple(slices)])

    @test_util.run_in_graph_and_eager_modes
    def testRandom(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in _TEST_DTYPES:
            for _ in range(5):
                self._RunAndVerify(dtype)
                self._RunAndVerify(dtype, large_num_splits=True)
                self._RunAndVerifyVariable(dtype)
                self._RunAndVerifyVariable(dtype, large_num_splits=True)

    def _testGradientsSimple(self, dtype):
        if False:
            i = 10
            return i + 15
        inp = self._makeData((4, 4), dtype)
        with self.cached_session():
            inp_tensor = ops.convert_to_tensor(inp)
            s = array_ops.split(value=inp_tensor, num_or_size_splits=4, axis=1)
            inp_grads = [self._makeData((4, 1), dtype) for _ in range(4)]
            grad_tensors = [constant_op.constant(x) for x in inp_grads]
            grad = gradients_impl.gradients(s, [inp_tensor], grad_tensors)[0]
            result = self.evaluate(grad)
        for i in range(4):
            self.assertAllEqual(result[:, i:i + 1], inp_grads[i])

    @test_util.run_deprecated_v1
    def testGradientsAll(self):
        if False:
            return 10
        for dtype in _TEST_DTYPES:
            if not dtype.is_integer and dtype not in [dtypes.float8_e5m2, dtypes.float8_e4m3fn]:
                self._testGradientsSimple(dtype)
                self._testGradientsSimpleVariable(dtype)

    @test_util.run_deprecated_v1
    def testShapeFunctionEdgeCases(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            array_ops.split(value=[[0, 1], [2, 3]], num_or_size_splits=4, axis=2)
        with self.assertRaises(ValueError):
            array_ops.split(value=[[0, 1], [2, 3]], num_or_size_splits=4, axis=-3)
        with self.assertRaisesRegex(ValueError, 'should evenly divide'):
            array_ops.split(value=[0, 1, 2, 3], num_or_size_splits=3, axis=0)
        splits = array_ops.split(value=[[0, 1, 2, 3]], num_or_size_splits=4, axis=array_ops.placeholder(dtypes.int32))
        for s in splits:
            self.assertEqual([None, None], s.get_shape().as_list())
        splits = array_ops.split(value=array_ops.placeholder(dtypes.float32), num_or_size_splits=4, axis=array_ops.placeholder(dtypes.int32))
        for s in splits:
            self.assertEqual(None, s.get_shape().ndims)

    @test_util.run_deprecated_v1
    def testVariableShapeFunction(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            array_ops.split([0, 1], [3, -1], axis=0)
        (s0, s1) = array_ops.split([0, 1, 2], [2, -1], axis=0)
        assert s0.shape.as_list() == [2]
        assert s1.shape.as_list() == [1]

    @test_util.run_deprecated_v1
    def testNonexistentDimTensor(self):
        if False:
            print('Hello World!')
        x = array_ops.placeholder(dtypes.int32)
        values = np.zeros([5, 30])
        splits = array_ops.placeholder(dtypes.int32)
        with self.assertRaisesRegex(ValueError, 'Cannot infer'):
            y = array_ops.split(values, splits, axis=x)
        splits = array_ops.placeholder(dtypes.int32, [3])
        y = array_ops.split(values, splits, axis=x)
        with self.session() as sess:
            with self.assertRaisesRegex(errors_impl.InvalidArgumentError, 'must have exactly one element'):
                sess.run(y, {x: np.array([], dtype=np.int32), splits: [4, 11, 15]})

    @test_util.run_in_graph_and_eager_modes
    def testNegativeSizes(self):
        if False:
            for i in range(10):
                print('nop')
        x = constant_op.constant([1, 2, 3], dtypes.float32)
        with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError), 'Split size at index 1 must be >= .*. Got: -2'):
            splits = [-1, -2]
            self.evaluate(array_ops.split(x, splits, axis=0))

    @test_util.run_in_graph_and_eager_modes
    def testBadSplitSizes(self):
        if False:
            return 10
        x = constant_op.constant([1, 2], dtypes.float32)
        with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError), "Determined shape must either match input|can't split axis"):
            splits = [1, 2]
            self.evaluate(array_ops.split(x, splits, axis=0))

    @test_util.run_in_graph_and_eager_modes
    def testSplitVBigTensors(self):
        if False:
            for i in range(10):
                print('nop')
        input_shape = [1, 64, 32768]
        x = np.linspace(start=1, stop=np.prod(input_shape), num=np.prod(input_shape), dtype=np.float32).reshape(input_shape)
        split_axis = 1
        size_splits = [1] * input_shape[split_axis]
        y = array_ops.split(x, num_or_size_splits=size_splits, axis=split_axis)
        for i in range(input_shape[split_axis]):
            result = y[i]
            expected = x[:, i:i + 1, :]
            self.assertAllEqual(result, expected)

    @test_util.run_in_graph_and_eager_modes
    def testSplitVBigTensorsWithIrregularSplits(self):
        if False:
            while True:
                i = 10
        input_shape = [1, 64, 32768]
        x = np.linspace(start=1, stop=np.prod(input_shape), num=np.prod(input_shape), dtype=np.float32).reshape(input_shape)
        split_axis = 1
        size_splits = [32, 16, 8, 4, 2, 1, 1]
        y = array_ops.split(x, num_or_size_splits=size_splits, axis=split_axis)
        start = 0
        for i in range(len(size_splits)):
            result = y[i]
            split_size = size_splits[i]
            expected = x[:, start:start + split_size, :]
            start += split_size
            self.assertAllEqual(result, expected)
if __name__ == '__main__':
    test.main()