"""Tests for ConstantOp."""
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.util import compat

class ConstantTest(test.TestCase):

    def _testCpu(self, x):
        if False:
            print('Hello World!')
        np_ans = np.array(x)
        with context.device('/device:CPU:0'):
            tf_ans = ops.convert_to_tensor(x).numpy()
        if np_ans.dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            self.assertAllClose(np_ans, tf_ans)
        else:
            self.assertAllEqual(np_ans, tf_ans)

    def _testGpu(self, x):
        if False:
            for i in range(10):
                print('nop')
        device = test_util.gpu_device_name()
        if device:
            np_ans = np.array(x)
            with context.device(device):
                tf_ans = ops.convert_to_tensor(x).numpy()
            if np_ans.dtype in [np.float32, np.float64, np.complex64, np.complex128]:
                self.assertAllClose(np_ans, tf_ans)
            else:
                self.assertAllEqual(np_ans, tf_ans)

    def _testAll(self, x):
        if False:
            i = 10
            return i + 15
        self._testCpu(x)
        self._testGpu(x)

    def testFloat(self):
        if False:
            while True:
                i = 10
        self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32))
        self._testAll(np.random.normal(size=30).reshape([2, 3, 5]).astype(np.float32))
        self._testAll(np.empty((2, 0, 5)).astype(np.float32))
        orig = [-1.0, 2.0, 0.0]
        tf_ans = constant_op.constant(orig)
        self.assertEqual(dtypes_lib.float32, tf_ans.dtype)
        self.assertAllClose(np.array(orig), tf_ans.numpy())
        orig = [-1.5, 2, 0]
        tf_ans = constant_op.constant(orig)
        self.assertEqual(dtypes_lib.float32, tf_ans.dtype)
        self.assertAllClose(np.array(orig), tf_ans.numpy())
        orig = [-5, 2.5, 0]
        tf_ans = constant_op.constant(orig)
        self.assertEqual(dtypes_lib.float32, tf_ans.dtype)
        self.assertAllClose(np.array(orig), tf_ans.numpy())
        orig = [1, 2 ** 42, 0.5]
        tf_ans = constant_op.constant(orig)
        self.assertEqual(dtypes_lib.float32, tf_ans.dtype)
        self.assertAllClose(np.array(orig), tf_ans.numpy())

    def testDouble(self):
        if False:
            while True:
                i = 10
        self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float64))
        self._testAll(np.random.normal(size=30).reshape([2, 3, 5]).astype(np.float64))
        self._testAll(np.empty((2, 0, 5)).astype(np.float64))
        orig = [-5, 2.5, 0]
        tf_ans = constant_op.constant(orig, dtypes_lib.float64)
        self.assertEqual(dtypes_lib.float64, tf_ans.dtype)
        self.assertAllClose(np.array(orig), tf_ans.numpy())
        tf_ans = constant_op.constant(2 ** 54 + 1, dtypes_lib.float64)
        self.assertEqual(2 ** 54, tf_ans.numpy())
        with self.assertRaisesRegex(ValueError, 'out-of-range integer'):
            constant_op.constant(10 ** 310, dtypes_lib.float64)

    def testInt32(self):
        if False:
            for i in range(10):
                print('nop')
        self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.int32))
        self._testAll((100 * np.random.normal(size=30)).reshape([2, 3, 5]).astype(np.int32))
        self._testAll(np.empty((2, 0, 5)).astype(np.int32))
        self._testAll([-1, 2])

    def testInt64(self):
        if False:
            for i in range(10):
                print('nop')
        self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.int64))
        self._testAll((100 * np.random.normal(size=30)).reshape([2, 3, 5]).astype(np.int64))
        self._testAll(np.empty((2, 0, 5)).astype(np.int64))
        orig = [2, 2 ** 48, -2 ** 48]
        tf_ans = constant_op.constant(orig)
        self.assertEqual(dtypes_lib.int64, tf_ans.dtype)
        self.assertAllClose(np.array(orig), tf_ans.numpy())
        with self.assertRaisesRegex(ValueError, 'out-of-range integer'):
            constant_op.constant([2 ** 72])

    def testComplex64(self):
        if False:
            return 10
        self._testAll((1 + 2j) * np.arange(-15, 15).reshape([2, 3, 5]).astype(np.complex64))
        self._testAll((1 + 2j) * np.random.normal(size=30).reshape([2, 3, 5]).astype(np.complex64))
        self._testAll(np.empty((2, 0, 5)).astype(np.complex64))

    def testComplex128(self):
        if False:
            for i in range(10):
                print('nop')
        self._testAll((1 + 2j) * np.arange(-15, 15).reshape([2, 3, 5]).astype(np.complex128))
        self._testAll((1 + 2j) * np.random.normal(size=30).reshape([2, 3, 5]).astype(np.complex128))
        self._testAll(np.empty((2, 0, 5)).astype(np.complex128))

    @test_util.disable_tfrt('support creating string tensors from empty numpy arrays.')
    def testString(self):
        if False:
            while True:
                i = 10
        val = [compat.as_bytes(str(x)) for x in np.arange(-15, 15)]
        self._testCpu(np.array(val).reshape([2, 3, 5]))
        self._testCpu(np.empty((2, 0, 5)).astype(np.str_))

    def testStringWithNulls(self):
        if False:
            i = 10
            return i + 15
        val = ops.convert_to_tensor(b'\x00\x00\x00\x00').numpy()
        self.assertEqual(len(val), 4)
        self.assertEqual(val, b'\x00\x00\x00\x00')
        val = ops.convert_to_tensor(b'xx\x00xx').numpy()
        self.assertEqual(len(val), 5)
        self.assertAllEqual(val, b'xx\x00xx')
        nested = [[b'\x00\x00\x00\x00', b'xx\x00xx'], [b'\x00_\x00_\x00_\x00', b'\x00']]
        val = ops.convert_to_tensor(nested).numpy()
        self.assertEqual(val.tolist(), nested)

    def testStringConstantOp(self):
        if False:
            return 10
        s = constant_op.constant('uiuc')
        self.assertEqual(s.numpy().decode('utf-8'), 'uiuc')
        s_array = constant_op.constant(['mit', 'stanford'])
        self.assertAllEqual(s_array.numpy(), ['mit', 'stanford'])
        with ops.device('/cpu:0'):
            s = constant_op.constant('cmu')
            self.assertEqual(s.numpy().decode('utf-8'), 'cmu')
            s_array = constant_op.constant(['berkeley', 'ucla'])
            self.assertAllEqual(s_array.numpy(), ['berkeley', 'ucla'])

    def testExplicitShapeNumPy(self):
        if False:
            print('Hello World!')
        c = constant_op.constant(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32), shape=[2, 3, 5])
        self.assertEqual(c.get_shape(), [2, 3, 5])

    def testImplicitShapeNumPy(self):
        if False:
            print('Hello World!')
        c = constant_op.constant(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32))
        self.assertEqual(c.get_shape(), [2, 3, 5])

    def testExplicitShapeList(self):
        if False:
            for i in range(10):
                print('nop')
        c = constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[7])
        self.assertEqual(c.get_shape(), [7])

    def testExplicitShapeFill(self):
        if False:
            return 10
        c = constant_op.constant(12, shape=[7])
        self.assertEqual(c.get_shape(), [7])
        self.assertAllEqual([12, 12, 12, 12, 12, 12, 12], c.numpy())

    def testExplicitShapeReshape(self):
        if False:
            for i in range(10):
                print('nop')
        c = constant_op.constant(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32), shape=[5, 2, 3])
        self.assertEqual(c.get_shape(), [5, 2, 3])

    def testImplicitShapeList(self):
        if False:
            for i in range(10):
                print('nop')
        c = constant_op.constant([1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(c.get_shape(), [7])

    def testExplicitShapeNumber(self):
        if False:
            i = 10
            return i + 15
        c = constant_op.constant(1, shape=[1])
        self.assertEqual(c.get_shape(), [1])

    def testImplicitShapeNumber(self):
        if False:
            for i in range(10):
                print('nop')
        c = constant_op.constant(1)
        self.assertEqual(c.get_shape(), [])

    def testShapeTooBig(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[10])

    def testShapeTooSmall(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[5])

    def testShapeWrong(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(TypeError, None):
            constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[5])

    def testShape(self):
        if False:
            print('Hello World!')
        self._testAll(constant_op.constant([1]).get_shape())

    def testDimension(self):
        if False:
            print('Hello World!')
        x = constant_op.constant([1]).shape[0]
        self._testAll(x)

    def testDimensionList(self):
        if False:
            print('Hello World!')
        x = [constant_op.constant([1]).shape[0]]
        self._testAll(x)
        self._testAll([1] + x)
        self._testAll(x + [1])

    def testDimensionTuple(self):
        if False:
            for i in range(10):
                print('nop')
        x = constant_op.constant([1]).shape[0]
        self._testAll((x,))
        self._testAll((1, x))
        self._testAll((x, 1))

    def testInvalidLength(self):
        if False:
            return 10

        class BadList(list):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super(BadList, self).__init__([1, 2, 3])

            def __len__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return -1
        with self.assertRaisesRegex(ValueError, 'should return >= 0'):
            constant_op.constant([BadList()])
        with self.assertRaisesRegex(ValueError, 'mixed types'):
            constant_op.constant([1, 2, BadList()])
        with self.assertRaisesRegex(ValueError, 'should return >= 0'):
            constant_op.constant(BadList())
        with self.assertRaisesRegex(ValueError, 'should return >= 0'):
            constant_op.constant([[BadList(), 2], 3])
        with self.assertRaisesRegex(ValueError, 'should return >= 0'):
            constant_op.constant([BadList(), [1, 2, 3]])
        with self.assertRaisesRegex(ValueError, 'should return >= 0'):
            constant_op.constant([BadList(), []])

    def testSparseValuesRaiseErrors(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'non-rectangular Python sequence'):
            constant_op.constant([[1, 2], [3]], dtype=dtypes_lib.int32)
        with self.assertRaisesRegex(ValueError, None):
            constant_op.constant([[1, 2], [3]])
        with self.assertRaisesRegex(ValueError, None):
            constant_op.constant([[1, 2], [3], [4, 5]])

    def testCustomSequence(self):
        if False:
            for i in range(10):
                print('nop')

        class MySeq(object):

            def __getitem__(self, key):
                if False:
                    for i in range(10):
                        print('nop')
                if key != 1 and key != 3:
                    raise KeyError(key)
                return key

            def __len__(self):
                if False:
                    return 10
                return 2

            def __iter__(self):
                if False:
                    return 10
                l = list([1, 3])
                return l.__iter__()
        self.assertAllEqual([1, 3], self.evaluate(constant_op.constant(MySeq())))

class AsTensorTest(test.TestCase):

    def testAsTensorForTensorInput(self):
        if False:
            print('Hello World!')
        t = constant_op.constant(10.0)
        x = ops.convert_to_tensor(t)
        self.assertIs(t, x)

    def testAsTensorForNonTensorInput(self):
        if False:
            print('Hello World!')
        x = ops.convert_to_tensor(10.0)
        self.assertTrue(isinstance(x, ops.EagerTensor))

class ZerosTest(test.TestCase):

    def _Zeros(self, shape):
        if False:
            for i in range(10):
                print('nop')
        ret = array_ops.zeros(shape)
        self.assertEqual(shape, ret.get_shape())
        return ret.numpy()

    def testConst(self):
        if False:
            return 10
        self.assertTrue(np.array_equal(self._Zeros([2, 3]), np.array([[0] * 3] * 2)))

    def testScalar(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(0, self._Zeros([]))
        self.assertEqual(0, self._Zeros(()))
        scalar = array_ops.zeros(constant_op.constant([], dtype=dtypes_lib.int32))
        self.assertEqual(0, scalar.numpy())

    def testDynamicSizes(self):
        if False:
            while True:
                i = 10
        np_ans = np.array([[0] * 3] * 2)
        d = array_ops.fill([2, 3], 12.0, name='fill')
        z = array_ops.zeros(array_ops.shape(d))
        out = z.numpy()
        self.assertAllEqual(np_ans, out)
        self.assertShapeEqual(np_ans, d)
        self.assertShapeEqual(np_ans, z)

    def testDtype(self):
        if False:
            while True:
                i = 10
        d = array_ops.fill([2, 3], 12.0, name='fill')
        self.assertEqual(d.get_shape(), [2, 3])
        z = array_ops.zeros([2, 3])
        self.assertEqual(z.dtype, dtypes_lib.float32)
        self.assertEqual([2, 3], z.get_shape())
        self.assertAllEqual(z.numpy(), np.zeros([2, 3]))
        z = array_ops.zeros(array_ops.shape(d))
        self.assertEqual(z.dtype, dtypes_lib.float32)
        self.assertEqual([2, 3], z.get_shape())
        self.assertAllEqual(z.numpy(), np.zeros([2, 3]))
        for dtype in [dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32, dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8, dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64, dtypes_lib.bool]:
            z = array_ops.zeros([2, 3], dtype=dtype)
            self.assertEqual(z.dtype, dtype)
            self.assertEqual([2, 3], z.get_shape())
            z_value = z.numpy()
            self.assertFalse(np.any(z_value))
            self.assertEqual((2, 3), z_value.shape)
            z = array_ops.zeros(array_ops.shape(d), dtype=dtype)
            self.assertEqual(z.dtype, dtype)
            self.assertEqual([2, 3], z.get_shape())
            z_value = z.numpy()
            self.assertFalse(np.any(z_value))
            self.assertEqual((2, 3), z_value.shape)

class ZerosLikeTest(test.TestCase):

    def _compareZeros(self, dtype, use_gpu):
        if False:
            i = 10
            return i + 15
        if dtype == dtypes_lib.string:
            numpy_dtype = np.string_
        else:
            numpy_dtype = dtype.as_numpy_dtype
        d = constant_op.constant(np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
        z_var = array_ops.zeros_like(d)
        self.assertEqual(z_var.dtype, dtype)
        self.assertEqual([2, 3], z_var.get_shape())
        z_value = z_var.numpy()
        self.assertFalse(np.any(z_value))
        self.assertEqual((2, 3), z_value.shape)

    @test_util.disable_tfrt('b/169112823: unsupported dtype for Op:ZerosLike.')
    def testZerosLikeCPU(self):
        if False:
            print('Hello World!')
        for dtype in [dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32, dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8, dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64]:
            self._compareZeros(dtype, use_gpu=False)

    @test_util.disable_tfrt('b/169112823: unsupported dtype for Op:ZerosLike.')
    def testZerosLikeGPU(self):
        if False:
            i = 10
            return i + 15
        for dtype in [dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32, dtypes_lib.bool, dtypes_lib.int64]:
            self._compareZeros(dtype, use_gpu=True)

    @test_util.disable_tfrt('b/169112823: unsupported dtype for Op:ZerosLike.')
    def testZerosLikeDtype(self):
        if False:
            for i in range(10):
                print('nop')
        shape = (3, 5)
        dtypes = (np.float32, np.complex64)
        for in_type in dtypes:
            x = np.arange(15).astype(in_type).reshape(*shape)
            for out_type in dtypes:
                y = array_ops.zeros_like(x, dtype=out_type).numpy()
                self.assertEqual(y.dtype, out_type)
                self.assertEqual(y.shape, shape)
                self.assertAllEqual(y, np.zeros(shape, dtype=out_type))

class OnesTest(test.TestCase):

    def _Ones(self, shape):
        if False:
            return 10
        ret = array_ops.ones(shape)
        self.assertEqual(shape, ret.get_shape())
        return ret.numpy()

    def testConst(self):
        if False:
            return 10
        self.assertTrue(np.array_equal(self._Ones([2, 3]), np.array([[1] * 3] * 2)))

    def testScalar(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(1, self._Ones([]))
        self.assertEqual(1, self._Ones(()))
        scalar = array_ops.ones(constant_op.constant([], dtype=dtypes_lib.int32))
        self.assertEqual(1, scalar.numpy())

    def testDynamicSizes(self):
        if False:
            while True:
                i = 10
        np_ans = np.array([[1] * 3] * 2)
        d = array_ops.fill([2, 3], 12.0, name='fill')
        z = array_ops.ones(array_ops.shape(d))
        out = z.numpy()
        self.assertAllEqual(np_ans, out)
        self.assertShapeEqual(np_ans, d)
        self.assertShapeEqual(np_ans, z)

    def testDtype(self):
        if False:
            return 10
        d = array_ops.fill([2, 3], 12.0, name='fill')
        self.assertEqual(d.get_shape(), [2, 3])
        z = array_ops.ones([2, 3])
        self.assertEqual(z.dtype, dtypes_lib.float32)
        self.assertEqual([2, 3], z.get_shape())
        self.assertAllEqual(z.numpy(), np.ones([2, 3]))
        z = array_ops.ones(array_ops.shape(d))
        self.assertEqual(z.dtype, dtypes_lib.float32)
        self.assertEqual([2, 3], z.get_shape())
        self.assertAllEqual(z.numpy(), np.ones([2, 3]))
        for dtype in (dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32, dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8, dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64, dtypes_lib.bool):
            z = array_ops.ones([2, 3], dtype=dtype)
            self.assertEqual(z.dtype, dtype)
            self.assertEqual([2, 3], z.get_shape())
            self.assertAllEqual(z.numpy(), np.ones([2, 3]))
            z = array_ops.ones(array_ops.shape(d), dtype=dtype)
            self.assertEqual(z.dtype, dtype)
            self.assertEqual([2, 3], z.get_shape())
            self.assertAllEqual(z.numpy(), np.ones([2, 3]))

class OnesLikeTest(test.TestCase):

    def testOnesLike(self):
        if False:
            while True:
                i = 10
        for dtype in [dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32, dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8, dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64]:
            numpy_dtype = dtype.as_numpy_dtype
            d = constant_op.constant(np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
            z_var = array_ops.ones_like(d)
            self.assertEqual(z_var.dtype, dtype)
            z_value = z_var.numpy()
            self.assertTrue(np.array_equal(z_value, np.array([[1] * 3] * 2)))
            self.assertEqual([2, 3], z_var.get_shape())

class FillTest(test.TestCase):

    def _compare(self, dims, val, np_ans, use_gpu):
        if False:
            for i in range(10):
                print('nop')
        ctx = context.context()
        device = 'GPU:0' if use_gpu and ctx.num_gpus() else 'CPU:0'
        with ops.device(device):
            tf_ans = array_ops.fill(dims, val, name='fill')
            out = tf_ans.numpy()
        self.assertAllClose(np_ans, out)

    def _compareAll(self, dims, val, np_ans):
        if False:
            return 10
        self._compare(dims, val, np_ans, False)
        self._compare(dims, val, np_ans, True)

    def testFillFloat(self):
        if False:
            for i in range(10):
                print('nop')
        np_ans = np.array([[3.1415] * 3] * 2).astype(np.float32)
        self._compareAll([2, 3], np_ans[0][0], np_ans)

    def testFillDouble(self):
        if False:
            for i in range(10):
                print('nop')
        np_ans = np.array([[3.1415] * 3] * 2).astype(np.float64)
        self._compareAll([2, 3], np_ans[0][0], np_ans)

    def testFillInt32(self):
        if False:
            return 10
        np_ans = np.array([[42] * 3] * 2).astype(np.int32)
        self._compareAll([2, 3], np_ans[0][0], np_ans)

    def testFillInt64(self):
        if False:
            i = 10
            return i + 15
        np_ans = np.array([[-42] * 3] * 2).astype(np.int64)
        self._compareAll([2, 3], np_ans[0][0], np_ans)

    def testFillComplex64(self):
        if False:
            return 10
        np_ans = np.array([[0.15] * 3] * 2).astype(np.complex64)
        self._compare([2, 3], np_ans[0][0], np_ans, use_gpu=False)

    def testFillComplex128(self):
        if False:
            for i in range(10):
                print('nop')
        np_ans = np.array([[0.15] * 3] * 2).astype(np.complex128)
        self._compare([2, 3], np_ans[0][0], np_ans, use_gpu=False)

    def testFillString(self):
        if False:
            print('Hello World!')
        np_ans = np.array([[b'yolo'] * 3] * 2)
        tf_ans = array_ops.fill([2, 3], np_ans[0][0], name='fill').numpy()
        self.assertAllEqual(np_ans, tf_ans)

    def testFillNegative(self):
        if False:
            for i in range(10):
                print('nop')
        for shape in ((-1,), (2, -1), (-1, 2), -2, -3):
            with self.assertRaises(errors_impl.InvalidArgumentError):
                array_ops.fill(shape, 7)

    def testShapeFunctionEdgeCases(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(errors_impl.InvalidArgumentError):
            array_ops.fill([[0, 1], [2, 3]], 1.0)
        with self.assertRaises(errors_impl.InvalidArgumentError):
            array_ops.fill([3, 2], [1.0, 2.0])
if __name__ == '__main__':
    test.main()