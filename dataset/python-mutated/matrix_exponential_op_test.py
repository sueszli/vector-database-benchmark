"""Tests for tensorflow.ops.linalg.linalg_impl.matrix_exponential."""
import itertools
import numpy as np
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

def np_expm(x):
    if False:
        for i in range(10):
            print('nop')
    'Slow but accurate Taylor series matrix exponential.'
    y = np.zeros(x.shape, dtype=x.dtype)
    xn = np.eye(x.shape[0], dtype=x.dtype)
    for n in range(40):
        if n > 0:
            xn /= float(n)
        y += xn
        xn = np.dot(xn, x)
    return y

@test_util.run_all_without_tensor_float_32('Avoid TF32-based matmuls.')
class ExponentialOpTest(test.TestCase):

    def _verifyExponential(self, x, np_type):
        if False:
            i = 10
            return i + 15
        inp = x.astype(np_type)
        with test_util.use_gpu():
            tf_ans = linalg_impl.matrix_exponential(inp)
            if x.size == 0:
                np_ans = np.empty(x.shape, dtype=np_type)
            elif x.ndim > 2:
                np_ans = np.zeros(inp.shape, dtype=np_type)
                for i in itertools.product(*[range(x) for x in inp.shape[:-2]]):
                    np_ans[i] = np_expm(inp[i])
            else:
                np_ans = np_expm(inp)
            out = self.evaluate(tf_ans)
            self.assertAllClose(np_ans, out, rtol=0.001, atol=0.001)

    def _verifyExponentialReal(self, x):
        if False:
            while True:
                i = 10
        for np_type in [np.float32, np.float64]:
            self._verifyExponential(x, np_type)

    def _verifyExponentialComplex(self, x):
        if False:
            print('Hello World!')
        for np_type in [np.complex64, np.complex128]:
            self._verifyExponential(x, np_type)

    def _makeBatch(self, matrix1, matrix2):
        if False:
            return 10
        matrix_batch = np.concatenate([np.expand_dims(matrix1, 0), np.expand_dims(matrix2, 0)])
        matrix_batch = np.tile(matrix_batch, [2, 3, 1, 1])
        return matrix_batch

    def testNonsymmetricReal(self):
        if False:
            while True:
                i = 10
        matrix1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        matrix2 = np.array([[1.0, 3.0], [3.0, 5.0]])
        self._verifyExponentialReal(matrix1)
        self._verifyExponentialReal(matrix2)
        self._verifyExponentialReal(self._makeBatch(matrix1, matrix2))

    @test_util.run_deprecated_v1
    def testNonsymmetricComplex(self):
        if False:
            i = 10
            return i + 15
        matrix1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        matrix2 = np.array([[1.0, 3.0], [3.0, 5.0]])
        matrix1 = matrix1.astype(np.complex64)
        matrix1 += 1j * matrix1
        matrix2 = matrix2.astype(np.complex64)
        matrix2 += 1j * matrix2
        self._verifyExponentialComplex(matrix1)
        self._verifyExponentialComplex(matrix2)
        self._verifyExponentialComplex(self._makeBatch(matrix1, matrix2))

    def testSymmetricPositiveDefiniteReal(self):
        if False:
            for i in range(10):
                print('nop')
        matrix1 = np.array([[2.0, 1.0], [1.0, 2.0]])
        matrix2 = np.array([[3.0, -1.0], [-1.0, 3.0]])
        self._verifyExponentialReal(matrix1)
        self._verifyExponentialReal(matrix2)
        self._verifyExponentialReal(self._makeBatch(matrix1, matrix2))

    def testSymmetricPositiveDefiniteComplex(self):
        if False:
            for i in range(10):
                print('nop')
        matrix1 = np.array([[2.0, 1.0], [1.0, 2.0]])
        matrix2 = np.array([[3.0, -1.0], [-1.0, 3.0]])
        matrix1 = matrix1.astype(np.complex64)
        matrix1 += 1j * matrix1
        matrix2 = matrix2.astype(np.complex64)
        matrix2 += 1j * matrix2
        self._verifyExponentialComplex(matrix1)
        self._verifyExponentialComplex(matrix2)
        self._verifyExponentialComplex(self._makeBatch(matrix1, matrix2))

    @test_util.run_deprecated_v1
    def testNonSquareMatrix(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            linalg_impl.matrix_exponential(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]))

    @test_util.run_deprecated_v1
    def testWrongDimensions(self):
        if False:
            while True:
                i = 10
        tensor3 = constant_op.constant([1.0, 2.0])
        with self.assertRaises(ValueError):
            linalg_impl.matrix_exponential(tensor3)

    def testInfinite(self):
        if False:
            i = 10
            return i + 15
        in_tensor = [[np.inf, 1.0], [1.0, 1.0]]
        result = self.evaluate(linalg_impl.matrix_exponential(in_tensor))
        self.assertTrue(np.all(np.isnan(result)))

    def testEmpty(self):
        if False:
            while True:
                i = 10
        self._verifyExponentialReal(np.empty([0, 2, 2]))
        self._verifyExponentialReal(np.empty([2, 0, 0]))

    @test_util.run_deprecated_v1
    def testDynamic(self):
        if False:
            return 10
        with self.session() as sess:
            inp = array_ops.placeholder(ops.dtypes.float32)
            expm = linalg_impl.matrix_exponential(inp)
            matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
            sess.run(expm, feed_dict={inp: matrix})

    @test_util.run_deprecated_v1
    def testConcurrentExecutesWithoutError(self):
        if False:
            print('Hello World!')
        with self.session():
            matrix1 = random_ops.random_normal([5, 5], seed=42)
            matrix2 = random_ops.random_normal([5, 5], seed=42)
            expm1 = linalg_impl.matrix_exponential(matrix1)
            expm2 = linalg_impl.matrix_exponential(matrix2)
            expm = self.evaluate([expm1, expm2])
            self.assertAllEqual(expm[0], expm[1])

class MatrixExponentialBenchmark(test.Benchmark):
    shapes = [(4, 4), (10, 10), (16, 16), (101, 101), (256, 256), (1000, 1000), (1024, 1024), (2048, 2048), (513, 4, 4), (513, 16, 16), (513, 256, 256)]

    def _GenerateMatrix(self, shape):
        if False:
            return 10
        batch_shape = shape[:-2]
        shape = shape[-2:]
        assert shape[0] == shape[1]
        n = shape[0]
        matrix = np.ones(shape).astype(np.float32) / (2.0 * n) + np.diag(np.ones(n).astype(np.float32))
        return variables.Variable(np.tile(matrix, batch_shape + (1, 1)))

    def benchmarkMatrixExponentialOp(self):
        if False:
            return 10
        for shape in self.shapes:
            with ops.Graph().as_default(), session.Session(config=benchmark.benchmark_config()) as sess, ops.device('/cpu:0'):
                matrix = self._GenerateMatrix(shape)
                expm = linalg_impl.matrix_exponential(matrix)
                self.evaluate(variables.global_variables_initializer())
                self.run_op_benchmark(sess, control_flow_ops.group(expm), min_iters=25, name='matrix_exponential_cpu_{shape}'.format(shape=shape))
            if test.is_gpu_available(True):
                with ops.Graph().as_default(), session.Session(config=benchmark.benchmark_config()) as sess, ops.device('/gpu:0'):
                    matrix = self._GenerateMatrix(shape)
                    expm = linalg_impl.matrix_exponential(matrix)
                    self.evaluate(variables.global_variables_initializer())
                    self.run_op_benchmark(sess, control_flow_ops.group(expm), min_iters=25, name='matrix_exponential_gpu_{shape}'.format(shape=shape))

def _TestRandomSmall(dtype, batch_dims, size):
    if False:
        return 10

    def Test(self):
        if False:
            while True:
                i = 10
        np.random.seed(42)
        shape = batch_dims + (size, size)
        matrix = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(dtype)
        self._verifyExponentialReal(matrix)
    return Test

def _TestL1Norms(dtype, shape, scale):
    if False:
        i = 10
        return i + 15

    def Test(self):
        if False:
            while True:
                i = 10
        np.random.seed(42)
        matrix = np.random.uniform(low=-1.0, high=1.0, size=np.prod(shape)).reshape(shape).astype(dtype)
        l1_norm = np.max(np.sum(np.abs(matrix), axis=matrix.ndim - 2))
        matrix /= l1_norm
        self._verifyExponentialReal(scale * matrix)
    return Test
if __name__ == '__main__':
    for dtype_ in [np.float32, np.float64, np.complex64, np.complex128]:
        for batch_ in [(), (2,), (2, 2)]:
            for size_ in [4, 7]:
                name = '%s_%d_%d' % (dtype_.__name__, len(batch_), size_)
                setattr(ExponentialOpTest, 'testL1Norms_' + name, _TestRandomSmall(dtype_, batch_, size_))
    for shape_ in [(3, 3), (2, 3, 3)]:
        for dtype_ in [np.float32, np.complex64]:
            for scale_ in [0.1, 1.5, 5.0, 20.0]:
                name = '%s_%d_%d' % (dtype_.__name__, len(shape_), int(scale_ * 10))
                setattr(ExponentialOpTest, 'testL1Norms_' + name, _TestL1Norms(dtype_, shape_, scale_))
        for dtype_ in [np.float64, np.complex128]:
            for scale_ in [0.01, 0.2, 0.5, 1.5, 6.0, 25.0]:
                name = '%s_%d_%d' % (dtype_.__name__, len(shape_), int(scale_ * 100))
                setattr(ExponentialOpTest, 'testL1Norms_' + name, _TestL1Norms(dtype_, shape_, scale_))
    test.main()