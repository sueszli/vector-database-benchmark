"""Tests for tensorflow.ops.linalg_ops.eig."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.platform import test

def _AddTest(test_class, op_name, testcase_name, fn):
    if False:
        return 10
    test_name = '_'.join(['test', op_name, testcase_name])
    if hasattr(test_class, test_name):
        raise RuntimeError('Test %s defined more than once' % test_name)
    setattr(test_class, test_name, fn)

class EigTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testWrongDimensions(self):
        if False:
            print('Hello World!')
        scalar = constant_op.constant(1.0)
        with self.assertRaises(ValueError):
            linalg_ops.eig(scalar)
        vector = constant_op.constant([1.0, 2.0])
        with self.assertRaises(ValueError):
            linalg_ops.eig(vector)

    @test_util.run_deprecated_v1
    def testConcurrentExecutesWithoutError(self):
        if False:
            return 10
        all_ops = []
        with self.session():
            for compute_v_ in (True, False):
                matrix1 = random_ops.random_normal([5, 5], seed=42)
                matrix2 = random_ops.random_normal([5, 5], seed=42)
                if compute_v_:
                    (e1, v1) = linalg_ops.eig(matrix1)
                    (e2, v2) = linalg_ops.eig(matrix2)
                    all_ops += [e1, v1, e2, v2]
                else:
                    e1 = linalg_ops.eigvals(matrix1)
                    e2 = linalg_ops.eigvals(matrix2)
                    all_ops += [e1, e2]
            val = self.evaluate(all_ops)
            self.assertAllEqual(val[0], val[2])
            self.assertAllClose(val[2], val[4])
            self.assertAllEqual(val[4], val[5])
            self.assertAllEqual(val[1], val[3])

    def testMatrixThatFailsWhenFlushingDenormsToZero(self):
        if False:
            i = 10
            return i + 15
        matrix = np.genfromtxt(test.test_src_dir_path('python/kernel_tests/linalg/testdata/self_adjoint_eig_fail_if_denorms_flushed.txt')).astype(np.float32)
        self.assertEqual(matrix.shape, (32, 32))
        matrix_tensor = constant_op.constant(matrix)
        with self.session() as _:
            (e, v) = self.evaluate(linalg_ops.self_adjoint_eig(matrix_tensor))
            self.assertEqual(e.size, 32)
            self.assertAllClose(np.matmul(v, v.transpose()), np.eye(32, dtype=np.float32), atol=0.002)
            self.assertAllClose(matrix, np.matmul(np.matmul(v, np.diag(e)), v.transpose()))

    def testMismatchedDtypes(self):
        if False:
            print('Hello World!')
        tensor = constant_op.constant([[0, 1], [2, 3]], dtype=dtypes_lib.float32)
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), 'Invalid output dtype'):
            self.evaluate(gen_linalg_ops.eig(input=tensor, Tout=dtypes_lib.complex128, compute_v=True))

def SortEigenValues(e):
    if False:
        print('Hello World!')
    perm = np.argsort(e.real + e.imag, -1)
    return np.take(e, perm, -1)

def SortEigenDecomposition(e, v):
    if False:
        i = 10
        return i + 15
    if v.ndim < 2:
        return (e, v)
    perm = np.argsort(e.real + e.imag, -1)
    return (np.take(e, perm, -1), np.take(v, perm, -1))

def EquilibrateEigenVectorPhases(x, y):
    if False:
        print('Hello World!')
    'Equilibrate the phase of the Eigenvectors in the columns of `x` and `y`.\n\n  Eigenvectors are only unique up to an arbitrary phase. This function rotates x\n  such that it matches y. Precondition: The columns of x and y differ by a\n  multiplicative complex phase factor only.\n\n  Args:\n    x: `np.ndarray` with Eigenvectors\n    y: `np.ndarray` with Eigenvectors\n\n  Returns:\n    `np.ndarray` containing an equilibrated version of x.\n  '
    phases = np.sum(np.conj(x) * y, -2, keepdims=True)
    phases /= np.abs(phases)
    return phases * x

def _GetEigTest(dtype_, shape_, compute_v_):
    if False:
        while True:
            i = 10

    def CompareEigenVectors(self, x, y, tol):
        if False:
            return 10
        x = EquilibrateEigenVectorPhases(x, y)
        self.assertAllClose(x, y, atol=tol)

    def CompareEigenDecompositions(self, x_e, x_v, y_e, y_v, tol):
        if False:
            print('Hello World!')
        num_batches = int(np.prod(x_e.shape[:-1]))
        n = x_e.shape[-1]
        x_e = np.reshape(x_e, [num_batches] + [n])
        x_v = np.reshape(x_v, [num_batches] + [n, n])
        y_e = np.reshape(y_e, [num_batches] + [n])
        y_v = np.reshape(y_v, [num_batches] + [n, n])
        for i in range(num_batches):
            (x_ei, x_vi) = SortEigenDecomposition(x_e[i, :], x_v[i, :, :])
            (y_ei, y_vi) = SortEigenDecomposition(y_e[i, :], y_v[i, :, :])
            self.assertAllClose(x_ei, y_ei, atol=tol, rtol=tol)
            CompareEigenVectors(self, x_vi, y_vi, tol)

    def Test(self):
        if False:
            print('Hello World!')
        np.random.seed(1)
        n = shape_[-1]
        batch_shape = shape_[:-2]
        np_dtype = dtype_.as_numpy_dtype

        def RandomInput():
            if False:
                for i in range(10):
                    print('nop')
            a = np.random.uniform(low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(np_dtype)
            if dtype_.is_complex:
                a += 1j * np.random.uniform(low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(np_dtype)
            a = np.tile(a, batch_shape + (1, 1))
            return a
        if dtype_ in (dtypes_lib.float32, dtypes_lib.complex64):
            atol = 0.0001
        else:
            atol = 1e-12
        a = RandomInput()
        (np_e, np_v) = np.linalg.eig(a)
        with self.session():
            if compute_v_:
                (tf_e, tf_v) = linalg_ops.eig(constant_op.constant(a))
                a_ev = math_ops.matmul(math_ops.matmul(tf_v, array_ops.matrix_diag(tf_e)), linalg_ops.matrix_inverse(tf_v))
                self.assertAllClose(self.evaluate(a_ev), a, atol=atol)
                CompareEigenDecompositions(self, np_e, np_v, self.evaluate(tf_e), self.evaluate(tf_v), atol)
            else:
                tf_e = linalg_ops.eigvals(constant_op.constant(a))
                self.assertAllClose(SortEigenValues(np_e), SortEigenValues(self.evaluate(tf_e)), atol=atol)
    return Test

class EigGradTest(test.TestCase):
    pass

def _GetEigGradTest(dtype_, shape_, compute_v_):
    if False:
        while True:
            i = 10

    def Test(self):
        if False:
            return 10
        np.random.seed(1)
        n = shape_[-1]
        batch_shape = shape_[:-2]
        np_dtype = dtype_.as_numpy_dtype

        def RandomInput():
            if False:
                return 10
            a = np.random.uniform(low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(np_dtype)
            if dtype_.is_complex:
                a += 1j * np.random.uniform(low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(np_dtype)
            a = np.tile(a, batch_shape + (1, 1))
            return a
        epsilon = np.finfo(np_dtype).eps
        delta = 0.1 * epsilon ** (1.0 / 3.0)
        _ = RandomInput()
        if dtype_ in (dtypes_lib.float32, dtypes_lib.complex64):
            tol = 0.01
        else:
            tol = 1e-07
        with self.session():

            def Compute(x):
                if False:
                    for i in range(10):
                        print('nop')
                (e, v) = linalg_ops.eig(x)
                b_dims = len(e.shape) - 1
                idx = sort_ops.argsort(math_ops.real(e) + math_ops.imag(e), axis=-1)
                e = array_ops.gather(e, idx, batch_dims=b_dims)
                v = array_ops.gather(v, idx, batch_dims=b_dims)
                top_rows = v[..., 0:1, :]
                angle = -math_ops.angle(top_rows)
                phase = math_ops.complex(math_ops.cos(angle), math_ops.sin(angle))
                v *= phase
                return (e, v)
            if compute_v_:
                funcs = [lambda x: Compute(x)[0], lambda x: Compute(x)[1]]
            else:
                funcs = [linalg_ops.eigvals]
            for f in funcs:
                (theoretical, numerical) = gradient_checker_v2.compute_gradient(f, [RandomInput()], delta=delta)
                self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)
    return Test
if __name__ == '__main__':
    dtypes_to_test = [dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.complex64, dtypes_lib.complex128]
    for compute_v in (True, False):
        for dtype in dtypes_to_test:
            for size in (1, 2, 5, 10):
                for batch_dims in [(), (3,)] + [(3, 2)] * (max(size, size) < 10):
                    shape = batch_dims + (size, size)
                    name = '%s_%s_%s' % (dtype.name, '_'.join(map(str, shape)), compute_v)
                    _AddTest(EigTest, 'Eig', name, _GetEigTest(dtype, shape, compute_v))
                    if dtype not in [dtypes_lib.float32, dtypes_lib.float64]:
                        _AddTest(EigGradTest, 'EigGrad', name, _GetEigGradTest(dtype, shape, compute_v))
    test.main()