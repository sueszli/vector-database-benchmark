"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import warnings
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.wraps import hermitian_wrap, nonneg_wrap, nonpos_wrap, psd_wrap, skew_symmetric_wrap, symmetric_wrap
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.utilities.linalg import gershgorin_psd_check

class TestExpressions(BaseTest):
    """ Unit tests for the expression/expression module. """

    def setUp(self) -> None:
        if False:
            return 10
        self.a = Variable(name='a')
        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')
        self.A = Variable((2, 2), name='A')
        self.B = Variable((2, 2), name='B')
        self.C = Variable((3, 2), name='C')
        self.intf = intf.DEFAULT_INTF

    def test_variable(self) -> None:
        if False:
            while True:
                i = 10
        x = Variable(2)
        y = Variable(2)
        assert y.name() != x.name()
        x = Variable(2, name='x')
        y = Variable()
        self.assertEqual(x.name(), 'x')
        self.assertEqual(x.shape, (2,))
        self.assertEqual(y.shape, tuple())
        self.assertEqual(x.curvature, s.AFFINE)
        self.assertEqual(repr(self.x), 'Variable((2,), x)')
        self.assertEqual(repr(self.A), 'Variable((2, 2), A)')
        self.assertEqual(repr(cp.Variable(name='x', nonneg=True)), 'Variable((), x, nonneg=True)')
        self.assertTrue(repr(cp.Variable()).startswith('Variable((), var'))
        self.assertEqual(cp.Variable(shape=[2], integer=True).shape, (2,))
        with self.assertRaises(Exception) as cm:
            Variable((2, 2), diag=True, symmetric=True)
        self.assertEqual(str(cm.exception), 'Cannot set more than one special attribute in Variable.')
        with self.assertRaises(Exception) as cm:
            Variable((2, 0))
        self.assertEqual(str(cm.exception), 'Invalid dimensions (2, 0).')
        with self.assertRaises(Exception) as cm:
            Variable((2, 0.5))
        self.assertEqual(str(cm.exception), 'Invalid dimensions (2, 0.5).')
        with self.assertRaises(Exception) as cm:
            Variable(2, 1)
        self.assertEqual(str(cm.exception), 'Variable name 1 must be a string.')

    def test_assign_var_value(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test assigning a value to a variable.\n        '
        a = Variable()
        a.value = 1
        self.assertEqual(a.value, 1)
        with self.assertRaises(Exception) as cm:
            a.value = [2, 1]
        self.assertEqual(str(cm.exception), 'Invalid dimensions (2,) for Variable value.')
        a.value = 1
        a.value = None
        assert a.value is None
        x = Variable(2)
        x.value = [2, 1]
        self.assertItemsAlmostEqual(x.value, [2, 1])
        A = Variable((3, 2))
        A.value = np.ones((3, 2))
        self.assertItemsAlmostEqual(A.value, np.ones((3, 2)))
        x = Variable(nonneg=True)
        with self.assertRaises(Exception) as cm:
            x.value = -2
        self.assertEqual(str(cm.exception), 'Variable value must be nonnegative.')

    def test_transpose_variable(self) -> None:
        if False:
            while True:
                i = 10
        var = self.a.T
        self.assertEqual(var.name(), 'a')
        self.assertEqual(var.shape, tuple())
        self.a.save_value(2)
        self.assertEqual(var.value, 2)
        var = self.x
        self.assertEqual(var.name(), 'x')
        self.assertEqual(var.shape, (2,))
        x = Variable((2, 1), name='x')
        var = x.T
        self.assertEqual(var.name(), 'x.T')
        self.assertEqual(var.shape, (1, 2))
        x.save_value(np.array([[1, 2]]).T)
        self.assertEqual(var.value[0, 0], 1)
        self.assertEqual(var.value[0, 1], 2)
        var = self.C.T
        self.assertEqual(var.name(), 'C.T')
        self.assertEqual(var.shape, (2, 3))
        index = var[1, 0]
        self.assertEqual(index.name(), 'C.T[1, 0]')
        self.assertEqual(index.shape, tuple())
        var = x.T.T
        self.assertEqual(var.name(), 'x.T.T')
        self.assertEqual(var.shape, (2, 1))

    def test_constants(self) -> None:
        if False:
            print('Hello World!')
        c = Constant(2.0)
        self.assertEqual(c.name(), str(2.0))
        c = Constant(2)
        self.assertEqual(c.value, 2)
        self.assertEqual(c.shape, tuple())
        self.assertEqual(c.curvature, s.CONSTANT)
        self.assertEqual(c.sign, s.NONNEG)
        self.assertEqual(Constant(-2).sign, s.NONPOS)
        self.assertEqual(Constant(0).sign, s.ZERO)
        c = Constant([[2], [2]])
        self.assertEqual(c.shape, (1, 2))
        self.assertEqual(c.sign, s.NONNEG)
        self.assertEqual((-c).sign, s.NONPOS)
        self.assertEqual((0 * c).sign, s.ZERO)
        c = Constant([[2], [-2]])
        self.assertEqual(c.sign, s.UNKNOWN)
        c = Constant(np.zeros((2, 1)))
        self.assertEqual(c.shape, (2, 1))
        c = Constant([1, 2])
        self.assertEqual(c.shape, (2,))
        A = Constant([[1, 1], [1, 1]])
        exp = c.T @ A @ c
        self.assertEqual(exp.sign, s.NONNEG)
        self.assertEqual((c.T @ c).sign, s.NONNEG)
        exp = c.T.T
        self.assertEqual(exp.sign, s.NONNEG)
        exp = c.T @ self.A
        self.assertEqual(exp.sign, s.UNKNOWN)
        self.assertEqual(repr(c), 'Constant(CONSTANT, NONNEGATIVE, (2,))')

    def test_constant_psd_nsd(self):
        if False:
            for i in range(10):
                print('nop')
        n = 5
        np.random.randn(0)
        U = np.random.randn(n, n)
        U = U @ U.T
        (evals, U) = np.linalg.eigh(U)
        v1 = np.array([3, 2, 1, 1e-08, -1])
        P = Constant(U @ np.diag(v1) @ U.T)
        self.assertFalse(P.is_psd())
        self.assertFalse(P.is_nsd())
        v2 = np.array([3, 2, 2, 1e-06, -1])
        P = Constant(U @ np.diag(v2) @ U.T)
        self.assertFalse(P.is_psd())
        self.assertFalse(P.is_nsd())
        v3 = np.array([3, 2, 2, 0.0001, -1e-06])
        P = Constant(U @ np.diag(v3) @ U.T)
        self.assertFalse(P.is_psd())
        self.assertFalse(P.is_nsd())
        v4 = np.array([-1, 3, 0, 0, 0])
        P = Constant(U @ np.diag(v4) @ U.T)
        self.assertFalse(P.is_psd())
        self.assertFalse(P.is_nsd())
        P = Constant(np.array([[1, 2], [2, 1]]))
        x = Variable(shape=(2,))
        expr = cp.quad_form(x, P)
        self.assertFalse(expr.is_dcp())
        self.assertFalse((-expr).is_dcp())
        self.assertFalse(gershgorin_psd_check(P.value, tol=0.99))
        P = Constant(np.array([[2, 1], [1, 2]]))
        self.assertTrue(gershgorin_psd_check(P.value, tol=0.0))
        P = Constant(np.diag(9 * [0.0001] + [-10000.0]))
        self.assertFalse(P.is_psd())
        self.assertFalse(P.is_nsd())
        P = Constant(np.ones(shape=(5, 5)))
        self.assertTrue(P.is_psd())
        self.assertFalse(P.is_nsd())
        P = Constant(sp.eye(10))
        self.assertTrue(gershgorin_psd_check(P.value, s.EIGVAL_TOL))
        self.assertTrue(P.is_psd())
        self.assertTrue((-P).is_nsd())
        Q = -s.EIGVAL_TOL / 2 * P
        self.assertTrue(gershgorin_psd_check(Q.value, s.EIGVAL_TOL))
        Q = -1.1 * s.EIGVAL_TOL * P
        self.assertFalse(gershgorin_psd_check(Q.value, s.EIGVAL_TOL))
        self.assertFalse(Q.is_psd())

    def test_constant_skew_symmetric(self) -> None:
        if False:
            print('Hello World!')
        M1_false = np.eye(3)
        M2_true = np.zeros((3, 3))
        M3_true = np.array([[0, 1], [-1, 0]])
        M4_true = np.array([[0, -1], [1, 0]])
        M5_false = np.array([[0, 1], [1, 0]])
        M6_false = np.array([[1, 1], [-1, 0]])
        M7_false = np.array([[0, 1], [-1.1, 0]])
        C = Constant(M1_false)
        self.assertFalse(C.is_skew_symmetric())
        C = Constant(M2_true)
        self.assertTrue(C.is_skew_symmetric())
        C = Constant(M3_true)
        self.assertTrue(C.is_skew_symmetric())
        C = Constant(M4_true)
        self.assertTrue(C.is_skew_symmetric())
        C = Constant(M5_false)
        self.assertFalse(C.is_skew_symmetric())
        C = Constant(M6_false)
        self.assertFalse(C.is_skew_symmetric())
        C = Constant(M7_false)
        self.assertFalse(C.is_skew_symmetric())
        C = Constant(sp.csc_matrix(M1_false))
        self.assertFalse(C.is_skew_symmetric())
        C = Constant(sp.csc_matrix(M2_true))
        self.assertTrue(C.is_skew_symmetric())
        C = Constant(sp.csc_matrix(M4_true))
        self.assertTrue(C.is_skew_symmetric())
        C = Constant(sp.csc_matrix(M5_false))
        self.assertFalse(C.is_skew_symmetric())
        C = Constant(sp.csc_matrix(M6_false))
        self.assertFalse(C.is_skew_symmetric())
        C = Constant(sp.csc_matrix(M7_false))
        self.assertFalse(C.is_skew_symmetric())
        C = Constant(1j * M2_true)
        self.assertFalse(C.is_skew_symmetric())
        C = Constant(1j * M3_true)
        self.assertFalse(C.is_skew_symmetric())
        C = Constant(1j * M4_true)
        self.assertFalse(C.is_skew_symmetric())
        pass

    def test_1D_array(self) -> None:
        if False:
            while True:
                i = 10
        'Test NumPy 1D arrays as constants.\n        '
        c = np.array([1, 2])
        p = Parameter(2)
        p.value = [1, 1]
        self.assertEqual((c @ p).value, 3)
        self.assertEqual((c @ self.x).shape, tuple())

    def test_parameters_successes(self) -> None:
        if False:
            while True:
                i = 10
        p = Parameter(name='p')
        self.assertEqual(p.name(), 'p')
        self.assertEqual(p.shape, tuple())
        val = -np.ones((4, 3))
        val[0, 0] = 2
        p = Parameter((4, 3))
        p.value = val
        p = Parameter(value=10)
        self.assertEqual(p.value, 10)
        p.value = 10
        p.value = None
        self.assertEqual(p.value, None)
        p = Parameter((4, 3), nonpos=True)
        self.assertEqual(repr(p), 'Parameter((4, 3), nonpos=True)')
        p = Parameter((2, 2), diag=True)
        p.value = sp.csc_matrix(np.eye(2))
        self.assertItemsAlmostEqual(p.value.todense(), np.eye(2), places=10)

    def test_psd_nsd_parameters(self) -> None:
        if False:
            return 10
        np.random.seed(42)
        a = np.random.normal(size=(100, 95))
        a2 = a.dot(a.T)
        p = Parameter((100, 100), PSD=True)
        p.value = a2
        self.assertItemsAlmostEqual(p.value, a2, places=10)
        (m, n) = (10, 5)
        A = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        A = np.dot(A.T.conj(), A)
        A = np.vstack([np.hstack([np.real(A), -np.imag(A)]), np.hstack([np.imag(A), np.real(A)])])
        p = Parameter(shape=(2 * n, 2 * n), PSD=True)
        p.value = A
        self.assertItemsAlmostEqual(p.value, A)
        p = Parameter(shape=(2, 2), PSD=True)
        self.assertTrue((2 * p).is_psd())
        self.assertTrue((p + p).is_psd())
        self.assertTrue((-p).is_nsd())
        self.assertTrue((-2 * -p).is_psd())
        n = 5
        P = Parameter(shape=(n, n), PSD=True)
        N = Parameter(shape=(n, n), NSD=True)
        np.random.randn(0)
        U = np.random.randn(n, n)
        U = U @ U.T
        (evals, U) = np.linalg.eigh(U)
        v1 = np.array([3, 2, 1, 1e-08, -1])
        v2 = np.array([3, 2, 2, 1e-06, -1])
        v3 = np.array([3, 2, 2, 0.0001, -1e-06])
        v4 = np.array([-1, 3, 0, 0, 0])
        vs = [v1, v2, v3, v4]
        for vi in vs:
            with self.assertRaises(Exception) as cm:
                P.value = U @ np.diag(vi) @ U.T
            self.assertEqual(str(cm.exception), 'Parameter value must be positive semidefinite.')
            with self.assertRaises(Exception) as cm:
                N.value = -U @ np.diag(vi) @ U.T
            self.assertEqual(str(cm.exception), 'Parameter value must be negative semidefinite.')

    def test_parameters_failures(self) -> None:
        if False:
            return 10
        p = Parameter(name='p')
        self.assertEqual(p.name(), 'p')
        self.assertEqual(p.shape, tuple())
        p = Parameter((4, 3), nonneg=True)
        with self.assertRaises(Exception) as cm:
            p.value = 1
        self.assertEqual(str(cm.exception), 'Invalid dimensions () for Parameter value.')
        val = -np.ones((4, 3))
        val[0, 0] = 2
        p = Parameter((4, 3), nonneg=True)
        with self.assertRaises(Exception) as cm:
            p.value = val
        self.assertEqual(str(cm.exception), 'Parameter value must be nonnegative.')
        p = Parameter((4, 3), nonpos=True)
        with self.assertRaises(Exception) as cm:
            p.value = val
        self.assertEqual(str(cm.exception), 'Parameter value must be nonpositive.')
        with self.assertRaises(Exception) as cm:
            p = Parameter(2, 1, nonpos=True, value=[2, 1])
        self.assertEqual(str(cm.exception), 'Parameter value must be nonpositive.')
        with self.assertRaises(Exception) as cm:
            p = Parameter((4, 3), nonneg=True, value=[1, 2])
        self.assertEqual(str(cm.exception), 'Invalid dimensions (2,) for Parameter value.')
        with self.assertRaises(Exception) as cm:
            p = Parameter((2, 2), diag=True, symmetric=True)
        self.assertEqual(str(cm.exception), 'Cannot set more than one special attribute in Parameter.')
        with self.assertRaises(Exception) as cm:
            p = Parameter((2, 2), boolean=True, value=[[1, 1], [1, -1]])
        self.assertEqual(str(cm.exception), 'Parameter value must be boolean.')
        with self.assertRaises(Exception) as cm:
            p = Parameter((2, 2), integer=True, value=[[1, 1.5], [1, -1]])
        self.assertEqual(str(cm.exception), 'Parameter value must be integer.')
        with self.assertRaises(Exception) as cm:
            p = Parameter((2, 2), diag=True, value=[[1, 1], [1, -1]])
        self.assertEqual(str(cm.exception), 'Parameter value must be diagonal.')
        with self.assertRaises(Exception) as cm:
            p = Parameter((2, 2), symmetric=True, value=[[1, 1], [-1, -1]])
        self.assertEqual(str(cm.exception), 'Parameter value must be symmetric.')

    def test_symmetric(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test symmetric variables.\n        '
        with self.assertRaises(Exception) as cm:
            v = Variable((4, 3), symmetric=True)
        self.assertEqual(str(cm.exception), 'Invalid dimensions (4, 3). Must be a square matrix.')
        v = Variable((2, 2), symmetric=True)
        assert v.is_symmetric()
        v = Variable((2, 2), PSD=True)
        assert v.is_symmetric()
        v = Variable((2, 2), NSD=True)
        assert v.is_symmetric()
        v = Variable((2, 2), diag=True)
        assert v.is_symmetric()
        assert self.a.is_symmetric()
        assert not self.A.is_symmetric()
        v = Variable((2, 2), symmetric=True)
        expr = v + v
        assert expr.is_symmetric()
        expr = -v
        assert expr.is_symmetric()
        expr = v.T
        assert expr.is_symmetric()
        expr = cp.real(v)
        assert expr.is_symmetric()
        expr = cp.imag(v)
        assert expr.is_symmetric()
        expr = cp.conj(v)
        assert expr.is_symmetric()
        expr = cp.promote(Variable(), (2, 2))
        assert expr.is_symmetric()

    def test_hermitian(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test Hermitian variables.\n        '
        with self.assertRaises(Exception) as cm:
            v = Variable((4, 3), hermitian=True)
        self.assertEqual(str(cm.exception), 'Invalid dimensions (4, 3). Must be a square matrix.')
        v = Variable((2, 2), hermitian=True)
        assert v.is_hermitian()
        v = Variable((2, 2), diag=True)
        assert v.is_hermitian()
        v = Variable((2, 2), hermitian=True)
        expr = v + v
        assert expr.is_hermitian()
        expr = -v
        assert expr.is_hermitian()
        expr = v.T
        assert expr.is_hermitian()
        expr = cp.real(v)
        assert expr.is_hermitian()
        expr = cp.imag(v)
        assert expr.is_hermitian()
        expr = cp.conj(v)
        assert expr.is_hermitian()
        expr = cp.promote(Variable(), (2, 2))
        assert expr.is_hermitian()

    def test_round_attr(self) -> None:
        if False:
            while True:
                i = 10
        'Test rounding for attributes.\n        '
        v = Variable(1, nonpos=True)
        self.assertAlmostEqual(v.project(1), 0)
        v = Variable(2, nonpos=True)
        self.assertItemsAlmostEqual(v.project(np.array([1, -1])), [0, -1])
        v = Variable(1, nonneg=True)
        self.assertAlmostEqual(v.project(-1), 0)
        v = Variable(2, nonneg=True)
        self.assertItemsAlmostEqual(v.project(np.array([1, -1])), [1, 0])
        v = Variable((2, 2), boolean=True)
        self.assertItemsAlmostEqual(v.project(np.array([[1, -1], [1, 0]]).T), [1, 0, 1, 0])
        v = Variable((2, 2), integer=True)
        self.assertItemsAlmostEqual(v.project(np.array([[1, -1.6], [1, 0]]).T), [1, -2, 1, 0])
        v = Variable((2, 2), symmetric=True)
        self.assertItemsAlmostEqual(v.project(np.array([[1, -1], [1, 0]])), [1, 0, 0, 0])
        v = Variable((2, 2), PSD=True)
        self.assertItemsAlmostEqual(v.project(np.array([[1, -1], [1, -1]])), [1, 0, 0, 0])
        v = Variable((2, 2), NSD=True)
        self.assertItemsAlmostEqual(v.project(np.array([[1, -1], [1, -1]])), [0, 0, 0, -1])
        v = Variable((2, 2), diag=True)
        self.assertItemsAlmostEqual(v.project(np.array([[1, -1], [1, 0]])).todense(), [1, 0, 0, 0])
        v = Variable((2, 2), hermitian=True)
        self.assertItemsAlmostEqual(v.project(np.array([[1, -1j], [1, 0]])), [1, 0.5 + 0.5j, 0.5 - 0.5j, 0])
        A = Constant(np.array([[1.0]]))
        self.assertEqual(A.is_psd(), True)
        self.assertEqual(A.is_nsd(), False)
        A = Constant(np.array([[-1.0]]))
        self.assertEqual(A.is_psd(), False)
        self.assertEqual(A.is_nsd(), True)
        A = Constant(np.array([[0.0]]))
        self.assertEqual(A.is_psd(), True)
        self.assertEqual(A.is_nsd(), True)

    def test_add_expression(self) -> None:
        if False:
            return 10
        c = Constant([2, 2])
        exp = self.x + c
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.sign, s.UNKNOWN)
        self.assertEqual(exp.shape, (2,))
        z = Variable(2, name='z')
        exp = exp + z + self.x
        with self.assertRaises(ValueError):
            self.x + self.y
        exp = self.A + self.B
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.shape, (2, 2))
        with self.assertRaises(ValueError):
            self.A + self.C
        with self.assertRaises(ValueError):
            AddExpression([self.A, self.C])
        exp = self.x + c + self.x
        self.assertEqual(len(exp.args), 3)
        self.assertEqual(repr(exp), 'Expression(AFFINE, UNKNOWN, (2,))')

    def test_sub_expression(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        c = Constant([2, 2])
        exp = self.x - c
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.sign, s.UNKNOWN)
        self.assertEqual(exp.shape, (2,))
        z = Variable(2, name='z')
        exp = exp - z - self.x
        with self.assertRaises(ValueError):
            self.x - self.y
        exp = self.A - self.B
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.shape, (2, 2))
        with self.assertRaises(ValueError):
            self.A - self.C
        self.assertEqual(repr(self.x - c), 'Expression(AFFINE, UNKNOWN, (2,))')

    def test_mul_expression(self) -> None:
        if False:
            return 10
        c = Constant([[2], [2]])
        exp = c @ self.x
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual((c[0] @ self.x).sign, s.UNKNOWN)
        self.assertEqual(exp.shape, (1,))
        with self.assertRaises(ValueError):
            [2, 2, 3] @ self.x
        with self.assertRaises(ValueError):
            Constant([[2, 1], [2, 2]]) @ self.C
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            q = self.A @ self.B
            self.assertTrue(q.is_quadratic())
        T = Constant([[1, 2, 3], [3, 5, 5]])
        exp = (T + T) @ self.B
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.shape, (3, 2))
        c = Constant([[2], [2], [-2]])
        exp = [[1], [2]] + c @ self.C
        self.assertEqual(exp.sign, s.UNKNOWN)
        c = Constant([[2], [2]])
        with warnings.catch_warnings(record=True) as w:
            c * self.x
            self.assertEqual(2, len(w))
            self.assertEqual(w[0].category, UserWarning)
            self.assertEqual(w[1].category, DeprecationWarning)
            c * self.x
            self.assertEqual(4, len(w))
            self.assertEqual(w[2].category, UserWarning)
            self.assertEqual(w[3].category, DeprecationWarning)
            warnings.simplefilter('ignore', DeprecationWarning)
            c * self.x
            self.assertEqual(5, len(w))
            warnings.simplefilter('ignore', UserWarning)
            c * self.x
            self.assertEqual(len(w), 5)
            warnings.simplefilter('error', UserWarning)
            with self.assertRaises(UserWarning):
                c * self.x

    def test_matmul_expression(self) -> None:
        if False:
            print('Hello World!')
        'Test matmul function, corresponding to .__matmul__( operator.\n        '
        c = Constant([[2], [2]])
        exp = c.__matmul__(self.x)
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.sign, s.UNKNOWN)
        self.assertEqual(exp.shape, (1,))
        with self.assertRaises(Exception) as cm:
            self.x.__matmul__(2)
        self.assertEqual(str(cm.exception), "Scalar operands are not allowed, use '*' instead")
        with self.assertRaises(ValueError) as cm:
            self.x.__matmul__(np.array([2, 2, 3]))
        with self.assertRaises(Exception) as cm:
            Constant([[2, 1], [2, 2]]).__matmul__(self.C)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            q = self.A.__matmul__(self.B)
            self.assertTrue(q.is_quadratic())
        T = Constant([[1, 2, 3], [3, 5, 5]])
        exp = (T + T).__matmul__(self.B)
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.shape, (3, 2))
        c = Constant([[2], [2], [-2]])
        exp = [[1], [2]] + c.__matmul__(self.C)
        self.assertEqual(exp.sign, s.UNKNOWN)
        a = Parameter((1,))
        x = Variable(shape=(1,))
        expr = a.__matmul__(x)
        self.assertEqual(expr.shape, ())
        a = Parameter((1,))
        x = Variable(shape=(1,))
        expr = a.__matmul__(x)
        self.assertEqual(expr.shape, ())
        A = Parameter((4, 4))
        z = Variable((4, 1))
        expr = A.__matmul__(z)
        self.assertEqual(expr.shape, (4, 1))
        v = Variable((1, 1))
        col_scalar = Parameter((1, 1))
        assert v.shape == col_scalar.shape == col_scalar.T.shape

    def test_div_expression(self) -> None:
        if False:
            print('Hello World!')
        exp = self.x / 2
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.sign, s.UNKNOWN)
        self.assertEqual(exp.shape, (2,))
        with self.assertRaises(Exception) as cm:
            self.x / [2, 2, 3]
        print(cm.exception)
        self.assertRegex(str(cm.exception), 'Incompatible shapes for division.*')
        c = Constant([3.0, 4.0, 12.0])
        self.assertItemsAlmostEqual((c / Constant([1.0, 2.0, 3.0])).value, np.array([3.0, 2.0, 4.0]))
        c = Constant(2)
        exp = c / (3 - 5)
        self.assertEqual(exp.curvature, s.CONSTANT)
        self.assertEqual(exp.shape, tuple())
        self.assertEqual(exp.sign, s.NONPOS)
        p = Parameter(nonneg=True)
        exp = 2 / p
        p.value = 2
        self.assertEqual(exp.value, 1)
        rho = Parameter(nonneg=True)
        rho.value = 1
        self.assertEqual(rho.sign, s.NONNEG)
        self.assertEqual(Constant(2).sign, s.NONNEG)
        self.assertEqual((Constant(2) / Constant(2)).sign, s.NONNEG)
        self.assertEqual((Constant(2) * rho).sign, s.NONNEG)
        self.assertEqual((rho / 2).sign, s.NONNEG)
        x = cp.Variable((3, 3))
        c = np.arange(1, 4)[:, None]
        expr = x / c
        self.assertEqual((3, 3), expr.shape)
        x.value = np.ones((3, 3))
        A = np.ones((3, 3)) / c
        self.assertItemsAlmostEqual(A, expr.value)
        with self.assertRaises(Exception) as cm:
            x / c[:, 0]
        print(cm.exception)
        self.assertRegex(str(cm.exception), 'Incompatible shapes for division.*')

    def test_neg_expression(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        exp = -self.x
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.shape, (2,))
        assert exp.is_affine()
        self.assertEqual(exp.sign, s.UNKNOWN)
        assert not exp.is_nonneg()
        self.assertEqual(exp.shape, self.x.shape)
        exp = -self.C
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.shape, (3, 2))

    def test_scalar_const_promotion(self) -> None:
        if False:
            while True:
                i = 10
        exp = self.x + 2
        self.assertEqual(exp.curvature, s.AFFINE)
        assert exp.is_affine()
        self.assertEqual(exp.sign, s.UNKNOWN)
        assert not exp.is_nonpos()
        self.assertEqual(exp.shape, (2,))
        self.assertEqual((4 - self.x).shape, (2,))
        self.assertEqual((4 * self.x).shape, (2,))
        self.assertEqual((4 <= self.x).shape, (2,))
        self.assertEqual((4 == self.x).shape, (2,))
        self.assertEqual((self.x >= 4).shape, (2,))
        exp = self.A + 2 + 4
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual((3 * self.A).shape, (2, 2))
        self.assertEqual(exp.shape, (2, 2))

    def test_index_expression(self) -> None:
        if False:
            return 10
        exp = self.x[1]
        self.assertEqual(exp.curvature, s.AFFINE)
        assert exp.is_affine()
        self.assertEqual(exp.shape, tuple())
        self.assertEqual(exp.value, None)
        exp = self.x[1].T
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.shape, tuple())
        with self.assertRaises(Exception) as cm:
            self.x[2, 0]
        self.assertEqual(str(cm.exception), 'Too many indices for expression.')
        with self.assertRaises(Exception) as cm:
            self.x[2]
        self.assertEqual(str(cm.exception), 'Index 2 is out of bounds for axis 0 with size 2.')
        exp = self.C[0:2, 1]
        self.assertEqual(exp.shape, (2,))
        exp = self.C[0:, 0:2]
        self.assertEqual(exp.shape, (3, 2))
        exp = self.C[0::2, 0::2]
        self.assertEqual(exp.shape, (2, 1))
        exp = self.C[:3, :1:2]
        self.assertEqual(exp.shape, (3, 1))
        exp = self.C[0:, 0]
        self.assertEqual(exp.shape, (3,))
        c = Constant([[1, -2], [0, 4]])
        exp = c[1, 1]
        self.assertEqual(exp.curvature, s.CONSTANT)
        self.assertEqual(exp.sign, s.UNKNOWN)
        self.assertEqual(c[0, 1].sign, s.UNKNOWN)
        self.assertEqual(c[1, 0].sign, s.UNKNOWN)
        self.assertEqual(exp.shape, tuple())
        self.assertEqual(exp.value, 4)
        c = Constant([[1, -2, 3], [0, 4, 5], [7, 8, 9]])
        exp = c[0:3, 0:4:2]
        self.assertEqual(exp.curvature, s.CONSTANT)
        assert exp.is_constant()
        self.assertEqual(exp.shape, (3, 2))
        self.assertEqual(exp[0, 1].value, 7)
        exp = self.C.T[0:2, 1:2]
        self.assertEqual(exp.shape, (2, 1))
        exp = (self.x + self.z)[1]
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.sign, s.UNKNOWN)
        self.assertEqual(exp.shape, tuple())
        exp = (self.x + self.a)[1:2]
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.shape, (1,))
        exp = (self.x - self.z)[1:2]
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.shape, (1,))
        exp = (self.x - self.a)[1]
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.shape, tuple())
        exp = (-self.x)[1]
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.shape, tuple())
        c = Constant([[1, 2], [3, 4]])
        exp = (c @ self.x)[1]
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.shape, tuple())
        c = Constant([[1, 2], [3, 4]])
        exp = (c * self.a)[1, 0:1]
        self.assertEqual(exp.curvature, s.AFFINE)
        self.assertEqual(exp.shape, (1,))

    def test_none_idx(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test None as index.\n        '
        expr = self.a[None, None]
        self.assertEqual(expr.shape, (1, 1))
        expr = self.x[:, None]
        self.assertEqual(expr.shape, (2, 1))
        expr = self.x[None, :]
        self.assertEqual(expr.shape, (1, 2))
        expr = Constant([1, 2])[None, :]
        self.assertEqual(expr.shape, (1, 2))
        self.assertItemsAlmostEqual(expr.value, [1, 2])

    def test_out_of_bounds(self) -> None:
        if False:
            print('Hello World!')
        'Test out of bounds indices.\n        '
        with self.assertRaises(Exception) as cm:
            self.x[100]
        self.assertEqual(str(cm.exception), 'Index 100 is out of bounds for axis 0 with size 2.')
        with self.assertRaises(Exception) as cm:
            self.x[-100]
        self.assertEqual(str(cm.exception), 'Index -100 is out of bounds for axis 0 with size 2.')
        exp = self.x[:-100]
        self.assertEqual(exp.size, 0)
        self.assertItemsAlmostEqual(exp.value, np.array([]))
        exp = self.C[100:2]
        self.assertEqual(exp.shape, (0, 2))
        exp = self.C[:, -199:2]
        self.assertEqual(exp.shape, (3, 2))
        exp = self.C[:, -199:-3]
        self.assertEqual(exp.shape, (3, 0))

    def test_float_is_invalid_index(self) -> None:
        if False:
            while True:
                i = 10
        with self.assertRaises(IndexError) as cm:
            self.x[1.0]
        self.assertEqual(str(cm.exception), 'float is an invalid index type.')
        with self.assertRaises(IndexError) as cm:
            self.x[1.0,]
        self.assertEqual(str(cm.exception), 'float is an invalid index type.')
        with self.assertRaises(IndexError) as cm:
            self.C[:2.0:40]
        self.assertEqual(str(cm.exception), 'float is an invalid index type.')
        with self.assertRaises(IndexError) as cm:
            self.x[np.array([1.0, 2.0])]
        self.assertEqual(str(cm.exception), 'arrays used as indices must be of integer (or boolean) type')

    def test_neg_indices(self) -> None:
        if False:
            print('Hello World!')
        'Test negative indices.\n        '
        c = Constant([[1, 2], [3, 4]])
        exp = c[-1, -1]
        self.assertEqual(exp.value, 4)
        self.assertEqual(exp.shape, tuple())
        self.assertEqual(exp.curvature, s.CONSTANT)
        c = Constant([1, 2, 3, 4])
        exp = c[1:-1]
        self.assertItemsAlmostEqual(exp.value, [2, 3])
        self.assertEqual(exp.shape, (2,))
        self.assertEqual(exp.curvature, s.CONSTANT)
        c = Constant([1, 2, 3, 4])
        exp = c[::-1]
        self.assertItemsAlmostEqual(exp.value, [4, 3, 2, 1])
        self.assertEqual(exp.shape, (4,))
        self.assertEqual(exp.curvature, s.CONSTANT)
        x = Variable(4)
        self.assertEqual(x[::-1].shape, (4,))
        Problem(Minimize(0), [x[::-1] == c]).solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(x.value, [4, 3, 2, 1])
        x = Variable(2)
        self.assertEqual(x[::-1].shape, (2,))
        x = Variable(100, name='x')
        self.assertEqual('x[0:99]', str(x[:-1]))
        c = Constant([[1, 2], [3, 4]])
        expr = c[0, 2:0:-1]
        self.assertEqual(expr.shape, (1,))
        self.assertAlmostEqual(expr.value, 3)
        expr = c[0, 2::-1]
        self.assertEqual(expr.shape, (2,))
        self.assertItemsAlmostEqual(expr.value, [3, 1])

    def test_logical_indices(self) -> None:
        if False:
            while True:
                i = 10
        'Test indexing with boolean arrays.\n        '
        A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        C = Constant(A)
        expr = C[A <= 2]
        self.assertEqual(expr.shape, (2,))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertItemsAlmostEqual(A[A <= 2], expr.value)
        expr = C[A % 2 == 0]
        self.assertEqual(expr.shape, (6,))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertItemsAlmostEqual(A[A % 2 == 0], expr.value)
        expr = C[np.array([True, False, True]), 3]
        self.assertEqual(expr.shape, (2,))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertItemsAlmostEqual(A[np.array([True, False, True]), 3], expr.value)
        expr = C[1, np.array([True, False, False, True])]
        self.assertEqual(expr.shape, (2,))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertItemsAlmostEqual(A[1, np.array([True, False, False, True])], expr.value)
        expr = C[np.array([True, True, True]), 1:3]
        self.assertEqual(expr.shape, (3, 2))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertItemsAlmostEqual(A[np.array([True, True, True]), 1:3], expr.value)
        expr = C[1:-1, np.array([True, False, True, True])]
        self.assertEqual(expr.shape, (1, 3))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertItemsAlmostEqual(A[1:-1, np.array([True, False, True, True])], expr.value)
        expr = C[np.array([True, True, True]), np.array([True, False, True, True])]
        self.assertEqual(expr.shape, (3,))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertItemsAlmostEqual(A[np.array([True, True, True]), np.array([True, False, True, True])], expr.value)

    def test_selector_list_indices(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test indexing with lists/ndarrays of indices.\n        '
        A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        C = Constant(A)
        expr = C[[1, 2]]
        self.assertEqual(expr.shape, (2, 4))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertItemsAlmostEqual(A[[1, 2]], expr.value)
        expr = C[[0, 2], 3]
        self.assertEqual(expr.shape, (2,))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertItemsAlmostEqual(A[[0, 2], 3], expr.value)
        expr = C[1, [0, 2]]
        self.assertEqual(expr.shape, (2,))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertItemsAlmostEqual(A[1, [0, 2]], expr.value)
        expr = C[[0, 2], 1:3]
        self.assertEqual(expr.shape, (2, 2))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertItemsAlmostEqual(A[[0, 2], 1:3], expr.value)
        expr = C[1:-1, [0, 2]]
        self.assertEqual(expr.shape, (1, 2))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertItemsAlmostEqual(A[1:-1, [0, 2]], expr.value)
        expr = C[[0, 1], [1, 3]]
        self.assertEqual(expr.shape, (2,))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertItemsAlmostEqual(A[[0, 1], [1, 3]], expr.value)
        expr = C[np.array([0, 1]), [1, 3]]
        self.assertEqual(expr.shape, (2,))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertItemsAlmostEqual(A[np.array([0, 1]), [1, 3]], expr.value)
        expr = C[np.array([0, 1]), np.array([1, 3])]
        self.assertEqual(expr.shape, (2,))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertItemsAlmostEqual(A[np.array([0, 1]), np.array([1, 3])], expr.value)

    def test_powers(self) -> None:
        if False:
            i = 10
            return i + 15
        exp = self.x ** 2
        self.assertEqual(exp.curvature, s.CONVEX)
        exp = self.x ** 0.5
        self.assertEqual(exp.curvature, s.CONCAVE)
        exp = self.x ** (-1)
        self.assertEqual(exp.curvature, s.CONVEX)

    def test_sum(self) -> None:
        if False:
            return 10
        'Test cvxpy sum function.\n        '
        self.a.value = 1
        expr = cp.sum(self.a)
        self.assertEqual(expr.value, 1)
        self.x.value = [1, 2]
        expr = cp.sum(self.x)
        self.assertEqual(expr.value, 3)

    def test_var_copy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test the copy function for variable types.\n        '
        x = Variable((3, 4), name='x')
        y = x.copy()
        self.assertEqual(y.shape, (3, 4))
        self.assertEqual(y.name(), 'x')
        x = Variable((5, 5), PSD=True, name='x')
        y = x.copy()
        self.assertEqual(y.shape, (5, 5))

    def test_param_copy(self) -> None:
        if False:
            return 10
        'Test the copy function for Parameters.\n        '
        x = Parameter((3, 4), name='x', nonneg=True)
        y = x.copy()
        self.assertEqual(y.shape, (3, 4))
        self.assertEqual(y.name(), 'x')
        self.assertEqual(y.sign, 'NONNEGATIVE')

    def test_constant_copy(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the copy function for Constants.\n        '
        x = Constant(2)
        y = x.copy()
        self.assertEqual(y.shape, tuple())
        self.assertEqual(y.value, 2)

    def test_is_pwl(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test is_pwl()\n        '
        A = np.ones((2, 3))
        b = np.ones(2)
        expr = A @ self.y - b
        self.assertEqual(expr.is_pwl(), True)
        expr = cp.maximum(1, 3 * self.y)
        self.assertEqual(expr.is_pwl(), True)
        expr = cp.abs(self.y)
        self.assertEqual(expr.is_pwl(), True)
        expr = cp.pnorm(3 * self.y, 1)
        self.assertEqual(expr.is_pwl(), True)
        expr = cp.pnorm(3 * self.y ** 2, 1)
        self.assertEqual(expr.is_pwl(), False)

    def test_broadcast_mul(self) -> None:
        if False:
            while True:
                i = 10
        'Test multiply broadcasting.\n        '
        y = Parameter((3, 1))
        z = Variable((1, 3))
        y.value = np.arange(3)[:, None]
        z.value = (np.arange(3) - 1)[None, :]
        expr = cp.multiply(y, z)
        self.assertItemsAlmostEqual(expr.value, y.value * z.value)
        prob = cp.Problem(cp.Minimize(cp.sum(expr)), [z == z.value])
        prob.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(expr.value, y.value * z.value)
        np.random.seed(0)
        (m, n) = (3, 4)
        A = np.random.rand(m, n)
        col_scale = Variable(n)
        with self.assertRaises(ValueError) as cm:
            cp.multiply(A, col_scale)
        self.assertEqual(str(cm.exception), 'Cannot broadcast dimensions  (3, 4) (4,)')
        col_scale = Variable([1, n])
        C = cp.multiply(A, col_scale)
        self.assertEqual(C.shape, (m, n))
        row_scale = Variable([m, 1])
        R = cp.multiply(A, row_scale)
        self.assertEqual(R.shape, (m, n))

    def test_broadcast_add(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test addition broadcasting.\n        '
        y = Parameter((3, 1))
        z = Variable((1, 3))
        y.value = np.arange(3)[:, None]
        z.value = (np.arange(3) - 1)[None, :]
        expr = y + z
        self.assertItemsAlmostEqual(expr.value, y.value + z.value)
        prob = cp.Problem(cp.Minimize(cp.sum(expr)), [z == z.value])
        prob.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(expr.value, y.value + z.value)
        np.random.seed(0)
        (m, n) = (3, 4)
        A = np.random.rand(m, n)
        col_scale = Variable(n)
        with self.assertRaises(ValueError) as cm:
            A + col_scale
        self.assertEqual(str(cm.exception), 'Cannot broadcast dimensions  (3, 4) (4,)')
        col_scale = Variable([1, n])
        C = A + col_scale
        self.assertEqual(C.shape, (m, n))
        row_scale = Variable([m, 1])
        R = A + row_scale
        self.assertEqual(R.shape, (m, n))

    def test_log_log_curvature(self) -> None:
        if False:
            return 10
        'Test that the curvature string is populated for log-log expressions.\n        '
        x = Variable(pos=True)
        monomial = x * x * x
        assert monomial.curvature == s.LOG_LOG_AFFINE
        posynomial = x * x * x + x
        assert posynomial.curvature == s.LOG_LOG_CONVEX
        llcv = 1 / (x * x * x + x)
        assert llcv.curvature == s.LOG_LOG_CONCAVE

    def test_quad_form_matmul(self) -> None:
        if False:
            return 10
        'Test conversion of native x.T @ A @ x into QuadForms.\n        '
        x = Variable(shape=(2,))
        A = Constant([[1, 0], [0, -1]])
        expr = x.T.__matmul__(A).__matmul__(x)
        assert isinstance(expr, cp.QuadForm)
        x = Variable(shape=(2,))
        A = Constant([[1, 0], [0, -1]])
        expr = 1 / 2 * x.T.__matmul__(A).__matmul__(x) + x.T.__matmul__(x)
        assert isinstance(expr.args[0].args[1], cp.QuadForm)
        assert expr.args[0].args[1].args[0] is x
        x = Variable(shape=(2,))
        A = Constant([[1, 0], [0, -1]])
        c = Constant([2, -2])
        expr = 1 / 2 * c.T.__matmul__(c) * x.T.__matmul__(A).__matmul__(x) + x.T.__matmul__(x)
        assert isinstance(expr.args[0].args[1], cp.QuadForm)
        assert expr.args[0].args[1].args[0] is x
        x = Variable(shape=(2,))
        A = Constant(sp.eye(2))
        expr = x.T.__matmul__(A).__matmul__(x)
        assert isinstance(expr, cp.QuadForm)
        x = Variable(shape=(2,))
        A = Constant(np.eye(3))
        with self.assertRaises(Exception) as _:
            x.T.__matmul__(A).__matmul__(x)
        x = cp.Variable(shape=(2,))
        A = cp.Constant([[1, 0], [0, 1]])
        expr = x.T.__matmul__(psd_wrap(A)).__matmul__(x)
        assert isinstance(expr, cp.QuadForm)
        x = cp.Variable(shape=(2,))
        A = cp.Constant([[2, 0, 0], [0, 0, 1]])
        M = cp.Constant([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        b = cp.Constant([1, 2, 3])
        y = A.__matmul__(x) - b
        expr = y.T.__matmul__(M).__matmul__(y)
        assert isinstance(expr, cp.QuadForm)
        assert expr.args[0] is y
        assert expr.args[1] is M
        x = Variable(shape=(2,))
        A = Parameter(shape=(2, 2), symmetric=True)
        expr = x.T.__matmul__(A).__matmul__(x)
        assert isinstance(expr, cp.QuadForm)
        x = Variable(shape=(2,))
        A = Constant([[1, 0], [1, 1]])
        with self.assertRaises(ValueError) as _:
            x.T.__matmul__(A).__matmul__(x)
        x = Variable(shape=(2,))
        A = Constant([[1, 1j], [1j, 1]])
        with self.assertRaises(ValueError) as _:
            x.T.__matmul__(A).__matmul__(x)
        x = Variable(shape=(2,))
        y = Variable(shape=(2,))
        A = Constant([[1, 0], [0, -1]])
        expr = x.T.__matmul__(A).__matmul__(y)
        assert not isinstance(expr, cp.QuadForm)
        x = Variable(shape=(2,))
        M = Variable(shape=(2, 2))
        expr = x.T.__matmul__(M).__matmul__(x)
        assert not isinstance(expr, cp.QuadForm)
        x = Constant([1, 0])
        M = Variable(shape=(2, 2))
        expr = x.T.__matmul__(M).__matmul__(x)
        assert not isinstance(expr, cp.QuadForm)

    def test_matmul_scalars(self) -> None:
        if False:
            print('Hello World!')
        'Test evaluating a matmul that reduces one argument internally to a scalar.\n        '
        x = cp.Variable((2,))
        quad = cp.quad_form(x, np.eye(2))
        a = np.array([2])
        expr = quad * a
        x.value = np.array([1, 2])
        P = np.eye(2)
        true_val = np.transpose(x.value) @ P @ x.value * a
        assert quad.shape == ()
        self.assertEqual(expr.value, true_val)

    def test_wraps(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test wrap classes.'
        x = cp.Variable(2)
        expr = nonneg_wrap(x)
        assert expr.is_nonneg()
        expr = nonpos_wrap(x)
        assert expr.is_nonpos()
        Z = cp.Variable((2, 2))
        U = cp.Variable((2, 2), complex=True)
        expr = psd_wrap(Z)
        assert expr.is_psd()
        assert not expr.is_complex()
        assert expr.is_symmetric()
        assert expr.is_hermitian()
        expr = psd_wrap(U)
        assert expr.is_psd()
        assert expr.is_complex()
        assert not expr.is_symmetric()
        assert expr.is_hermitian()
        expr = symmetric_wrap(Z)
        assert expr.is_symmetric()
        assert expr.is_hermitian()
        expr = skew_symmetric_wrap(Z)
        assert expr.is_skew_symmetric()
        expr = hermitian_wrap(U)
        assert expr.is_hermitian()