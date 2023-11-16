"""
Copyright 2023 CVXPY Developers

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
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest

class TestExpressionMethods(BaseTest):
    """ Unit tests for the atoms module. """

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.a = Variable(name='a')
        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')
        self.A = Variable((2, 2), name='A')
        self.B = Variable((2, 2), name='B')
        self.C = Variable((3, 2), name='C')

    def test_all_expressions(self) -> None:
        if False:
            return 10
        complex_X = Constant(np.array([[1.0, 4.0, 7.0], [2.0, -4.0 + 3j, 3.0], [99.0, -2.0 - 9j, 2.4]]))
        X = Constant(np.array([[1.0, 4.0, 7.0], [2.0, -4.0, 3.0], [99.0, -2.0, 2.4]]))
        for method in ['conj']:
            fn = getattr(cp, method)
            method_fn = getattr(complex_X, method)
            assert fn(complex_X).shape == method_fn().shape
            assert np.allclose(fn(complex_X).value, method_fn().value)
        for method in ['conj', 'trace', 'cumsum', 'max', 'min', 'mean', 'ptp', 'prod', 'sum', 'std', 'var']:
            fn = getattr(cp, method)
            method_fn = getattr(X, method)
            assert fn(X).shape == method_fn().shape
            assert np.allclose(fn(X).value, method_fn().value)
        for method in ['cumsum']:
            for axis in [None, 0, 1]:
                fn = getattr(cp, method)(X, axis)
                method_fn = getattr(X, method)(axis)
                assert fn.shape == method_fn.shape
                assert np.allclose(fn.value, method_fn.value)
        for method in ['max', 'mean', 'min', 'prod', 'ptp', 'sum']:
            for axis in [None, 0, 1]:
                for keepdims in [True, False]:
                    fn = getattr(cp, method)(X, axis, keepdims)
                    method_fn = getattr(X, method)(axis, keepdims=keepdims)
                    assert fn.shape == method_fn.shape
                    assert np.allclose(fn.value, method_fn.value)
        for method in ['std']:
            for axis in [None, 0, 1]:
                for keepdims in [True, False]:
                    for ddof in [0, 1, 2]:
                        fn = getattr(cp, method)(X, axis, keepdims, ddof=ddof)
                        method_fn = getattr(X, method)(axis, keepdims=keepdims, ddof=ddof)
                        assert fn.shape == method_fn.shape
                        assert np.allclose(fn.value, method_fn.value)
        for method in ['var']:
            for ddof in [0, 1, 2]:
                fn = getattr(cp, method)(X, ddof=ddof)
                method_fn = getattr(X, method)(ddof=ddof)
                assert fn.shape == method_fn.shape
                assert np.allclose(fn.value, method_fn.value)

    def test_reshape(self) -> None:
        if False:
            while True:
                i = 10
        'Test the reshape class.\n        '
        expr = self.A.reshape((4, 1))
        self.assertEqual(expr.sign, s.UNKNOWN)
        self.assertEqual(expr.curvature, s.AFFINE)
        self.assertEqual(expr.shape, (4, 1))
        expr = expr.reshape((2, 2))
        self.assertEqual(expr.shape, (2, 2))
        expr = cp.square(self.x).reshape((1, 2))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertEqual(expr.curvature, s.CONVEX)
        self.assertEqual(expr.shape, (1, 2))
        with self.assertRaises(Exception) as cm:
            self.C.reshape((5, 4))
        self.assertEqual(str(cm.exception), 'Invalid reshape dimensions (5, 4).')
        a = np.arange(10)
        A_np = np.reshape(a, (5, 2), order='C')
        A_cp = Constant(a).reshape((5, 2), order='C')
        self.assertItemsAlmostEqual(A_np, A_cp.value)
        X = cp.Variable((5, 2))
        prob = cp.Problem(cp.Minimize(0), [X == A_cp])
        prob.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(A_np, X.value)
        a_np = np.reshape(A_np, 10, order='C')
        a_cp = A_cp.reshape(10, order='C')
        self.assertItemsAlmostEqual(a_np, a_cp.value)
        x = cp.Variable(10)
        prob = cp.Problem(cp.Minimize(0), [x == a_cp])
        prob.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(a_np, x.value)
        b = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        b_reshaped = b.reshape((2, 6), order='C')
        X = cp.Variable(b.shape)
        X_reshaped = X.reshape((2, 6), order='C')
        prob = cp.Problem(cp.Minimize(0), [X_reshaped == b_reshaped])
        prob.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(b_reshaped, X_reshaped.value)
        self.assertItemsAlmostEqual(b, X.value)
        b = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        b_reshaped = b.reshape((2, 6), order='F')
        X = cp.Variable(b.shape)
        X_reshaped = X.reshape((2, 6))
        prob = cp.Problem(cp.Minimize(0), [X_reshaped == b_reshaped])
        prob.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(b_reshaped, X_reshaped.value)
        self.assertItemsAlmostEqual(b, X.value)

    def test_reshape_negative_one(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test the reshape class with -1 in the shape.\n        '
        expr = cp.Variable((2, 3))
        numpy_expr = np.ones((2, 3))
        shapes = [(-1, 1), (1, -1), (-1, 2), -1, (-1,)]
        expected_shapes = [(6, 1), (1, 6), (3, 2), (6,), (6,)]
        for (shape, expected_shape) in zip(shapes, expected_shapes):
            expr_reshaped = expr.reshape(shape)
            self.assertEqual(expr_reshaped.shape, expected_shape)
            numpy_expr_reshaped = np.reshape(numpy_expr, shape)
            self.assertEqual(numpy_expr_reshaped.shape, expected_shape)
        with pytest.raises(ValueError, match='Cannot reshape expression'):
            expr.reshape((8, -1))
        with pytest.raises(AssertionError, match='Only one'):
            expr.reshape((-1, -1))
        with pytest.raises(ValueError, match='Invalid reshape dimensions'):
            expr.reshape((-1, 0))
        with pytest.raises(AssertionError, match='Specified dimension must be nonnegative'):
            expr.reshape((-1, -2))
        A = np.array([[1, 2, 3], [4, 5, 6]])
        A_reshaped = Constant(A).reshape(-1, order='C')
        assert np.allclose(A_reshaped.value, A.reshape(-1, order='C'))
        A_reshaped = Constant(A).reshape(-1, order='F')
        assert np.allclose(A_reshaped.value, A.reshape(-1, order='F'))

    def test_max(self) -> None:
        if False:
            while True:
                i = 10
        'Test max.\n        '
        self.assertEqual(Variable().max().sign, s.UNKNOWN)
        self.assertEqual(Variable(2).max(axis=0, keepdims=True).shape, (1,))
        self.assertEqual(Variable(2).max(axis=1).shape, (2,))
        self.assertEqual(Variable((2, 3)).max(axis=0, keepdims=True).shape, (1, 3))
        self.assertEqual(Variable((2, 3)).max(axis=1).shape, (2,))
        with self.assertRaises(Exception) as cm:
            self.x.max(axis=4)
        self.assertEqual(str(cm.exception), 'Invalid argument for axis.')

    def test_min(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test min.\n        '
        self.assertEqual(Variable().min().sign, s.UNKNOWN)
        self.assertEqual(Variable(2).min(axis=0).shape, tuple())
        self.assertEqual(Variable(2).min(axis=1).shape, (2,))
        self.assertEqual(Variable((2, 3)).min(axis=0).shape, (3,))
        self.assertEqual(Variable((2, 3)).min(axis=1).shape, (2,))
        with self.assertRaises(Exception) as cm:
            self.x.min(axis=4)
        self.assertEqual(str(cm.exception), 'Invalid argument for axis.')

    def test_sum(self) -> None:
        if False:
            return 10
        'Test the sum atom.\n        '
        self.assertEqual(Constant([1, -1]).sum().sign, s.UNKNOWN)
        self.assertEqual(Constant([1, -1]).sum().curvature, s.CONSTANT)
        self.assertEqual(Variable(2).sum().sign, s.UNKNOWN)
        self.assertEqual(Variable(2).sum().shape, tuple())
        self.assertEqual(Variable(2).sum().curvature, s.AFFINE)
        self.assertEqual(Variable((2, 1)).sum(keepdims=True).shape, (1, 1))
        mat = np.array([[1, -1]])
        self.assertEqual(cp.sum(mat @ cp.square(Variable(2))).curvature, s.UNKNOWN)
        self.assertEqual(Variable(2).sum(axis=0).shape, tuple())
        self.assertEqual(Variable(2).sum(axis=1).shape, (2,))
        self.assertEqual(Variable((2, 3)).sum(axis=0, keepdims=True).shape, (1, 3))
        self.assertEqual(Variable((2, 3)).sum(axis=0, keepdims=False).shape, (3,))
        self.assertEqual(Variable((2, 3)).sum(axis=1).shape, (2,))
        with self.assertRaises(Exception) as cm:
            cp.sum(self.x, axis=4)
        self.assertEqual(str(cm.exception), 'Invalid argument for axis.')
        A = sp.eye(3)
        self.assertEqual(Constant(A).sum().value, 3)
        A = sp.eye(3)
        self.assertItemsAlmostEqual(Constant(A).sum(axis=0).value, [1, 1, 1])

    def test_trace(self) -> None:
        if False:
            return 10
        'Test the trace atom.\n        '
        expr = self.A.trace()
        self.assertEqual(expr.sign, s.UNKNOWN)
        self.assertEqual(expr.curvature, s.AFFINE)
        self.assertEqual(expr.shape, tuple())
        with self.assertRaises(Exception) as cm:
            self.C.trace()
        self.assertEqual(str(cm.exception), 'Argument to trace must be a square matrix.')

    def test_trace_sign_psd(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test sign of trace for psd/nsd inputs.\n        '
        X_psd = cp.Variable((2, 2), PSD=True)
        X_nsd = cp.Variable((2, 2), NSD=True)
        psd_trace = X_psd.trace()
        nsd_trace = X_nsd.trace()
        assert psd_trace.is_nonneg()
        assert nsd_trace.is_nonpos()

    def test_ptp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test the ptp atom.\n        '
        a = Constant(np.array([[10.0, -10.0, 3.0], [6.0, 0.0, -1.5]]))
        expr = a.ptp()
        assert expr.is_nonneg()
        assert expr.shape == ()
        assert np.isclose(expr.value, 20.0)
        expr = a.ptp(axis=0)
        assert expr.is_nonneg()
        assert expr.shape == (3,)
        assert np.allclose(expr.value, np.array([4, 10, 4.5]))
        expr = a.ptp(axis=1)
        assert expr.is_nonneg()
        expr.shape == (2,)
        assert np.allclose(expr.value, np.array([20.0, 7.5]))
        expr = a.ptp(0, keepdims=True)
        assert expr.is_nonneg()
        assert expr.shape == (1, 3)
        assert np.allclose(expr.value, np.array([[4, 10, 4.5]]))
        expr = a.ptp(1, keepdims=True)
        assert expr.is_nonneg()
        assert expr.shape == (2, 1)
        assert np.allclose(expr.value, np.array([[20.0], [7.5]]))

    def test_stats(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the mean, std, var atoms.\n        '
        a_np = np.array([[10.0, 10.0, 3.0], [6.0, 0.0, 1.5]])
        a = Constant(a_np)
        expr_mean = a.mean()
        expr_var = a.var()
        expr_std = a.std()
        assert expr_mean.is_nonneg()
        assert expr_var.is_nonneg()
        assert expr_std.is_nonneg()
        assert np.isclose(a_np.mean(), expr_mean.value)
        assert np.isclose(a_np.var(), expr_var.value)
        assert np.isclose(a_np.std(), expr_std.value)
        for ddof in [0, 1]:
            expr_var = a.var(ddof=ddof)
            expr_std = a.std(ddof=ddof)
            assert np.isclose(a_np.var(ddof=ddof), expr_var.value)
            assert np.isclose(a_np.std(ddof=ddof), expr_std.value)
        for axis in [0, 1]:
            for keepdims in [True, False]:
                expr_mean = a.mean(axis=axis, keepdims=keepdims)
                expr_std = a.std(axis=axis, keepdims=keepdims)
                assert expr_mean.shape == a_np.mean(axis=axis, keepdims=keepdims).shape
                assert expr_std.shape == a_np.std(axis=axis, keepdims=keepdims).shape
                assert np.allclose(a_np.mean(axis=axis, keepdims=keepdims), expr_mean.value)
                assert np.allclose(a_np.std(axis=axis, keepdims=keepdims), expr_std.value)

    def test_conj(self) -> None:
        if False:
            print('Hello World!')
        'Test conj.\n        '
        v = cp.Variable((4,))
        obj = cp.Minimize(cp.sum(v))
        prob = cp.Problem(obj, [v.conj() >= 1])
        prob.solve(solver=cp.SCS)
        assert np.allclose(v.value, np.ones((4,)))

    def test_conjugate(self) -> None:
        if False:
            print('Hello World!')
        'Test conj.\n        '
        v = cp.Variable((4,))
        obj = cp.Minimize(cp.sum(v))
        prob = cp.Problem(obj, [v.conjugate() >= 1])
        prob.solve(solver=cp.SCS)
        assert np.allclose(v.value, np.ones((4,)))