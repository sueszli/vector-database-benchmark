"""Utilities for testing `LinearOperator` and sub-classes."""
import abc
import itertools
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_v2
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as load_model
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import save as save_model
from tensorflow.python.util import nest

class OperatorShapesInfo:
    """Object encoding expected shape for a test.

  Encodes the expected shape of a matrix for a test. Also
  allows additional metadata for the test harness.
  """

    def __init__(self, shape, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.shape = shape
        self.__dict__.update(kwargs)

class CheckTapeSafeSkipOptions:
    DETERMINANT = 'determinant'
    DIAG_PART = 'diag_part'
    LOG_ABS_DETERMINANT = 'log_abs_determinant'
    TRACE = 'trace'

class LinearOperatorDerivedClassTest(test.TestCase, metaclass=abc.ABCMeta):
    """Tests for derived classes.

  Subclasses should implement every abstractmethod, and this will enable all
  test methods to work.
  """
    _atol = {dtypes.float16: 0.001, dtypes.float32: 1e-06, dtypes.float64: 1e-12, dtypes.complex64: 1e-06, dtypes.complex128: 1e-12}
    _rtol = {dtypes.float16: 0.001, dtypes.float32: 1e-06, dtypes.float64: 1e-12, dtypes.complex64: 1e-06, dtypes.complex128: 1e-12}

    def assertAC(self, x, y, check_dtype=False):
        if False:
            for i in range(10):
                print('nop')
        'Derived classes can set _atol, _rtol to get different tolerance.'
        dtype = dtypes.as_dtype(x.dtype)
        atol = self._atol[dtype]
        rtol = self._rtol[dtype]
        self.assertAllClose(x, y, atol=atol, rtol=rtol)
        if check_dtype:
            self.assertDTypeEqual(x, y.dtype)

    @staticmethod
    def adjoint_options():
        if False:
            return 10
        return [False, True]

    @staticmethod
    def adjoint_arg_options():
        if False:
            return 10
        return [False, True]

    @staticmethod
    def dtypes_to_test():
        if False:
            print('Hello World!')
        return [dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]

    @staticmethod
    def use_placeholder_options():
        if False:
            while True:
                i = 10
        return [False, True]

    @staticmethod
    def use_blockwise_arg():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def operator_shapes_infos():
        if False:
            i = 10
            return i + 15
        'Returns list of OperatorShapesInfo, encapsulating the shape to test.'
        raise NotImplementedError('operator_shapes_infos has not been implemented.')

    @abc.abstractmethod
    def operator_and_matrix(self, shapes_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False):
        if False:
            for i in range(10):
                print('nop')
        'Build a batch matrix and an Operator that should have similar behavior.\n\n    Every operator acts like a (batch) matrix.  This method returns both\n    together, and is used by tests.\n\n    Args:\n      shapes_info: `OperatorShapesInfo`, encoding shape information about the\n        operator.\n      dtype:  Numpy dtype.  Data type of returned array/operator.\n      use_placeholder:  Python bool.  If True, initialize the operator with a\n        placeholder of undefined shape and correct dtype.\n      ensure_self_adjoint_and_pd: If `True`,\n        construct this operator to be Hermitian Positive Definite, as well\n        as ensuring the hints `is_positive_definite` and `is_self_adjoint`\n        are set.\n        This is useful for testing methods such as `cholesky`.\n\n    Returns:\n      operator:  `LinearOperator` subclass instance.\n      mat:  `Tensor` representing operator.\n    '
        raise NotImplementedError('Not implemented yet.')

    @abc.abstractmethod
    def make_rhs(self, operator, adjoint, with_batch=True):
        if False:
            for i in range(10):
                print('nop')
        "Make a rhs appropriate for calling operator.solve(rhs).\n\n    Args:\n      operator:  A `LinearOperator`\n      adjoint:  Python `bool`.  If `True`, we are making a 'rhs' value for the\n        adjoint operator.\n      with_batch: Python `bool`. If `True`, create `rhs` with the same batch\n        shape as operator, and otherwise create a matrix without any batch\n        shape.\n\n    Returns:\n      A `Tensor`\n    "
        raise NotImplementedError('make_rhs is not defined.')

    @abc.abstractmethod
    def make_x(self, operator, adjoint, with_batch=True):
        if False:
            for i in range(10):
                print('nop')
        "Make an 'x' appropriate for calling operator.matmul(x).\n\n    Args:\n      operator:  A `LinearOperator`\n      adjoint:  Python `bool`.  If `True`, we are making an 'x' value for the\n        adjoint operator.\n      with_batch: Python `bool`. If `True`, create `x` with the same batch shape\n        as operator, and otherwise create a matrix without any batch shape.\n\n    Returns:\n      A `Tensor`\n    "
        raise NotImplementedError('make_x is not defined.')

    @staticmethod
    def skip_these_tests():
        if False:
            while True:
                i = 10
        'List of test names to skip.'
        return []

    @staticmethod
    def optional_tests():
        if False:
            print('Hello World!')
        'List of optional test names to run.'
        return []

    def assertRaisesError(self, msg):
        if False:
            for i in range(10):
                print('nop')
        'assertRaisesRegexp or OpError, depending on context.executing_eagerly.'
        if context.executing_eagerly():
            return self.assertRaisesRegexp(Exception, msg)
        return self.assertRaisesOpError(msg)

    def check_convert_variables_to_tensors(self, operator):
        if False:
            return 10
        'Checks that internal Variables are correctly converted to Tensors.'
        self.assertIsInstance(operator, composite_tensor.CompositeTensor)
        tensor_operator = composite_tensor.convert_variables_to_tensors(operator)
        self.assertIs(type(operator), type(tensor_operator))
        self.assertEmpty(tensor_operator.variables)
        self._check_tensors_equal_variables(operator, tensor_operator)

    def _check_tensors_equal_variables(self, obj, tensor_obj):
        if False:
            print('Hello World!')
        'Checks that Variables in `obj` have equivalent Tensors in `tensor_obj.'
        if isinstance(obj, variables.Variable):
            self.assertAllClose(ops.convert_to_tensor(obj), ops.convert_to_tensor(tensor_obj))
        elif isinstance(obj, composite_tensor.CompositeTensor):
            params = getattr(obj, 'parameters', {})
            tensor_params = getattr(tensor_obj, 'parameters', {})
            self.assertAllEqual(params.keys(), tensor_params.keys())
            self._check_tensors_equal_variables(params, tensor_params)
        elif nest.is_mapping(obj):
            for (k, v) in obj.items():
                self._check_tensors_equal_variables(v, tensor_obj[k])
        elif nest.is_nested(obj):
            for (x, y) in zip(obj, tensor_obj):
                self._check_tensors_equal_variables(x, y)
        else:
            pass

    def check_tape_safe(self, operator, skip_options=None):
        if False:
            print('Hello World!')
        'Check gradients are not None w.r.t. operator.variables.\n\n    Meant to be called from the derived class.\n\n    This ensures grads are not w.r.t every variable in operator.variables.  If\n    more fine-grained testing is needed, a custom test should be written.\n\n    Args:\n      operator: LinearOperator.  Exact checks done will depend on hints.\n      skip_options: Optional list of CheckTapeSafeSkipOptions.\n        Makes this test skip particular checks.\n    '
        skip_options = skip_options or []
        if not operator.variables:
            raise AssertionError('`operator.variables` was empty')

        def _assert_not_none(iterable):
            if False:
                return 10
            for item in iterable:
                self.assertIsNotNone(item)
        with backprop.GradientTape() as tape:
            grad = tape.gradient(operator.to_dense(), operator.variables)
            _assert_not_none(grad)
        with backprop.GradientTape() as tape:
            var_grad = tape.gradient(operator, operator.variables)
            _assert_not_none(var_grad)
            nest.assert_same_structure(var_grad, grad)
        with backprop.GradientTape() as tape:
            _assert_not_none(tape.gradient(operator.adjoint().to_dense(), operator.variables))
        x = math_ops.cast(array_ops.ones(shape=operator.H.shape_tensor()[:-1]), operator.dtype)
        with backprop.GradientTape() as tape:
            _assert_not_none(tape.gradient(operator.matvec(x), operator.variables))
        if not operator.is_square:
            return
        for option in [CheckTapeSafeSkipOptions.DETERMINANT, CheckTapeSafeSkipOptions.LOG_ABS_DETERMINANT, CheckTapeSafeSkipOptions.DIAG_PART, CheckTapeSafeSkipOptions.TRACE]:
            with backprop.GradientTape() as tape:
                if option not in skip_options:
                    _assert_not_none(tape.gradient(getattr(operator, option)(), operator.variables))
        if operator.is_non_singular is False:
            return
        with backprop.GradientTape() as tape:
            _assert_not_none(tape.gradient(operator.inverse().to_dense(), operator.variables))
        with backprop.GradientTape() as tape:
            _assert_not_none(tape.gradient(operator.solvevec(x), operator.variables))
        if not (operator.is_self_adjoint and operator.is_positive_definite):
            return
        with backprop.GradientTape() as tape:
            _assert_not_none(tape.gradient(operator.cholesky().to_dense(), operator.variables))

def _test_slicing(use_placeholder, shapes_info, dtype):
    if False:
        i = 10
        return i + 15

    def test_slicing(self: 'LinearOperatorDerivedClassTest'):
        if False:
            return 10
        with self.session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            batch_shape = shapes_info.shape[:-2]
            if not batch_shape or batch_shape[0] <= 1:
                return
            slices = [slice(1, -1)]
            if len(batch_shape) > 1:
                slices += [..., slice(0, 1)]
            sliced_operator = operator[slices]
            matrix_slices = slices + [slice(None), slice(None)]
            sliced_matrix = mat[matrix_slices]
            sliced_op_dense = sliced_operator.to_dense()
            (op_dense_v, mat_v) = sess.run([sliced_op_dense, sliced_matrix])
            self.assertAC(op_dense_v, mat_v)
    return test_slicing

def _test_to_dense(use_placeholder, shapes_info, dtype):
    if False:
        i = 10
        return i + 15

    def test_to_dense(self: 'LinearOperatorDerivedClassTest'):
        if False:
            i = 10
            return i + 15
        with self.session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            op_dense = operator.to_dense()
            if not use_placeholder:
                self.assertAllEqual(shapes_info.shape, op_dense.shape)
            (op_dense_v, mat_v) = sess.run([op_dense, mat])
            self.assertAC(op_dense_v, mat_v)
    return test_to_dense

def _test_det(use_placeholder, shapes_info, dtype):
    if False:
        print('Hello World!')

    def test_det(self: 'LinearOperatorDerivedClassTest'):
        if False:
            while True:
                i = 10
        with self.session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            op_det = operator.determinant()
            if not use_placeholder:
                self.assertAllEqual(shapes_info.shape[:-2], op_det.shape)
            (op_det_v, mat_det_v) = sess.run([op_det, linalg_ops.matrix_determinant(mat)])
            self.assertAC(op_det_v, mat_det_v)
    return test_det

def _test_log_abs_det(use_placeholder, shapes_info, dtype):
    if False:
        for i in range(10):
            print('nop')

    def test_log_abs_det(self: 'LinearOperatorDerivedClassTest'):
        if False:
            return 10
        with self.session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            op_log_abs_det = operator.log_abs_determinant()
            (_, mat_log_abs_det) = linalg.slogdet(mat)
            if not use_placeholder:
                self.assertAllEqual(shapes_info.shape[:-2], op_log_abs_det.shape)
            (op_log_abs_det_v, mat_log_abs_det_v) = sess.run([op_log_abs_det, mat_log_abs_det])
            self.assertAC(op_log_abs_det_v, mat_log_abs_det_v)
    return test_log_abs_det

def _test_operator_matmul_with_same_type(use_placeholder, shapes_info, dtype):
    if False:
        while True:
            i = 10
    'op_a.matmul(op_b), in the case where the same type is returned.'

    @test_util.run_without_tensor_float_32('Use FP32 in matmul')
    def test_operator_matmul_with_same_type(self: 'LinearOperatorDerivedClassTest'):
        if False:
            print('Hello World!')
        with self.session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator_a, mat_a) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            (operator_b, mat_b) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            mat_matmul = math_ops.matmul(mat_a, mat_b)
            op_matmul = operator_a.matmul(operator_b)
            (mat_matmul_v, op_matmul_v) = sess.run([mat_matmul, op_matmul.to_dense()])
            self.assertIsInstance(op_matmul, operator_a.__class__)
            self.assertAC(mat_matmul_v, op_matmul_v)
    return test_operator_matmul_with_same_type

def _test_operator_solve_with_same_type(use_placeholder, shapes_info, dtype):
    if False:
        i = 10
        return i + 15
    'op_a.solve(op_b), in the case where the same type is returned.'

    def test_operator_solve_with_same_type(self: 'LinearOperatorDerivedClassTest'):
        if False:
            i = 10
            return i + 15
        with self.session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator_a, mat_a) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            (operator_b, mat_b) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            mat_solve = linear_operator_util.matrix_solve_with_broadcast(mat_a, mat_b)
            op_solve = operator_a.solve(operator_b)
            (mat_solve_v, op_solve_v) = sess.run([mat_solve, op_solve.to_dense()])
            self.assertIsInstance(op_solve, operator_a.__class__)
            self.assertAC(mat_solve_v, op_solve_v)
    return test_operator_solve_with_same_type

def _test_matmul_base(self: 'LinearOperatorDerivedClassTest', use_placeholder, shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg, with_batch):
    if False:
        i = 10
        return i + 15
    if not with_batch and len(shapes_info.shape) <= 2:
        return
    with self.session(graph=ops.Graph()) as sess:
        sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
        (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
        x = self.make_x(operator, adjoint=adjoint, with_batch=with_batch)
        if adjoint_arg:
            op_matmul = operator.matmul(linalg.adjoint(x), adjoint=adjoint, adjoint_arg=adjoint_arg)
        else:
            op_matmul = operator.matmul(x, adjoint=adjoint)
        mat_matmul = math_ops.matmul(mat, x, adjoint_a=adjoint)
        if not use_placeholder:
            self.assertAllEqual(op_matmul.shape, mat_matmul.shape)
        if blockwise_arg and len(operator.operators) > 1:
            block_dimensions = operator._block_range_dimensions() if adjoint else operator._block_domain_dimensions()
            block_dimensions_fn = operator._block_range_dimension_tensors if adjoint else operator._block_domain_dimension_tensors
            split_x = linear_operator_util.split_arg_into_blocks(block_dimensions, block_dimensions_fn, x, axis=-2)
            if adjoint_arg:
                split_x = [linalg.adjoint(y) for y in split_x]
            split_matmul = operator.matmul(split_x, adjoint=adjoint, adjoint_arg=adjoint_arg)
            self.assertEqual(len(split_matmul), len(operator.operators))
            split_matmul = linear_operator_util.broadcast_matrix_batch_dims(split_matmul)
            fused_block_matmul = array_ops.concat(split_matmul, axis=-2)
            (op_matmul_v, mat_matmul_v, fused_block_matmul_v) = sess.run([op_matmul, mat_matmul, fused_block_matmul])
            self.assertAC(fused_block_matmul_v, mat_matmul_v)
        else:
            (op_matmul_v, mat_matmul_v) = sess.run([op_matmul, mat_matmul])
        self.assertAC(op_matmul_v, mat_matmul_v)

def _test_matmul(use_placeholder, shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg):
    if False:
        for i in range(10):
            print('nop')

    @test_util.run_without_tensor_float_32('Use FP32 in matmul')
    def test_matmul(self: 'LinearOperatorDerivedClassTest'):
        if False:
            i = 10
            return i + 15
        _test_matmul_base(self, use_placeholder, shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg, with_batch=True)
    return test_matmul

def _test_matmul_with_broadcast(use_placeholder, shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg):
    if False:
        i = 10
        return i + 15

    @test_util.run_without_tensor_float_32('Use FP32 in matmul')
    def test_matmul_with_broadcast(self: 'LinearOperatorDerivedClassTest'):
        if False:
            i = 10
            return i + 15
        _test_matmul_base(self, use_placeholder, shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg, with_batch=True)
    return test_matmul_with_broadcast

def _test_adjoint(use_placeholder, shapes_info, dtype):
    if False:
        for i in range(10):
            print('nop')

    def test_adjoint(self: 'LinearOperatorDerivedClassTest'):
        if False:
            return 10
        with self.test_session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            op_adjoint = operator.adjoint().to_dense()
            op_adjoint_h = operator.H.to_dense()
            mat_adjoint = linalg.adjoint(mat)
            (op_adjoint_v, op_adjoint_h_v, mat_adjoint_v) = sess.run([op_adjoint, op_adjoint_h, mat_adjoint])
            self.assertAC(mat_adjoint_v, op_adjoint_v)
            self.assertAC(mat_adjoint_v, op_adjoint_h_v)
    return test_adjoint

def _test_cholesky(use_placeholder, shapes_info, dtype):
    if False:
        print('Hello World!')

    def test_cholesky(self: 'LinearOperatorDerivedClassTest'):
        if False:
            for i in range(10):
                print('nop')
        with self.test_session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED + 2
            (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder, ensure_self_adjoint_and_pd=True)
            op_chol = operator.cholesky().to_dense()
            mat_chol = linalg_ops.cholesky(mat)
            (op_chol_v, mat_chol_v) = sess.run([op_chol, mat_chol])
            self.assertAC(mat_chol_v, op_chol_v)
    return test_cholesky

def _test_eigvalsh(use_placeholder, shapes_info, dtype):
    if False:
        return 10

    def test_eigvalsh(self: 'LinearOperatorDerivedClassTest'):
        if False:
            i = 10
            return i + 15
        with self.test_session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder, ensure_self_adjoint_and_pd=True)
            op_eigvals = sort_ops.sort(math_ops.cast(operator.eigvals(), dtype=dtypes.float64), axis=-1)
            if dtype.is_complex:
                mat = math_ops.cast(mat, dtype=dtypes.complex128)
            else:
                mat = math_ops.cast(mat, dtype=dtypes.float64)
            mat_eigvals = sort_ops.sort(math_ops.cast(linalg_ops.self_adjoint_eigvals(mat), dtype=dtypes.float64), axis=-1)
            (op_eigvals_v, mat_eigvals_v) = sess.run([op_eigvals, mat_eigvals])
            atol = self._atol[dtype]
            rtol = self._rtol[dtype]
            if dtype == dtypes.float32 or dtype == dtypes.complex64:
                atol = 0.0002
                rtol = 0.0002
            self.assertAllClose(op_eigvals_v, mat_eigvals_v, atol=atol, rtol=rtol)
    return test_eigvalsh

def _test_cond(use_placeholder, shapes_info, dtype):
    if False:
        return 10

    def test_cond(self: 'LinearOperatorDerivedClassTest'):
        if False:
            while True:
                i = 10
        with self.test_session(graph=ops.Graph()) as sess:
            if 0 in shapes_info.shape[-2:]:
                return
            if test.is_built_with_rocm() and (dtype == dtypes.complex64 or dtype == dtypes.complex128):
                return
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder, ensure_self_adjoint_and_pd=True)
            op_cond = operator.cond()
            s = math_ops.abs(linalg_ops.svd(mat, compute_uv=False))
            mat_cond = math_ops.reduce_max(s, axis=-1) / math_ops.reduce_min(s, axis=-1)
            (op_cond_v, mat_cond_v) = sess.run([op_cond, mat_cond])
            atol_override = {dtypes.float16: 0.01, dtypes.float32: 0.001, dtypes.float64: 1e-06, dtypes.complex64: 0.001, dtypes.complex128: 1e-06}
            rtol_override = {dtypes.float16: 0.01, dtypes.float32: 0.001, dtypes.float64: 0.0001, dtypes.complex64: 0.001, dtypes.complex128: 1e-06}
            atol = atol_override[dtype]
            rtol = rtol_override[dtype]
            self.assertAllClose(op_cond_v, mat_cond_v, atol=atol, rtol=rtol)
    return test_cond

def _test_solve_base(self: 'LinearOperatorDerivedClassTest', use_placeholder, shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg, with_batch):
    if False:
        while True:
            i = 10
    if not with_batch and len(shapes_info.shape) <= 2:
        return
    with self.session(graph=ops.Graph()) as sess:
        sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
        (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
        rhs = self.make_rhs(operator, adjoint=adjoint, with_batch=with_batch)
        if adjoint_arg:
            op_solve = operator.solve(linalg.adjoint(rhs), adjoint=adjoint, adjoint_arg=adjoint_arg)
        else:
            op_solve = operator.solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
        mat_solve = linear_operator_util.matrix_solve_with_broadcast(mat, rhs, adjoint=adjoint)
        if not use_placeholder:
            self.assertAllEqual(op_solve.shape, mat_solve.shape)
        if blockwise_arg and len(operator.operators) > 1:
            block_dimensions = operator._block_range_dimensions() if adjoint else operator._block_domain_dimensions()
            block_dimensions_fn = operator._block_range_dimension_tensors if adjoint else operator._block_domain_dimension_tensors
            split_rhs = linear_operator_util.split_arg_into_blocks(block_dimensions, block_dimensions_fn, rhs, axis=-2)
            if adjoint_arg:
                split_rhs = [linalg.adjoint(y) for y in split_rhs]
            split_solve = operator.solve(split_rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
            self.assertEqual(len(split_solve), len(operator.operators))
            split_solve = linear_operator_util.broadcast_matrix_batch_dims(split_solve)
            fused_block_solve = array_ops.concat(split_solve, axis=-2)
            (op_solve_v, mat_solve_v, fused_block_solve_v) = sess.run([op_solve, mat_solve, fused_block_solve])
            self.assertAC(mat_solve_v, fused_block_solve_v)
        else:
            (op_solve_v, mat_solve_v) = sess.run([op_solve, mat_solve])
        self.assertAC(op_solve_v, mat_solve_v)

def _test_solve(use_placeholder, shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg):
    if False:
        return 10

    def test_solve(self: 'LinearOperatorDerivedClassTest'):
        if False:
            print('Hello World!')
        _test_solve_base(self, use_placeholder, shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg, with_batch=True)
    return test_solve

def _test_solve_with_broadcast(use_placeholder, shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg):
    if False:
        for i in range(10):
            print('nop')

    def test_solve_with_broadcast(self: 'LinearOperatorDerivedClassTest'):
        if False:
            i = 10
            return i + 15
        _test_solve_base(self, use_placeholder, shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg, with_batch=False)
    return test_solve_with_broadcast

def _test_inverse(use_placeholder, shapes_info, dtype):
    if False:
        for i in range(10):
            print('nop')

    def test_inverse(self: 'LinearOperatorDerivedClassTest'):
        if False:
            i = 10
            return i + 15
        with self.session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            (op_inverse_v, mat_inverse_v) = sess.run([operator.inverse().to_dense(), linalg.inv(mat)])
            self.assertAC(op_inverse_v, mat_inverse_v, check_dtype=True)
    return test_inverse

def _test_trace(use_placeholder, shapes_info, dtype):
    if False:
        i = 10
        return i + 15

    def test_trace(self: 'LinearOperatorDerivedClassTest'):
        if False:
            i = 10
            return i + 15
        with self.session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            op_trace = operator.trace()
            mat_trace = math_ops.trace(mat)
            if not use_placeholder:
                self.assertAllEqual(op_trace.shape, mat_trace.shape)
            (op_trace_v, mat_trace_v) = sess.run([op_trace, mat_trace])
            self.assertAC(op_trace_v, mat_trace_v)
    return test_trace

def _test_add_to_tensor(use_placeholder, shapes_info, dtype):
    if False:
        while True:
            i = 10

    def test_add_to_tensor(self: 'LinearOperatorDerivedClassTest'):
        if False:
            while True:
                i = 10
        with self.session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            op_plus_2mat = operator.add_to_tensor(2 * mat)
            if not use_placeholder:
                self.assertAllEqual(shapes_info.shape, op_plus_2mat.shape)
            (op_plus_2mat_v, mat_v) = sess.run([op_plus_2mat, mat])
            self.assertAC(op_plus_2mat_v, 3 * mat_v)
    return test_add_to_tensor

def _test_diag_part(use_placeholder, shapes_info, dtype):
    if False:
        return 10

    def test_diag_part(self: 'LinearOperatorDerivedClassTest'):
        if False:
            return 10
        with self.session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            op_diag_part = operator.diag_part()
            mat_diag_part = array_ops.matrix_diag_part(mat)
            if not use_placeholder:
                self.assertAllEqual(mat_diag_part.shape, op_diag_part.shape)
            (op_diag_part_, mat_diag_part_) = sess.run([op_diag_part, mat_diag_part])
            self.assertAC(op_diag_part_, mat_diag_part_)
    return test_diag_part

def _test_composite_tensor(use_placeholder, shapes_info, dtype):
    if False:
        return 10

    @test_util.run_without_tensor_float_32('Use FP32 in matmul')
    def test_composite_tensor(self: 'LinearOperatorDerivedClassTest'):
        if False:
            for i in range(10):
                print('nop')
        with self.session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            self.assertIsInstance(operator, composite_tensor.CompositeTensor)
            flat = nest.flatten(operator, expand_composites=True)
            unflat = nest.pack_sequence_as(operator, flat, expand_composites=True)
            self.assertIsInstance(unflat, type(operator))
            x = self.make_x(operator, adjoint=False)
            op_y = def_function.function(lambda op: op.matmul(x))(unflat)
            mat_y = math_ops.matmul(mat, x)
            if not use_placeholder:
                self.assertAllEqual(mat_y.shape, op_y.shape)

            def body(op):
                if False:
                    for i in range(10):
                        print('nop')
                return (type(op)(**op.parameters),)
            (op_out,) = while_v2.while_loop(cond=lambda _: True, body=body, loop_vars=(operator,), maximum_iterations=3)
            loop_y = op_out.matmul(x)
            (op_y_, loop_y_, mat_y_) = sess.run([op_y, loop_y, mat_y])
            self.assertAC(op_y_, mat_y_)
            self.assertAC(loop_y_, mat_y_)
            nested_structure_coder.encode_structure(operator._type_spec)
    return test_composite_tensor

def _test_saved_model(use_placeholder, shapes_info, dtype):
    if False:
        return 10

    @test_util.run_without_tensor_float_32('Use FP32 in matmul')
    def test_saved_model(self: 'LinearOperatorDerivedClassTest'):
        if False:
            for i in range(10):
                print('nop')
        with self.session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator, mat) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            x = self.make_x(operator, adjoint=False)

            class Model(module.Module):

                def __init__(self, init_x):
                    if False:
                        i = 10
                        return i + 15
                    self.x = nest.map_structure(lambda x_: variables.Variable(x_, shape=None), init_x)

                @def_function.function(input_signature=(operator._type_spec,))
                def do_matmul(self, op):
                    if False:
                        return 10
                    return op.matmul(self.x)
            saved_model_dir = self.get_temp_dir()
            m1 = Model(x)
            sess.run([v.initializer for v in m1.variables])
            sess.run(m1.x.assign(m1.x + 1.0))
            save_model.save(m1, saved_model_dir)
            m2 = load_model.load(saved_model_dir)
            sess.run(m2.x.initializer)
            sess.run(m2.x.assign(m2.x + 1.0))
            y_op = m2.do_matmul(operator)
            y_mat = math_ops.matmul(mat, m2.x)
            (y_op_, y_mat_) = sess.run([y_op, y_mat])
            self.assertAC(y_op_, y_mat_)
    return test_saved_model

def _test_composite_tensor_gradient(use_placeholder, shapes_info, dtype):
    if False:
        i = 10
        return i + 15

    def test_composite_tensor_gradient(self: 'LinearOperatorDerivedClassTest'):
        if False:
            i = 10
            return i + 15
        with self.session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            (operator, _) = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            x = self.make_x(operator, adjoint=False)
            y = operator.matmul(x)
            (op_g,) = gradients_impl.gradients(y, operator, grad_ys=array_ops.ones_like(y))

            def _unflatten_and_matmul(components):
                if False:
                    return 10
                unflat_op = nest.pack_sequence_as(operator, components, expand_composites=True)
                return unflat_op.matmul(x)
            flat_op = nest.flatten(operator, expand_composites=True)
            y_ = _unflatten_and_matmul(flat_op)
            flat_g = gradients_impl.gradients(y_, flat_op, grad_ys=array_ops.ones_like(y_))
            if all((g is None for g in flat_g)):
                self.assertIsNone(op_g)
            else:
                self.assertIsInstance(op_g, operator.__class__)
                for (g, ug) in zip(nest.flatten(op_g, expand_composites=True), nest.flatten(flat_g, expand_composites=True)):
                    self.assertAllClose(g, ug)
    return test_composite_tensor_gradient

def add_tests(test_cls):
    if False:
        i = 10
        return i + 15
    'Add tests for LinearOperator methods.'
    test_name_dict = {'add_to_tensor': _test_add_to_tensor, 'adjoint': _test_adjoint, 'cholesky': _test_cholesky, 'cond': _test_cond, 'composite_tensor': _test_composite_tensor, 'composite_tensor_gradient': _test_composite_tensor_gradient, 'det': _test_det, 'diag_part': _test_diag_part, 'eigvalsh': _test_eigvalsh, 'inverse': _test_inverse, 'log_abs_det': _test_log_abs_det, 'operator_matmul_with_same_type': _test_operator_matmul_with_same_type, 'operator_solve_with_same_type': _test_operator_solve_with_same_type, 'matmul': _test_matmul, 'matmul_with_broadcast': _test_matmul_with_broadcast, 'saved_model': _test_saved_model, 'slicing': _test_slicing, 'solve': _test_solve, 'solve_with_broadcast': _test_solve_with_broadcast, 'to_dense': _test_to_dense, 'trace': _test_trace}
    optional_tests = ['operator_matmul_with_same_type', 'operator_solve_with_same_type']
    tests_with_adjoint_args = ['matmul', 'matmul_with_broadcast', 'solve', 'solve_with_broadcast']
    if set(test_cls.skip_these_tests()).intersection(test_cls.optional_tests()):
        raise ValueError(f"Test class {{test_cls}} had intersecting 'skip_these_tests' {test_cls.skip_these_tests()} and 'optional_tests' {test_cls.optional_tests()}.")
    for (name, test_template_fn) in test_name_dict.items():
        if name in test_cls.skip_these_tests():
            continue
        if name in optional_tests and name not in test_cls.optional_tests():
            continue
        for (dtype, use_placeholder, shape_info) in itertools.product(test_cls.dtypes_to_test(), test_cls.use_placeholder_options(), test_cls.operator_shapes_infos()):
            base_test_name = '_'.join(['test', name, '_shape={},dtype={},use_placeholder={}'.format(shape_info.shape, dtype, use_placeholder)])
            if name in tests_with_adjoint_args:
                for adjoint in test_cls.adjoint_options():
                    for adjoint_arg in test_cls.adjoint_arg_options():
                        test_name = base_test_name + ',adjoint={},adjoint_arg={}'.format(adjoint, adjoint_arg)
                        if hasattr(test_cls, test_name):
                            raise RuntimeError('Test %s defined more than once' % test_name)
                        setattr(test_cls, test_name, test_util.run_deprecated_v1(test_template_fn(use_placeholder, shape_info, dtype, adjoint, adjoint_arg, test_cls.use_blockwise_arg())))
            else:
                if hasattr(test_cls, base_test_name):
                    raise RuntimeError('Test %s defined more than once' % base_test_name)
                setattr(test_cls, base_test_name, test_util.run_deprecated_v1(test_template_fn(use_placeholder, shape_info, dtype)))

class SquareLinearOperatorDerivedClassTest(LinearOperatorDerivedClassTest, metaclass=abc.ABCMeta):
    """Base test class appropriate for square operators.

  Sub-classes must still define all abstractmethods from
  LinearOperatorDerivedClassTest that are not defined here.
  """

    @staticmethod
    def operator_shapes_infos():
        if False:
            return 10
        shapes_info = OperatorShapesInfo
        return [shapes_info((0, 0)), shapes_info((1, 1)), shapes_info((1, 3, 3)), shapes_info((3, 4, 4)), shapes_info((2, 1, 4, 4))]

    def make_rhs(self, operator, adjoint, with_batch=True):
        if False:
            print('Hello World!')
        return self.make_x(operator, adjoint=not adjoint, with_batch=with_batch)

    def make_x(self, operator, adjoint, with_batch=True):
        if False:
            for i in range(10):
                print('nop')
        r = self._get_num_systems(operator)
        if operator.shape.is_fully_defined():
            batch_shape = operator.batch_shape.as_list()
            n = operator.domain_dimension.value
            if with_batch:
                x_shape = batch_shape + [n, r]
            else:
                x_shape = [n, r]
        else:
            batch_shape = operator.batch_shape_tensor()
            n = operator.domain_dimension_tensor()
            if with_batch:
                x_shape = array_ops.concat((batch_shape, [n, r]), 0)
            else:
                x_shape = [n, r]
        return random_normal(x_shape, dtype=operator.dtype)

    def _get_num_systems(self, operator):
        if False:
            print('Hello World!')
        'Get some number, either 1 or 2, depending on operator.'
        if operator.tensor_rank is None or operator.tensor_rank % 2:
            return 1
        else:
            return 2

class NonSquareLinearOperatorDerivedClassTest(LinearOperatorDerivedClassTest, metaclass=abc.ABCMeta):
    """Base test class appropriate for generic rectangular operators.

  Square shapes are never tested by this class, so if you want to test your
  operator with a square shape, create two test classes, the other subclassing
  SquareLinearOperatorFullMatrixTest.

  Sub-classes must still define all abstractmethods from
  LinearOperatorDerivedClassTest that are not defined here.
  """

    @staticmethod
    def skip_these_tests():
        if False:
            i = 10
            return i + 15
        'List of test names to skip.'
        return ['cholesky', 'eigvalsh', 'inverse', 'solve', 'solve_with_broadcast', 'det', 'log_abs_det']

    @staticmethod
    def operator_shapes_infos():
        if False:
            return 10
        shapes_info = OperatorShapesInfo
        return [shapes_info((2, 1)), shapes_info((1, 2)), shapes_info((1, 3, 2)), shapes_info((3, 3, 4)), shapes_info((2, 1, 2, 4))]

    def make_rhs(self, operator, adjoint, with_batch=True):
        if False:
            while True:
                i = 10
        raise NotImplementedError("make_rhs not implemented because we don't test solve")

    def make_x(self, operator, adjoint, with_batch=True):
        if False:
            while True:
                i = 10
        r = self._get_num_systems(operator)
        if operator.shape.is_fully_defined():
            batch_shape = operator.batch_shape.as_list()
            if adjoint:
                n = operator.range_dimension.value
            else:
                n = operator.domain_dimension.value
            if with_batch:
                x_shape = batch_shape + [n, r]
            else:
                x_shape = [n, r]
        else:
            batch_shape = operator.batch_shape_tensor()
            if adjoint:
                n = operator.range_dimension_tensor()
            else:
                n = operator.domain_dimension_tensor()
            if with_batch:
                x_shape = array_ops.concat((batch_shape, [n, r]), 0)
            else:
                x_shape = [n, r]
        return random_normal(x_shape, dtype=operator.dtype)

    def _get_num_systems(self, operator):
        if False:
            return 10
        'Get some number, either 1 or 2, depending on operator.'
        if operator.tensor_rank is None or operator.tensor_rank % 2:
            return 1
        else:
            return 2

def random_positive_definite_matrix(shape, dtype, oversampling_ratio=4, force_well_conditioned=False):
    if False:
        while True:
            i = 10
    '[batch] positive definite Wisart matrix.\n\n  A Wishart(N, S) matrix is the S sample covariance matrix of an N-variate\n  (standard) Normal random variable.\n\n  Args:\n    shape:  `TensorShape` or Python list.  Shape of the returned matrix.\n    dtype:  `TensorFlow` `dtype` or Python dtype.\n    oversampling_ratio: S / N in the above.  If S < N, the matrix will be\n      singular (unless `force_well_conditioned is True`).\n    force_well_conditioned:  Python bool.  If `True`, add `1` to the diagonal\n      of the Wishart matrix, then divide by 2, ensuring most eigenvalues are\n      close to 1.\n\n  Returns:\n    `Tensor` with desired shape and dtype.\n  '
    dtype = dtypes.as_dtype(dtype)
    if not tensor_util.is_tf_type(shape):
        shape = tensor_shape.TensorShape(shape)
        shape.dims[-1].assert_is_compatible_with(shape.dims[-2])
    shape = shape.as_list()
    n = shape[-2]
    s = oversampling_ratio * shape[-1]
    wigner_shape = shape[:-2] + [n, s]
    with ops.name_scope('random_positive_definite_matrix'):
        wigner = random_normal(wigner_shape, dtype=dtype, stddev=math_ops.cast(1 / np.sqrt(s), dtype.real_dtype))
        wishart = math_ops.matmul(wigner, wigner, adjoint_b=True)
        if force_well_conditioned:
            wishart += linalg_ops.eye(n, dtype=dtype)
            wishart /= math_ops.cast(2, dtype)
        return wishart

def random_tril_matrix(shape, dtype, force_well_conditioned=False, remove_upper=True):
    if False:
        while True:
            i = 10
    "[batch] lower triangular matrix.\n\n  Args:\n    shape:  `TensorShape` or Python `list`.  Shape of the returned matrix.\n    dtype:  `TensorFlow` `dtype` or Python dtype\n    force_well_conditioned:  Python `bool`. If `True`, returned matrix will have\n      eigenvalues with modulus in `(1, 2)`.  Otherwise, eigenvalues are unit\n      normal random variables.\n    remove_upper:  Python `bool`.\n      If `True`, zero out the strictly upper triangle.\n      If `False`, the lower triangle of returned matrix will have desired\n      properties, but will not have the strictly upper triangle zero'd out.\n\n  Returns:\n    `Tensor` with desired shape and dtype.\n  "
    with ops.name_scope('random_tril_matrix'):
        tril = random_normal(shape, dtype=dtype)
        if remove_upper:
            tril = array_ops.matrix_band_part(tril, -1, 0)
        if force_well_conditioned:
            maxval = ops.convert_to_tensor(np.sqrt(2.0), dtype=dtype.real_dtype)
            diag = random_sign_uniform(shape[:-1], dtype=dtype, minval=1.0, maxval=maxval)
            tril = array_ops.matrix_set_diag(tril, diag)
        return tril

def random_normal(shape, mean=0.0, stddev=1.0, dtype=dtypes.float32, seed=None):
    if False:
        return 10
    'Tensor with (possibly complex) Gaussian entries.\n\n  Samples are distributed like\n\n  ```\n  N(mean, stddev^2), if dtype is real,\n  X + iY,  where X, Y ~ N(mean, stddev^2) if dtype is complex.\n  ```\n\n  Args:\n    shape:  `TensorShape` or Python list.  Shape of the returned tensor.\n    mean:  `Tensor` giving mean of normal to sample from.\n    stddev:  `Tensor` giving stdev of normal to sample from.\n    dtype:  `TensorFlow` `dtype` or numpy dtype\n    seed:  Python integer seed for the RNG.\n\n  Returns:\n    `Tensor` with desired shape and dtype.\n  '
    dtype = dtypes.as_dtype(dtype)
    with ops.name_scope('random_normal'):
        samples = random_ops.random_normal(shape, mean=mean, stddev=stddev, dtype=dtype.real_dtype, seed=seed)
        if dtype.is_complex:
            if seed is not None:
                seed += 1234
            more_samples = random_ops.random_normal(shape, mean=mean, stddev=stddev, dtype=dtype.real_dtype, seed=seed)
            samples = math_ops.complex(samples, more_samples)
        return samples

def random_uniform(shape, minval=None, maxval=None, dtype=dtypes.float32, seed=None):
    if False:
        print('Hello World!')
    'Tensor with (possibly complex) Uniform entries.\n\n  Samples are distributed like\n\n  ```\n  Uniform[minval, maxval], if dtype is real,\n  X + iY,  where X, Y ~ Uniform[minval, maxval], if dtype is complex.\n  ```\n\n  Args:\n    shape:  `TensorShape` or Python list.  Shape of the returned tensor.\n    minval:  `0-D` `Tensor` giving the minimum values.\n    maxval:  `0-D` `Tensor` giving the maximum values.\n    dtype:  `TensorFlow` `dtype` or Python dtype\n    seed:  Python integer seed for the RNG.\n\n  Returns:\n    `Tensor` with desired shape and dtype.\n  '
    dtype = dtypes.as_dtype(dtype)
    with ops.name_scope('random_uniform'):
        samples = random_ops.random_uniform(shape, dtype=dtype.real_dtype, minval=minval, maxval=maxval, seed=seed)
        if dtype.is_complex:
            if seed is not None:
                seed += 12345
            more_samples = random_ops.random_uniform(shape, dtype=dtype.real_dtype, minval=minval, maxval=maxval, seed=seed)
            samples = math_ops.complex(samples, more_samples)
        return samples

def random_sign_uniform(shape, minval=None, maxval=None, dtype=dtypes.float32, seed=None):
    if False:
        i = 10
        return i + 15
    'Tensor with (possibly complex) random entries from a "sign Uniform".\n\n  Letting `Z` be a random variable equal to `-1` and `1` with equal probability,\n  Samples from this `Op` are distributed like\n\n  ```\n  Z * X, where X ~ Uniform[minval, maxval], if dtype is real,\n  Z * (X + iY),  where X, Y ~ Uniform[minval, maxval], if dtype is complex.\n  ```\n\n  Args:\n    shape:  `TensorShape` or Python list.  Shape of the returned tensor.\n    minval:  `0-D` `Tensor` giving the minimum values.\n    maxval:  `0-D` `Tensor` giving the maximum values.\n    dtype:  `TensorFlow` `dtype` or Python dtype\n    seed:  Python integer seed for the RNG.\n\n  Returns:\n    `Tensor` with desired shape and dtype.\n  '
    dtype = dtypes.as_dtype(dtype)
    with ops.name_scope('random_sign_uniform'):
        unsigned_samples = random_uniform(shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)
        if seed is not None:
            seed += 12
        signs = math_ops.sign(random_ops.random_uniform(shape, minval=-1.0, maxval=1.0, seed=seed))
        return unsigned_samples * math_ops.cast(signs, unsigned_samples.dtype)

def random_normal_correlated_columns(shape, mean=0.0, stddev=1.0, dtype=dtypes.float32, eps=0.0001, seed=None):
    if False:
        i = 10
        return i + 15
    'Batch matrix with (possibly complex) Gaussian entries and correlated cols.\n\n  Returns random batch matrix `A` with specified element-wise `mean`, `stddev`,\n  living close to an embedded hyperplane.\n\n  Suppose `shape[-2:] = (M, N)`.\n\n  If `M < N`, `A` is a random `M x N` [batch] matrix with iid Gaussian entries.\n\n  If `M >= N`, then the columns of `A` will be made almost dependent as follows:\n\n  ```\n  L = random normal N x N-1 matrix, mean = 0, stddev = 1 / sqrt(N - 1)\n  B = random normal M x N-1 matrix, mean = 0, stddev = stddev.\n\n  G = (L B^H)^H, a random normal M x N matrix, living on N-1 dim hyperplane\n  E = a random normal M x N matrix, mean = 0, stddev = eps\n  mu = a constant M x N matrix, equal to the argument "mean"\n\n  A = G + E + mu\n  ```\n\n  Args:\n    shape:  Python list of integers.\n      Shape of the returned tensor.  Must be at least length two.\n    mean:  `Tensor` giving mean of normal to sample from.\n    stddev:  `Tensor` giving stdev of normal to sample from.\n    dtype:  `TensorFlow` `dtype` or numpy dtype\n    eps:  Distance each column is perturbed from the low-dimensional subspace.\n    seed:  Python integer seed for the RNG.\n\n  Returns:\n    `Tensor` with desired shape and dtype.\n\n  Raises:\n    ValueError:  If `shape` is not at least length 2.\n  '
    dtype = dtypes.as_dtype(dtype)
    if len(shape) < 2:
        raise ValueError('Argument shape must be at least length 2.  Found: %s' % shape)
    shape = list(shape)
    batch_shape = shape[:-2]
    (m, n) = shape[-2:]
    if n < 2 or n < m:
        return random_normal(shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)
    smaller_shape = batch_shape + [m, n - 1]
    embedding_mat_shape = batch_shape + [n, n - 1]
    stddev_mat = 1 / np.sqrt(n - 1)
    with ops.name_scope('random_normal_correlated_columns'):
        smaller_mat = random_normal(smaller_shape, mean=0.0, stddev=stddev_mat, dtype=dtype, seed=seed)
        if seed is not None:
            seed += 1287
        embedding_mat = random_normal(embedding_mat_shape, dtype=dtype, seed=seed)
        embedded_t = math_ops.matmul(embedding_mat, smaller_mat, transpose_b=True)
        embedded = array_ops.matrix_transpose(embedded_t)
        mean_mat = array_ops.ones_like(embedded) * mean
        return embedded + random_normal(shape, stddev=eps, dtype=dtype) + mean_mat