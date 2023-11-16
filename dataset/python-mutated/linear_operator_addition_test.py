import numpy as np
from tensorflow.python.framework import test_util
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_addition
from tensorflow.python.platform import test
linalg = linalg_lib
rng = np.random.RandomState(0)
add_operators = linear_operator_addition.add_operators

class _BadAdder(linear_operator_addition._Adder):
    """Adder that will fail if used."""

    def can_add(self, op1, op2):
        if False:
            while True:
                i = 10
        raise AssertionError('BadAdder.can_add called!')

    def _add(self, op1, op2, operator_name, hints):
        if False:
            return 10
        raise AssertionError('This line should not be reached')

class LinearOperatorAdditionCorrectnessTest(test.TestCase):
    """Tests correctness of addition with combinations of a few Adders.

  Tests here are done with the _DEFAULT_ADDITION_TIERS, which means
  add_operators should reduce all operators resulting in one single operator.

  This shows that we are able to correctly combine adders using the tiered
  system.  All Adders should be tested separately, and there is no need to test
  every Adder within this class.
  """

    def test_one_operator_is_returned_unchanged(self):
        if False:
            while True:
                i = 10
        op_a = linalg.LinearOperatorDiag([1.0, 1.0])
        op_sum = add_operators([op_a])
        self.assertEqual(1, len(op_sum))
        self.assertIs(op_sum[0], op_a)

    def test_at_least_one_operators_required(self):
        if False:
            return 10
        with self.assertRaisesRegex(ValueError, 'must contain at least one'):
            add_operators([])

    def test_attempting_to_add_numbers_raises(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(TypeError, 'contain only LinearOperator'):
            add_operators([1, 2])

    @test_util.run_deprecated_v1
    def test_two_diag_operators(self):
        if False:
            print('Hello World!')
        op_a = linalg.LinearOperatorDiag([1.0, 1.0], is_positive_definite=True, name='A')
        op_b = linalg.LinearOperatorDiag([2.0, 2.0], is_positive_definite=True, name='B')
        with self.cached_session():
            op_sum = add_operators([op_a, op_b])
            self.assertEqual(1, len(op_sum))
            op = op_sum[0]
            self.assertIsInstance(op, linalg_lib.LinearOperatorDiag)
            self.assertAllClose([[3.0, 0.0], [0.0, 3.0]], op.to_dense())
            self.assertTrue(op.is_positive_definite)
            self.assertTrue(op.is_self_adjoint)
            self.assertTrue(op.is_non_singular)
            self.assertEqual('Add/B__A/', op.name)

    @test_util.run_deprecated_v1
    def test_three_diag_operators(self):
        if False:
            while True:
                i = 10
        op1 = linalg.LinearOperatorDiag([1.0, 1.0], is_positive_definite=True, name='op1')
        op2 = linalg.LinearOperatorDiag([2.0, 2.0], is_positive_definite=True, name='op2')
        op3 = linalg.LinearOperatorDiag([3.0, 3.0], is_positive_definite=True, name='op3')
        with self.cached_session():
            op_sum = add_operators([op1, op2, op3])
            self.assertEqual(1, len(op_sum))
            op = op_sum[0]
            self.assertTrue(isinstance(op, linalg_lib.LinearOperatorDiag))
            self.assertAllClose([[6.0, 0.0], [0.0, 6.0]], op.to_dense())
            self.assertTrue(op.is_positive_definite)
            self.assertTrue(op.is_self_adjoint)
            self.assertTrue(op.is_non_singular)

    @test_util.run_deprecated_v1
    def test_diag_tril_diag(self):
        if False:
            print('Hello World!')
        op1 = linalg.LinearOperatorDiag([1.0, 1.0], is_non_singular=True, name='diag_a')
        op2 = linalg.LinearOperatorLowerTriangular([[2.0, 0.0], [0.0, 2.0]], is_self_adjoint=True, is_non_singular=True, name='tril')
        op3 = linalg.LinearOperatorDiag([3.0, 3.0], is_non_singular=True, name='diag_b')
        with self.cached_session():
            op_sum = add_operators([op1, op2, op3])
            self.assertEqual(1, len(op_sum))
            op = op_sum[0]
            self.assertIsInstance(op, linalg_lib.LinearOperatorLowerTriangular)
            self.assertAllClose([[6.0, 0.0], [0.0, 6.0]], op.to_dense())
            self.assertTrue(op.is_self_adjoint)
            self.assertEqual(None, op.is_non_singular)

    @test_util.run_deprecated_v1
    def test_matrix_diag_tril_diag_uses_custom_name(self):
        if False:
            i = 10
            return i + 15
        op0 = linalg.LinearOperatorFullMatrix([[-1.0, -1.0], [-1.0, -1.0]], name='matrix')
        op1 = linalg.LinearOperatorDiag([1.0, 1.0], name='diag_a')
        op2 = linalg.LinearOperatorLowerTriangular([[2.0, 0.0], [1.5, 2.0]], name='tril')
        op3 = linalg.LinearOperatorDiag([3.0, 3.0], name='diag_b')
        with self.cached_session():
            op_sum = add_operators([op0, op1, op2, op3], operator_name='my_operator')
            self.assertEqual(1, len(op_sum))
            op = op_sum[0]
            self.assertIsInstance(op, linalg_lib.LinearOperatorFullMatrix)
            self.assertAllClose([[5.0, -1.0], [0.5, 5.0]], op.to_dense())
            self.assertEqual('my_operator', op.name)

    def test_incompatible_domain_dimensions_raises(self):
        if False:
            for i in range(10):
                print('nop')
        op1 = linalg.LinearOperatorFullMatrix(rng.rand(2, 3))
        op2 = linalg.LinearOperatorDiag(rng.rand(2, 4))
        with self.assertRaisesRegex(ValueError, 'must.*same `domain_dimension`'):
            add_operators([op1, op2])

    def test_incompatible_range_dimensions_raises(self):
        if False:
            return 10
        op1 = linalg.LinearOperatorFullMatrix(rng.rand(2, 3))
        op2 = linalg.LinearOperatorDiag(rng.rand(3, 3))
        with self.assertRaisesRegex(ValueError, 'must.*same `range_dimension`'):
            add_operators([op1, op2])

    def test_non_broadcastable_batch_shape_raises(self):
        if False:
            while True:
                i = 10
        op1 = linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3))
        op2 = linalg.LinearOperatorDiag(rng.rand(4, 3, 3))
        with self.assertRaisesRegex(ValueError, 'Incompatible shapes'):
            add_operators([op1, op2])

class LinearOperatorOrderOfAdditionTest(test.TestCase):
    """Test that the order of addition is done as specified by tiers."""

    def test_tier_0_additions_done_in_tier_0(self):
        if False:
            i = 10
            return i + 15
        diag1 = linalg.LinearOperatorDiag([1.0])
        diag2 = linalg.LinearOperatorDiag([1.0])
        diag3 = linalg.LinearOperatorDiag([1.0])
        addition_tiers = [[linear_operator_addition._AddAndReturnDiag()], [_BadAdder()]]
        op_sum = add_operators([diag1, diag2, diag3], addition_tiers=addition_tiers)
        self.assertEqual(1, len(op_sum))
        self.assertIsInstance(op_sum[0], linalg.LinearOperatorDiag)

    def test_tier_1_additions_done_by_tier_1(self):
        if False:
            while True:
                i = 10
        diag1 = linalg.LinearOperatorDiag([1.0])
        diag2 = linalg.LinearOperatorDiag([1.0])
        tril = linalg.LinearOperatorLowerTriangular([[1.0]])
        addition_tiers = [[linear_operator_addition._AddAndReturnDiag()], [linear_operator_addition._AddAndReturnTriL()], [_BadAdder()]]
        op_sum = add_operators([diag1, diag2, tril], addition_tiers=addition_tiers)
        self.assertEqual(1, len(op_sum))
        self.assertIsInstance(op_sum[0], linalg.LinearOperatorLowerTriangular)

    def test_tier_1_additions_done_by_tier_1_with_order_flipped(self):
        if False:
            print('Hello World!')
        diag1 = linalg.LinearOperatorDiag([1.0])
        diag2 = linalg.LinearOperatorDiag([1.0])
        tril = linalg.LinearOperatorLowerTriangular([[1.0]])
        addition_tiers = [[linear_operator_addition._AddAndReturnTriL()], [linear_operator_addition._AddAndReturnDiag()], [_BadAdder()]]
        op_sum = add_operators([diag1, diag2, tril], addition_tiers=addition_tiers)
        self.assertEqual(1, len(op_sum))
        self.assertIsInstance(op_sum[0], linalg.LinearOperatorLowerTriangular)

    @test_util.run_deprecated_v1
    def test_cannot_add_everything_so_return_more_than_one_operator(self):
        if False:
            i = 10
            return i + 15
        diag1 = linalg.LinearOperatorDiag([1.0])
        diag2 = linalg.LinearOperatorDiag([2.0])
        tril5 = linalg.LinearOperatorLowerTriangular([[5.0]])
        addition_tiers = [[linear_operator_addition._AddAndReturnDiag()]]
        op_sum = add_operators([diag1, diag2, tril5], addition_tiers=addition_tiers)
        self.assertEqual(2, len(op_sum))
        found_diag = False
        found_tril = False
        with self.cached_session():
            for op in op_sum:
                if isinstance(op, linalg.LinearOperatorDiag):
                    found_diag = True
                    self.assertAllClose([[3.0]], op.to_dense())
                if isinstance(op, linalg.LinearOperatorLowerTriangular):
                    found_tril = True
                    self.assertAllClose([[5.0]], op.to_dense())
            self.assertTrue(found_diag and found_tril)

    def test_intermediate_tier_is_not_skipped(self):
        if False:
            for i in range(10):
                print('nop')
        diag1 = linalg.LinearOperatorDiag([1.0])
        diag2 = linalg.LinearOperatorDiag([1.0])
        tril = linalg.LinearOperatorLowerTriangular([[1.0]])
        addition_tiers = [[linear_operator_addition._AddAndReturnDiag()], [_BadAdder()], [linear_operator_addition._AddAndReturnTriL()]]
        with self.assertRaisesRegex(AssertionError, 'BadAdder.can_add called'):
            add_operators([diag1, diag2, tril], addition_tiers=addition_tiers)

class AddAndReturnScaledIdentityTest(test.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._adder = linear_operator_addition._AddAndReturnScaledIdentity()

    @test_util.run_deprecated_v1
    def test_identity_plus_identity(self):
        if False:
            for i in range(10):
                print('nop')
        id1 = linalg.LinearOperatorIdentity(num_rows=2)
        id2 = linalg.LinearOperatorIdentity(num_rows=2, batch_shape=[3])
        hints = linear_operator_addition._Hints(is_positive_definite=True, is_non_singular=True)
        self.assertTrue(self._adder.can_add(id1, id2))
        operator = self._adder.add(id1, id2, 'my_operator', hints)
        self.assertIsInstance(operator, linalg.LinearOperatorScaledIdentity)
        with self.cached_session():
            self.assertAllClose(2 * linalg_ops.eye(num_rows=2, batch_shape=[3]), operator.to_dense())
        self.assertTrue(operator.is_positive_definite)
        self.assertTrue(operator.is_non_singular)
        self.assertEqual('my_operator', operator.name)

    @test_util.run_deprecated_v1
    def test_identity_plus_scaled_identity(self):
        if False:
            print('Hello World!')
        id1 = linalg.LinearOperatorIdentity(num_rows=2, batch_shape=[3])
        id2 = linalg.LinearOperatorScaledIdentity(num_rows=2, multiplier=2.2)
        hints = linear_operator_addition._Hints(is_positive_definite=True, is_non_singular=True)
        self.assertTrue(self._adder.can_add(id1, id2))
        operator = self._adder.add(id1, id2, 'my_operator', hints)
        self.assertIsInstance(operator, linalg.LinearOperatorScaledIdentity)
        with self.cached_session():
            self.assertAllClose(3.2 * linalg_ops.eye(num_rows=2, batch_shape=[3]), operator.to_dense())
        self.assertTrue(operator.is_positive_definite)
        self.assertTrue(operator.is_non_singular)
        self.assertEqual('my_operator', operator.name)

    @test_util.run_deprecated_v1
    def test_scaled_identity_plus_scaled_identity(self):
        if False:
            for i in range(10):
                print('nop')
        id1 = linalg.LinearOperatorScaledIdentity(num_rows=2, multiplier=[2.2, 2.2, 2.2])
        id2 = linalg.LinearOperatorScaledIdentity(num_rows=2, multiplier=-1.0)
        hints = linear_operator_addition._Hints(is_positive_definite=True, is_non_singular=True)
        self.assertTrue(self._adder.can_add(id1, id2))
        operator = self._adder.add(id1, id2, 'my_operator', hints)
        self.assertIsInstance(operator, linalg.LinearOperatorScaledIdentity)
        with self.cached_session():
            self.assertAllClose(1.2 * linalg_ops.eye(num_rows=2, batch_shape=[3]), operator.to_dense())
        self.assertTrue(operator.is_positive_definite)
        self.assertTrue(operator.is_non_singular)
        self.assertEqual('my_operator', operator.name)

class AddAndReturnDiagTest(test.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._adder = linear_operator_addition._AddAndReturnDiag()

    @test_util.run_deprecated_v1
    def test_identity_plus_identity_returns_diag(self):
        if False:
            return 10
        id1 = linalg.LinearOperatorIdentity(num_rows=2)
        id2 = linalg.LinearOperatorIdentity(num_rows=2, batch_shape=[3])
        hints = linear_operator_addition._Hints(is_positive_definite=True, is_non_singular=True)
        self.assertTrue(self._adder.can_add(id1, id2))
        operator = self._adder.add(id1, id2, 'my_operator', hints)
        self.assertIsInstance(operator, linalg.LinearOperatorDiag)
        with self.cached_session():
            self.assertAllClose(2 * linalg_ops.eye(num_rows=2, batch_shape=[3]), operator.to_dense())
        self.assertTrue(operator.is_positive_definite)
        self.assertTrue(operator.is_non_singular)
        self.assertEqual('my_operator', operator.name)

    @test_util.run_deprecated_v1
    def test_diag_plus_diag(self):
        if False:
            for i in range(10):
                print('nop')
        diag1 = rng.rand(2, 3, 4)
        diag2 = rng.rand(4)
        op1 = linalg.LinearOperatorDiag(diag1)
        op2 = linalg.LinearOperatorDiag(diag2)
        hints = linear_operator_addition._Hints(is_positive_definite=True, is_non_singular=True)
        self.assertTrue(self._adder.can_add(op1, op2))
        operator = self._adder.add(op1, op2, 'my_operator', hints)
        self.assertIsInstance(operator, linalg.LinearOperatorDiag)
        with self.cached_session():
            self.assertAllClose(linalg.LinearOperatorDiag(diag1 + diag2).to_dense(), operator.to_dense())
        self.assertTrue(operator.is_positive_definite)
        self.assertTrue(operator.is_non_singular)
        self.assertEqual('my_operator', operator.name)

class AddAndReturnTriLTest(test.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._adder = linear_operator_addition._AddAndReturnTriL()

    @test_util.run_deprecated_v1
    def test_diag_plus_tril(self):
        if False:
            i = 10
            return i + 15
        diag = linalg.LinearOperatorDiag([1.0, 2.0])
        tril = linalg.LinearOperatorLowerTriangular([[10.0, 0.0], [30.0, 0.0]])
        hints = linear_operator_addition._Hints(is_positive_definite=True, is_non_singular=True)
        self.assertTrue(self._adder.can_add(diag, diag))
        self.assertTrue(self._adder.can_add(diag, tril))
        operator = self._adder.add(diag, tril, 'my_operator', hints)
        self.assertIsInstance(operator, linalg.LinearOperatorLowerTriangular)
        with self.cached_session():
            self.assertAllClose([[11.0, 0.0], [30.0, 2.0]], operator.to_dense())
        self.assertTrue(operator.is_positive_definite)
        self.assertTrue(operator.is_non_singular)
        self.assertEqual('my_operator', operator.name)

class AddAndReturnMatrixTest(test.TestCase):

    def setUp(self):
        if False:
            return 10
        self._adder = linear_operator_addition._AddAndReturnMatrix()

    @test_util.run_deprecated_v1
    def test_diag_plus_diag(self):
        if False:
            return 10
        diag1 = linalg.LinearOperatorDiag([1.0, 2.0])
        diag2 = linalg.LinearOperatorDiag([-1.0, 3.0])
        hints = linear_operator_addition._Hints(is_positive_definite=False, is_non_singular=False)
        self.assertTrue(self._adder.can_add(diag1, diag2))
        operator = self._adder.add(diag1, diag2, 'my_operator', hints)
        self.assertIsInstance(operator, linalg.LinearOperatorFullMatrix)
        with self.cached_session():
            self.assertAllClose([[0.0, 0.0], [0.0, 5.0]], operator.to_dense())
        self.assertFalse(operator.is_positive_definite)
        self.assertFalse(operator.is_non_singular)
        self.assertEqual('my_operator', operator.name)
if __name__ == '__main__':
    test.main()