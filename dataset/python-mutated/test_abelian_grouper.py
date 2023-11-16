"""Test Abelian Grouper"""
import random
import unittest
from itertools import combinations, product
from test.python.opflow import QiskitOpflowTestCase
from ddt import data, ddt, unpack
from qiskit.opflow import AbelianGrouper, commutator, I, OpflowError, Plus, SummedOp, X, Y, Z, Zero

@ddt
class TestAbelianGrouper(QiskitOpflowTestCase):
    """Abelian Grouper tests."""

    @data(*product(['h2_op', 'generic'], [True, False]))
    @unpack
    def test_abelian_grouper(self, pauli_op, is_summed_op):
        if False:
            return 10
        'Abelian grouper test'
        if pauli_op == 'h2_op':
            paulis = (-1.052373245772859 * I ^ I) + (0.39793742484318045 * I ^ Z) + (-0.39793742484318045 * Z ^ I) + (-0.01128010425623538 * Z ^ Z) + (0.18093119978423156 * X ^ X)
            num_groups = 2
        else:
            paulis = (I ^ I ^ X ^ X * 0.2) + (Z ^ Z ^ X ^ X * 0.3) + (Z ^ Z ^ Z ^ Z * 0.4) + (X ^ X ^ Z ^ Z * 0.5) + (X ^ X ^ X ^ X * 0.6) + (I ^ X ^ X ^ X * 0.7)
            num_groups = 4
        if is_summed_op:
            paulis = paulis.to_pauli_op()
        grouped_sum = AbelianGrouper().convert(paulis)
        self.assertEqual(len(grouped_sum.oplist), num_groups)
        for group in grouped_sum:
            for (op_1, op_2) in combinations(group, 2):
                if is_summed_op:
                    self.assertEqual(op_1 @ op_2, op_2 @ op_1)
                else:
                    self.assertTrue(commutator(op_1, op_2).is_zero())

    def test_ablian_grouper_no_commute(self):
        if False:
            for i in range(10):
                print('nop')
        'Abelian grouper test when non-PauliOp is given'
        ops = Zero ^ Plus + X ^ Y
        with self.assertRaises(OpflowError):
            _ = AbelianGrouper.group_subops(ops)

    @data(True, False)
    def test_group_subops(self, is_summed_op):
        if False:
            i = 10
            return i + 15
        'grouper subroutine test'
        paulis = (I ^ X) + (2 * X ^ X) + (3 * Z ^ Y)
        if is_summed_op:
            paulis = paulis.to_pauli_op()
        grouped_sum = AbelianGrouper.group_subops(paulis)
        self.assertEqual(len(grouped_sum), 2)
        with self.subTest('test group subops 1'):
            if is_summed_op:
                expected = SummedOp([SummedOp([I ^ X, 2.0 * X ^ X], abelian=True), SummedOp([3.0 * Z ^ Y], abelian=True)])
                self.assertEqual(grouped_sum, expected)
            else:
                self.assertSetEqual(frozenset((frozenset(grouped_sum[i].primitive.to_list()) for i in range(2))), frozenset({frozenset({('ZY', 3)}), frozenset({('IX', 1), ('XX', 2)})}))
        paulis = X + 2 * Y + 3 * Z
        if is_summed_op:
            paulis = paulis.to_pauli_op()
        grouped_sum = AbelianGrouper.group_subops(paulis)
        self.assertEqual(len(grouped_sum), 3)
        with self.subTest('test group subops 2'):
            if is_summed_op:
                self.assertEqual(grouped_sum, paulis)
            else:
                self.assertSetEqual(frozenset(sum((grouped_sum[i].primitive.to_list() for i in range(3)), [])), frozenset([('X', 1), ('Y', 2), ('Z', 3)]))

    @data(True, False)
    def test_abelian_grouper_random(self, is_summed_op):
        if False:
            return 10
        'Abelian grouper test with random paulis'
        random.seed(1234)
        k = 10
        n = 100
        num_tests = 20
        for _ in range(num_tests):
            paulis = []
            for _ in range(n):
                pauliop = 1
                for eachop in random.choices([I] * 5 + [X, Y, Z], k=k):
                    pauliop ^= eachop
                paulis.append(pauliop)
            pauli_sum = sum(paulis)
            if is_summed_op:
                pauli_sum = pauli_sum.to_pauli_op()
            grouped_sum = AbelianGrouper().convert(pauli_sum)
            for group in grouped_sum:
                for (op_1, op_2) in combinations(group, 2):
                    if is_summed_op:
                        self.assertEqual(op_1 @ op_2, op_2 @ op_1)
                    else:
                        self.assertTrue(commutator(op_1, op_2).is_zero())
if __name__ == '__main__':
    unittest.main()