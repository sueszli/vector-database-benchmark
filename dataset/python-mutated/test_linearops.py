"""Equivalence tests for quantum channel methods."""
import unittest
import numpy as np
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.channel.choi import Choi
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.kraus import Kraus
from qiskit.quantum_info.operators.channel.stinespring import Stinespring
from qiskit.quantum_info.operators.channel.ptm import PTM
from qiskit.quantum_info.operators.channel.chi import Chi
from .channel_test_case import ChannelTestCase

class TestEquivalence(ChannelTestCase):
    """Tests for channel equivalence for linear operations.

    This tests that addition, subtraction, multiplication and negation
    work for all representations as if they were performed in the SuperOp
    representation.s equivalent to performing the same
    operations in other representations.
    """

    def _compare_add_to_superop(self, rep, dim, samples, unitary=False):
        if False:
            i = 10
            return i + 15
        'Test channel addition is equivalent to SuperOp'
        for _ in range(samples):
            if unitary:
                mat1 = self.rand_matrix(dim, dim)
                mat2 = self.rand_matrix(dim, dim)
                sop1 = np.kron(np.conj(mat1), mat1)
                sop2 = np.kron(np.conj(mat2), mat2)
            else:
                sop1 = self.rand_matrix(dim * dim, dim * dim)
                sop2 = self.rand_matrix(dim * dim, dim * dim)
            target = SuperOp(sop1 + sop2)
            channel = SuperOp(rep(SuperOp(sop1))._add(rep(SuperOp(sop2))))
            self.assertEqual(channel, target)

    def _compare_subtract_to_superop(self, rep, dim, samples, unitary=False):
        if False:
            i = 10
            return i + 15
        'Test channel subtraction is equivalent to SuperOp'
        for _ in range(samples):
            if unitary:
                mat1 = self.rand_matrix(dim, dim)
                mat2 = self.rand_matrix(dim, dim)
                sop1 = np.kron(np.conj(mat1), mat1)
                sop2 = np.kron(np.conj(mat2), mat2)
            else:
                sop1 = self.rand_matrix(dim * dim, dim * dim)
                sop2 = self.rand_matrix(dim * dim, dim * dim)
            target = SuperOp(sop1 - sop2)
            channel = SuperOp(rep(SuperOp(sop1))._add(rep(-SuperOp(sop2))))
            self.assertEqual(channel, target)

    def _compare_subtract_operator_to_superop(self, rep, dim, samples, unitary=False):
        if False:
            print('Hello World!')
        'Test channel addition is equivalent to SuperOp'
        for _ in range(samples):
            if unitary:
                mat1 = self.rand_matrix(dim, dim)
                sop1 = np.kron(np.conj(mat1), mat1)
            else:
                sop1 = self.rand_matrix(dim * dim, dim * dim)
            mat2 = self.rand_matrix(dim, dim)
            target = SuperOp(sop1) - SuperOp(Operator(mat2))
            channel = SuperOp(rep(SuperOp(sop1)) - Operator(mat2))
            self.assertEqual(channel, target)

    def _compare_multiply_to_superop(self, rep, dim, samples, unitary=False):
        if False:
            while True:
                i = 10
        'Test channel scalar multiplication is equivalent to SuperOp'
        for _ in range(samples):
            if unitary:
                mat1 = self.rand_matrix(dim, dim)
                sop1 = np.kron(np.conj(mat1), mat1)
            else:
                sop1 = self.rand_matrix(dim * dim, dim * dim)
            val = 0.7
            target = SuperOp(val * sop1)
            channel = SuperOp(rep(SuperOp(sop1))._multiply(val))
            self.assertEqual(channel, target)

    def _compare_negate_to_superop(self, rep, dim, samples, unitary=False):
        if False:
            return 10
        'Test negative channel is equivalent to SuperOp'
        for _ in range(samples):
            if unitary:
                mat1 = self.rand_matrix(dim, dim)
                sop1 = np.kron(np.conj(mat1), mat1)
            else:
                sop1 = self.rand_matrix(dim * dim, dim * dim)
            target = SuperOp(-1 * sop1)
            channel = SuperOp(-rep(SuperOp(sop1)))
            self.assertEqual(channel, target)

    def _check_add_other_reps(self, chan):
        if False:
            for i in range(10):
                print('nop')
        'Check addition works for other representations'
        current_rep = chan.__class__
        other_reps = [Operator, Choi, SuperOp, Kraus, Stinespring, Chi, PTM]
        for rep in other_reps:
            self.assertEqual(current_rep, chan._add(rep(chan)).__class__)

    def _check_subtract_other_reps(self, chan):
        if False:
            while True:
                i = 10
        'Check subtraction works for other representations'
        current_rep = chan.__class__
        other_reps = [Operator, Choi, SuperOp, Kraus, Stinespring, Chi, PTM]
        for rep in other_reps:
            self.assertEqual(current_rep, chan._add(-rep(chan)).__class__)

    def test_choi_add(self):
        if False:
            for i in range(10):
                print('nop')
        'Test addition of Choi matrices is correct.'
        self._compare_add_to_superop(Choi, 4, 10)

    def test_kraus_add(self):
        if False:
            while True:
                i = 10
        'Test addition of Kraus matrices is correct.'
        self._compare_add_to_superop(Kraus, 4, 10)

    def test_stinespring_add(self):
        if False:
            print('Hello World!')
        'Test addition of Stinespring matrices is correct.'
        self._compare_add_to_superop(Stinespring, 4, 10)

    def test_chi_add(self):
        if False:
            print('Hello World!')
        'Test addition of Chi matrices is correct.'
        self._compare_add_to_superop(Chi, 4, 10)

    def test_ptm_add(self):
        if False:
            while True:
                i = 10
        'Test addition of PTM matrices is correct.'
        self._compare_add_to_superop(PTM, 4, 10)

    def test_choi_subtract(self):
        if False:
            print('Hello World!')
        'Test subtraction of Choi matrices is correct.'
        self._compare_subtract_to_superop(Choi, 4, 10)

    def test_kraus_subtract(self):
        if False:
            return 10
        'Test subtraction of Kraus matrices is correct.'
        self._compare_subtract_to_superop(Kraus, 4, 10)

    def test_stinespring_subtract(self):
        if False:
            return 10
        'Test subtraction of Stinespring matrices is correct.'
        self._compare_subtract_to_superop(Stinespring, 4, 10)

    def test_chi_subtract(self):
        if False:
            for i in range(10):
                print('nop')
        'Test subtraction of Chi matrices is correct.'
        self._compare_subtract_to_superop(Chi, 4, 10)

    def test_ptm_subtract(self):
        if False:
            i = 10
            return i + 15
        'Test subtraction of PTM matrices is correct.'
        self._compare_subtract_to_superop(PTM, 4, 10)

    def test_choi_subtract_operator(self):
        if False:
            print('Hello World!')
        'Test subtraction of Operator from Choi is correct.'
        self._compare_subtract_operator_to_superop(Choi, 4, 10)

    def test_kraus_subtract_operator(self):
        if False:
            return 10
        'Test subtraction of Operator from Kraus is correct.'
        self._compare_subtract_operator_to_superop(Kraus, 4, 10)

    def test_stinespring_subtract_operator(self):
        if False:
            while True:
                i = 10
        'Test subtraction of Operator from Stinespring is correct.'
        self._compare_subtract_operator_to_superop(Stinespring, 4, 10)

    def test_chi_subtract_operator(self):
        if False:
            while True:
                i = 10
        'Test subtraction of Operator from Chi is correct.'
        self._compare_subtract_operator_to_superop(Chi, 4, 10)

    def test_ptm_subtract_operator(self):
        if False:
            print('Hello World!')
        'Test subtraction of Operator from PTM is correct.'
        self._compare_subtract_operator_to_superop(PTM, 4, 10)

    def test_choi_multiply(self):
        if False:
            return 10
        'Test scalar multiplication of Choi matrices is correct.'
        self._compare_multiply_to_superop(Choi, 4, 10)

    def test_kraus_multiply(self):
        if False:
            while True:
                i = 10
        'Test scalar multiplication of Kraus matrices is correct.'
        self._compare_multiply_to_superop(Kraus, 4, 10)

    def test_stinespring_multiply(self):
        if False:
            for i in range(10):
                print('nop')
        'Test scalar multiplication of Stinespring matrices is correct.'
        self._compare_multiply_to_superop(Stinespring, 4, 10)

    def test_chi_multiply(self):
        if False:
            while True:
                i = 10
        'Test scalar multiplication of Chi matrices is correct.'
        self._compare_multiply_to_superop(Chi, 4, 10)

    def test_ptm_multiply(self):
        if False:
            i = 10
            return i + 15
        'Test scalar multiplication of PTM matrices is correct.'
        self._compare_multiply_to_superop(PTM, 4, 10)

    def test_choi_add_other_rep(self):
        if False:
            print('Hello World!')
        'Test addition of Choi matrices is correct.'
        chan = Choi(self.choiI)
        self._check_add_other_reps(chan)

    def test_superop_add_other_rep(self):
        if False:
            i = 10
            return i + 15
        'Test addition of SuperOp matrices is correct.'
        chan = SuperOp(self.sopI)
        self._check_add_other_reps(chan)

    def test_kraus_add_other_rep(self):
        if False:
            i = 10
            return i + 15
        'Test addition of Kraus matrices is correct.'
        chan = Kraus(self.UI)
        self._check_add_other_reps(chan)

    def test_stinespring_add_other_rep(self):
        if False:
            while True:
                i = 10
        'Test addition of Stinespring matrices is correct.'
        chan = Stinespring(self.UI)
        self._check_add_other_reps(chan)

    def test_chi_add_other_rep(self):
        if False:
            while True:
                i = 10
        'Test addition of Chi matrices is correct.'
        chan = Chi(self.chiI)
        self._check_add_other_reps(chan)

    def test_ptm_add_other_rep(self):
        if False:
            print('Hello World!')
        'Test addition of PTM matrices is correct.'
        chan = PTM(self.ptmI)
        self._check_add_other_reps(chan)

    def test_choi_subtract_other_rep(self):
        if False:
            return 10
        'Test subtraction of Choi matrices is correct.'
        chan = Choi(self.choiI)
        self._check_subtract_other_reps(chan)

    def test_superop_subtract_other_rep(self):
        if False:
            for i in range(10):
                print('nop')
        'Test subtraction of SuperOp matrices is correct.'
        chan = SuperOp(self.sopI)
        self._check_subtract_other_reps(chan)

    def test_kraus_subtract_other_rep(self):
        if False:
            for i in range(10):
                print('nop')
        'Test subtraction of Kraus matrices is correct.'
        chan = Kraus(self.UI)
        self._check_subtract_other_reps(chan)

    def test_stinespring_subtract_other_rep(self):
        if False:
            print('Hello World!')
        'Test subtraction of Stinespring matrices is correct.'
        chan = Stinespring(self.UI)
        self._check_subtract_other_reps(chan)

    def test_chi_subtract_other_rep(self):
        if False:
            while True:
                i = 10
        'Test subtraction of Chi matrices is correct.'
        chan = Chi(self.chiI)
        self._check_subtract_other_reps(chan)

    def test_ptm_subtract_other_rep(self):
        if False:
            i = 10
            return i + 15
        'Test subtraction of PTM matrices is correct.'
        chan = PTM(self.ptmI)
        self._check_subtract_other_reps(chan)
if __name__ == '__main__':
    unittest.main()