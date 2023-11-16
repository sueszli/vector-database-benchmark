"""Tests for qiskit.quantum_info.analysis"""
import unittest
from qiskit.result import Counts, QuasiDistribution, ProbDistribution, sampled_expectation_value
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.opflow import PauliOp, PauliSumOp
from qiskit.test import QiskitTestCase
PROBS = {'1000': 0.0022, '1001': 0.0045, '1110': 0.0081, '0001': 0.0036, '0010': 0.0319, '0101': 0.001, '1100': 0.0008, '1010': 0.0009, '1111': 0.3951, '0011': 0.0007, '0111': 0.01, '0000': 0.4666, '1101': 0.0355, '1011': 0.0211, '0110': 0.0081, '0100': 0.0099}

class TestSampledExpval(QiskitTestCase):
    """Test sampled expectation values"""

    def test_simple(self):
        if False:
            return 10
        'Test that basic exp values work'
        dist2 = {'00': 0.5, '11': 0.5}
        dist3 = {'000': 0.5, '111': 0.5}
        self.assertAlmostEqual(sampled_expectation_value(dist2, 'ZZ'), 1.0)
        self.assertAlmostEqual(sampled_expectation_value(dist3, 'ZZZ'), 0.0)
        self.assertAlmostEqual(sampled_expectation_value(dist3, 'III'), 1.0)
        self.assertAlmostEqual(sampled_expectation_value(dist2, 'IZ'), 0.0)
        self.assertAlmostEqual(sampled_expectation_value(dist2, 'ZI'), 0.0)
        self.assertAlmostEqual(sampled_expectation_value(PROBS, 'ZZZZ'), 0.7554)

    def test_same(self):
        if False:
            while True:
                i = 10
        'Test that all operators agree with each other for counts input'
        ans = 0.9356
        counts = Counts({'001': 67, '110': 113, '100': 83, '011': 205, '111': 4535, '101': 100, '010': 42, '000': 4855})
        oper = 'IZZ'
        exp1 = sampled_expectation_value(counts, oper)
        self.assertAlmostEqual(exp1, ans)
        exp2 = sampled_expectation_value(counts, Pauli(oper))
        self.assertAlmostEqual(exp2, ans)
        with self.assertWarns(DeprecationWarning):
            exp3 = sampled_expectation_value(counts, PauliOp(Pauli(oper)))
        self.assertAlmostEqual(exp3, ans)
        spo = SparsePauliOp([oper], coeffs=[1])
        with self.assertWarns(DeprecationWarning):
            exp4 = sampled_expectation_value(counts, PauliSumOp(spo, coeff=2))
        self.assertAlmostEqual(exp4, 2 * ans)
        exp5 = sampled_expectation_value(counts, SparsePauliOp.from_list([[oper, 1]]))
        self.assertAlmostEqual(exp5, ans)

    def test_asym_ops(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that asymmetric exp values work'
        dist = QuasiDistribution(PROBS)
        self.assertAlmostEqual(sampled_expectation_value(dist, '0III'), 0.5318)
        self.assertAlmostEqual(sampled_expectation_value(dist, 'III0'), 0.5285)
        self.assertAlmostEqual(sampled_expectation_value(dist, '1011'), 0.0211)

    def test_probdist(self):
        if False:
            print('Hello World!')
        'Test that ProbDistro'
        dist = ProbDistribution(PROBS)
        result = sampled_expectation_value(dist, 'IZIZ')
        self.assertAlmostEqual(result, 0.8864)
        result2 = sampled_expectation_value(dist, '00ZI')
        self.assertAlmostEqual(result2, 0.4376)
if __name__ == '__main__':
    unittest.main(verbosity=2)