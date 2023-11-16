"""
Tests for qiskit-terra/qiskit/quantum_info/synthesis/xx_decompose/qiskit.py .
"""
from statistics import mean
import unittest
import ddt
import numpy as np
from scipy.stats import unitary_group
import qiskit
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.synthesis.xx_decompose.decomposer import XXDecomposer, TwoQubitWeylDecomposition
from .utilities import canonical_matrix
EPSILON = 1e-08

@ddt.ddt
class TestXXDecomposer(unittest.TestCase):
    """Tests for decomposition of two-qubit unitaries over discrete gates from XX family."""
    decomposer = XXDecomposer(euler_basis='PSX')

    def __init__(self, *args, seed=42, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self._random_state = np.random.Generator(np.random.PCG64(seed))

    def test_random_compilation(self):
        if False:
            return 10
        'Test that compilation gives correct results.'
        for _ in range(100):
            unitary = unitary_group.rvs(4, random_state=self._random_state)
            unitary /= np.linalg.det(unitary) ** (1 / 4)
            circuit = self.decomposer(unitary, approximate=False)
            decomposed_unitary = Operator(circuit).data
            self.assertTrue(np.all(unitary - decomposed_unitary < EPSILON))

    def test_compilation_determinism(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that compilation is stable under multiple calls.'
        for _ in range(10):
            unitary = unitary_group.rvs(4, random_state=self._random_state)
            unitary /= np.linalg.det(unitary) ** (1 / 4)
            circuit1 = self.decomposer(unitary, approximate=False)
            circuit2 = self.decomposer(unitary, approximate=False)
            self.assertEqual(circuit1, circuit2)

    @ddt.data(np.pi / 3, np.pi / 5, np.pi / 2)
    def test_default_embodiment(self, angle):
        if False:
            i = 10
            return i + 15
        'Test that _default_embodiment actually does yield XX gates.'
        embodiment = self.decomposer._default_embodiment(angle)
        embodiment_matrix = Operator(embodiment).data
        self.assertTrue(np.all(canonical_matrix(angle, 0, 0) - embodiment_matrix < EPSILON))

    def test_check_embodiment(self):
        if False:
            while True:
                i = 10
        'Test that XXDecomposer._check_embodiments correctly diagnoses il/legal embodiments.'
        good_angle = np.pi / 2
        good_embodiment = qiskit.QuantumCircuit(2)
        good_embodiment.h(0)
        good_embodiment.cx(0, 1)
        good_embodiment.h(1)
        good_embodiment.rz(np.pi / 2, 0)
        good_embodiment.rz(np.pi / 2, 1)
        good_embodiment.h(1)
        good_embodiment.h(0)
        good_embodiment.global_phase += np.pi / 4
        bad_angle = np.pi / 10
        bad_embodiment = qiskit.QuantumCircuit(2)
        XXDecomposer(embodiments={good_angle: good_embodiment})
        self.assertRaises(qiskit.exceptions.QiskitError, XXDecomposer, embodiments={bad_angle: bad_embodiment})

    def test_compilation_improvement(self):
        if False:
            i = 10
            return i + 15
        'Test that compilation to CX, CX/2, CX/3 improves over CX alone.'
        (slope, offset) = (64 * 90 / 1000000, 909 / 1000000 + 1 / 1000)
        strength_table = self.decomposer._strength_to_infidelity(basis_fidelity={strength: 1 - (slope * strength / (np.pi / 2) + offset) for strength in [np.pi / 2, np.pi / 4, np.pi / 6]}, approximate=True)
        limited_strength_table = {np.pi / 2: strength_table[np.pi / 2]}
        clever_costs = []
        naive_costs = []
        for _ in range(200):
            unitary = unitary_group.rvs(4, random_state=self._random_state)
            unitary /= np.linalg.det(unitary) ** (1 / 4)
            weyl_decomposition = TwoQubitWeylDecomposition(unitary)
            target = [getattr(weyl_decomposition, x) for x in ('a', 'b', 'c')]
            if target[-1] < -EPSILON:
                target = [np.pi / 2 - target[0], target[1], -target[2]]
            clever_costs.append(self.decomposer._best_decomposition(target, strength_table)['cost'])
            naive_costs.append(self.decomposer._best_decomposition(target, limited_strength_table)['cost'])
        self.assertAlmostEqual(mean(clever_costs), 0.01445, delta=0.005)
        self.assertAlmostEqual(mean(naive_costs), 0.02058, delta=0.005)

    def test_error_on_empty_basis_fidelity(self):
        if False:
            print('Hello World!')
        'Test synthesizing entangling gate with no entangling basis fails.'
        decomposer = XXDecomposer(basis_fidelity={})
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        mat = Operator(qc).to_matrix()
        with self.assertRaisesRegex(qiskit.exceptions.QiskitError, 'Attempting to synthesize entangling gate with no controlled gates in basis set.'):
            decomposer(mat)

    def test_no_error_on_empty_basis_fidelity_trivial_target(self):
        if False:
            i = 10
            return i + 15
        'Test synthesizing non-entangling gate with no entangling basis succeeds.'
        decomposer = XXDecomposer(basis_fidelity={})
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.y(1)
        mat = Operator(qc).to_matrix()
        dqc = decomposer(mat)
        self.assertTrue(np.allclose(mat, Operator(dqc).to_matrix()))