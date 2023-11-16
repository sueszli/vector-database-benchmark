"""
Tests for qiskit-terra/qiskit/quantum_info/synthesis/xx_decompose/circuits.py .
"""
from operator import itemgetter
import unittest
import ddt
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RZGate, UnitaryGate
import qiskit.quantum_info.operators
from qiskit.quantum_info.synthesis.weyl import weyl_coordinates
from qiskit.quantum_info.synthesis.xx_decompose.circuits import decompose_xxyy_into_xxyy_xx, xx_circuit_step
from .utilities import canonical_matrix
EPSILON = 0.001

@ddt.ddt
class TestMonodromyCircuits(unittest.TestCase):
    """Check circuit synthesis step routines."""

    def __init__(self, *args, seed=42, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(seed)

    def _generate_xxyy_test_case(self):
        if False:
            while True:
                i = 10
        '\n        Generates a random pentuple of values (a_source, b_source), beta, (a_target, b_target) s.t.\n\n            CAN(a_source, b_source) * exp(-i(s ZI + t IZ)) * CAN(beta)\n                                    =\n            exp(-i(u ZI + v IZ)) * CAN(a_target, b_target) * exp(-i(x ZI + y IZ))\n\n        admits a solution in (s, t, u, v, x, y).\n\n        Returns (source_coordinate, interaction, target_coordinate).\n        '
        source_coordinate = [self.rng.random(), self.rng.random(), 0.0]
        source_coordinate = [source_coordinate[0] * np.pi / 8, source_coordinate[1] * source_coordinate[0] * np.pi / 8, 0.0]
        interaction = [self.rng.random() * np.pi / 8]
        z_angles = [self.rng.random() * np.pi / 8, self.rng.random() * np.pi / 8]
        prod = canonical_matrix(*source_coordinate) @ np.kron(RZGate(2 * z_angles[0]).to_matrix(), RZGate(2 * z_angles[1]).to_matrix()) @ canonical_matrix(interaction[0], 0.0, 0.0)
        target_coordinate = weyl_coordinates(prod)
        self.assertAlmostEqual(target_coordinate[-1], 0.0, delta=EPSILON)
        return (source_coordinate, interaction, target_coordinate)

    def test_decompose_xxyy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that decompose_xxyy_into_xxyy_xx correctly recovers decompositions.\n        '
        for _ in range(100):
            (source_coordinate, interaction, target_coordinate) = self._generate_xxyy_test_case()
            (r, s, u, v, x, y) = decompose_xxyy_into_xxyy_xx(target_coordinate[0], target_coordinate[1], source_coordinate[0], source_coordinate[1], interaction[0])
            prod = np.kron(RZGate(2 * r).to_matrix(), RZGate(2 * s).to_matrix()) @ canonical_matrix(*source_coordinate) @ np.kron(RZGate(2 * u).to_matrix(), RZGate(2 * v).to_matrix()) @ canonical_matrix(interaction[0], 0.0, 0.0) @ np.kron(RZGate(2 * x).to_matrix(), RZGate(2 * y).to_matrix())
            expected = canonical_matrix(*target_coordinate)
            self.assertTrue(np.all(np.abs(prod - expected) < EPSILON))

    def test_xx_circuit_step(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that `xx_circuit_step` correctly generates prefix/affix circuits relating source\n        canonical coordinates to target canonical coordinates along prescribed interactions, all\n        randomly selected.\n        '
        for _ in range(100):
            (source_coordinate, interaction, target_coordinate) = self._generate_xxyy_test_case()
            source_embodiment = qiskit.QuantumCircuit(2)
            source_embodiment.append(UnitaryGate(canonical_matrix(*source_coordinate)), [0, 1])
            interaction_embodiment = qiskit.QuantumCircuit(2)
            interaction_embodiment.append(UnitaryGate(canonical_matrix(*interaction)), [0, 1])
            (prefix_circuit, affix_circuit) = itemgetter('prefix_circuit', 'affix_circuit')(xx_circuit_step(source_coordinate, interaction[0], target_coordinate, interaction_embodiment))
            target_embodiment = QuantumCircuit(2)
            target_embodiment.compose(prefix_circuit, inplace=True)
            target_embodiment.compose(source_embodiment, inplace=True)
            target_embodiment.compose(affix_circuit, inplace=True)
            self.assertTrue(np.all(np.abs(qiskit.quantum_info.operators.Operator(target_embodiment).data - canonical_matrix(*target_coordinate)) < EPSILON))