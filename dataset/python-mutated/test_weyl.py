"""
Tests for qiskit-terra/qiskit/quantum_info/synthesis/xx_decompose/weyl.py .
"""
from itertools import permutations
import unittest
import ddt
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.synthesis.xx_decompose.weyl import apply_reflection, apply_shift, canonical_rotation_circuit, reflection_options, shift_options
from .utilities import canonical_matrix
EPSILON = 0.001

@ddt.ddt
class TestMonodromyWeyl(unittest.TestCase):
    """Check Weyl action routines."""

    def __init__(self, *args, seed=42, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(seed)

    def test_reflections(self):
        if False:
            return 10
        'Check that reflection circuits behave as expected.'
        for name in reflection_options:
            coordinate = [self.rng.random() for _ in range(3)]
            (reflected_coordinate, reflection_circuit, reflection_phase) = apply_reflection(name, coordinate)
            original_matrix = canonical_matrix(*coordinate)
            reflected_matrix = canonical_matrix(*reflected_coordinate)
            reflect_matrix = Operator(reflection_circuit).data
            self.assertTrue(np.all(np.abs(reflect_matrix.conjugate().transpose(1, 0) @ original_matrix @ reflect_matrix - reflected_matrix * reflection_phase) < EPSILON))

    def test_shifts(self):
        if False:
            print('Hello World!')
        'Check that shift circuits behave as expected.'
        for name in shift_options:
            coordinate = [self.rng.random() for _ in range(3)]
            (shifted_coordinate, shift_circuit, shift_phase) = apply_shift(name, coordinate)
            original_matrix = canonical_matrix(*coordinate)
            shifted_matrix = canonical_matrix(*shifted_coordinate)
            shift_matrix = Operator(shift_circuit).data
            self.assertTrue(np.all(np.abs(original_matrix @ shift_matrix - shifted_matrix * shift_phase) < EPSILON))

    def test_rotations(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that rotation circuits behave as expected.'
        for permutation in permutations([0, 1, 2]):
            coordinate = [self.rng.random() for _ in range(3)]
            rotation_circuit = canonical_rotation_circuit(permutation[0], permutation[1])
            original_matrix = canonical_matrix(*coordinate)
            rotation_matrix = Operator(rotation_circuit).data
            rotated_matrix = canonical_matrix(coordinate[permutation[0]], coordinate[permutation[1]], coordinate[permutation[2]])
            self.assertTrue(np.all(np.abs(rotation_matrix.conjugate().transpose(1, 0) @ original_matrix @ rotation_matrix - rotated_matrix) < EPSILON))