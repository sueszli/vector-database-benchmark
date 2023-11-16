"""Tests for visualization of latex state and unitary drawers"""
import unittest
from qiskit.quantum_info import Statevector
from qiskit.visualization.state_visualization import state_drawer
from .visualization import QiskitVisualizationTestCase

class TestLatexStateDrawer(QiskitVisualizationTestCase):
    """Qiskit state and unitary latex drawer."""

    def test_state(self):
        if False:
            for i in range(10):
                print('nop')
        'Test latex state vector drawer works with default settings.'
        sv = Statevector.from_label('+-rl')
        output = state_drawer(sv, 'latex_source')
        expected_output = '\\frac{1}{4} |0000\\rangle- \\frac{i}{4} |0001\\rangle+\\frac{i}{4} |0010\\rangle+\\frac{1}{4} |0011\\rangle- \\frac{1}{4} |0100\\rangle+\\frac{i}{4} |0101\\rangle + \\ldots +\\frac{1}{4} |1011\\rangle- \\frac{1}{4} |1100\\rangle+\\frac{i}{4} |1101\\rangle- \\frac{i}{4} |1110\\rangle- \\frac{1}{4} |1111\\rangle'
        self.assertEqual(output, expected_output)

    def test_state_max_size(self):
        if False:
            return 10
        'Test `max_size` parameter for latex ket notation.'
        sv = Statevector.from_label('+-rl')
        output = state_drawer(sv, 'latex_source', max_size=4)
        expected_output = '\\frac{1}{4} |0000\\rangle- \\frac{i}{4} |0001\\rangle + \\ldots - \\frac{1}{4} |1111\\rangle'
        self.assertEqual(output, expected_output)
if __name__ == '__main__':
    unittest.main(verbosity=2)