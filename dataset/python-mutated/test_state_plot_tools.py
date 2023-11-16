"""Tests for functions used in state visualization"""
import unittest
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.visualization.state_visualization import _paulivec_data
from qiskit.test import QiskitTestCase

class TestStatePlotTools(QiskitTestCase):
    """State Plotting Tools"""

    def test_state_paulivec(self):
        if False:
            print('Hello World!')
        'Test paulivec.'
        sv = Statevector.from_label('+-rl')
        output = _paulivec_data(sv)
        labels = ['IIII', 'IIIY', 'IIYI', 'IIYY', 'IXII', 'IXIY', 'IXYI', 'IXYY', 'XIII', 'XIIY', 'XIYI', 'XIYY', 'XXII', 'XXIY', 'XXYI', 'XXYY']
        values = [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1]
        self.assertEqual(output[0], labels)
        self.assertTrue(np.allclose(output[1], values))
if __name__ == '__main__':
    unittest.main(verbosity=2)