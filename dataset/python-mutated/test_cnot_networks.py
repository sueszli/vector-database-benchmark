"""
Tests building up CNOT unit structures.
"""
from test.python.transpiler.aqc.sample_data import CARTAN_4, CARTAN_3
import numpy as np
from ddt import ddt, data, unpack
from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.aqc import make_cnot_network

@ddt
class TestCNOTNetworks(QiskitTestCase):
    """Tests constructing various CNOT structures."""

    @data((3, 'spin', 'full', 3, [[0, 1, 0], [1, 2, 1]]), (4, 'spin', 'full', 4, [[0, 2, 1, 0], [1, 3, 2, 1]]), (3, 'sequ', 'full', 4, [[0, 0, 1, 0], [1, 2, 2, 1]]), (4, 'sequ', 'full', 7, [[0, 0, 0, 1, 1, 2, 0], [1, 2, 3, 2, 3, 3, 1]]), (3, 'sequ', 'line', 3, [[0, 1, 0], [1, 2, 1]]), (4, 'sequ', 'line', 4, [[0, 1, 2, 0], [1, 2, 3, 1]]), (3, 'sequ', 'star', 3, [[0, 0, 0], [1, 2, 1]]), (4, 'sequ', 'star', 4, [[0, 0, 0, 0], [1, 2, 3, 1]]), (3, 'cart', 'full', 0, CARTAN_3), (4, 'cart', 'full', 0, CARTAN_4), (3, 'cyclic_spin', 'full', 3, [[0, 1, 0], [1, 2, 1]]), (4, 'cyclic_spin', 'full', 5, [[0, 2, 1, 3, 0], [1, 3, 2, 0, 1]]), (3, 'cyclic_line', 'line', 4, [[0, 1, 2, 0], [1, 2, 0, 1]]), (4, 'cyclic_line', 'line', 5, [[0, 1, 2, 3, 0], [1, 2, 3, 0, 1]]))
    @unpack
    def test_basic_networks(self, num_qubits, network_layout, connectivity, depth, output):
        if False:
            return 10
        'Tests basic CNOT networks.'
        cnots = make_cnot_network(num_qubits=num_qubits, network_layout=network_layout, connectivity_type=connectivity, depth=depth)
        np.testing.assert_array_equal(cnots, output)