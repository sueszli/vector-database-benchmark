"""Test library of graph state circuits."""
import unittest
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import GraphState
from qiskit.quantum_info import Clifford
from qiskit.test.base import QiskitTestCase

class TestGraphStateLibrary(QiskitTestCase):
    """Test the graph state circuit."""

    def assertGraphStateIsCorrect(self, adjacency_matrix, graph_state):
        if False:
            i = 10
            return i + 15
        'Check the stabilizers of the graph state against the expected stabilizers.\n        Based on https://arxiv.org/pdf/quant-ph/0307130.pdf, Eq. (6).\n        '
        stabilizers = [stabilizer[1:] for stabilizer in Clifford(graph_state).to_labels(mode='S')]
        expected_stabilizers = []
        num_vertices = len(adjacency_matrix)
        for vertex_a in range(num_vertices):
            stabilizer = [None] * num_vertices
            for vertex_b in range(num_vertices):
                if vertex_a == vertex_b:
                    stabilizer[vertex_a] = 'X'
                elif adjacency_matrix[vertex_a][vertex_b] != 0:
                    stabilizer[vertex_b] = 'Z'
                else:
                    stabilizer[vertex_b] = 'I'
            expected_stabilizers.append(''.join(stabilizer)[::-1])
        self.assertListEqual(expected_stabilizers, stabilizers)

    def test_graph_state(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify the GraphState by checking if the circuit has the expected stabilizers.'
        adjacency_matrix = [[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]]
        graph_state = GraphState(adjacency_matrix)
        self.assertGraphStateIsCorrect(adjacency_matrix, graph_state)

    def test_non_symmetric_raises(self):
        if False:
            return 10
        'Test that adjacency matrix is required to be symmetric.'
        adjacency_matrix = [[1, 1, 0], [1, 0, 1], [1, 1, 1]]
        with self.assertRaises(CircuitError):
            GraphState(adjacency_matrix)
if __name__ == '__main__':
    unittest.main()