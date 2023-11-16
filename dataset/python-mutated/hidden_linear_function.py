"""Hidden Linear Function circuit."""
from typing import Union, List
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError

class HiddenLinearFunction(QuantumCircuit):
    """Circuit to solve the hidden linear function problem.

    The 2D Hidden Linear Function problem is determined by a 2D adjacency
    matrix A, where only elements that are nearest-neighbor on a grid have
    non-zero entries. Each row/column corresponds to one binary variable
    :math:`x_i`.

    The hidden linear function problem is as follows:

    Consider the quadratic form

    .. math::

        q(x) = \\sum_{i,j=1}^{n}{x_i x_j} ~(\\mathrm{mod}~ 4)

    and restrict :math:`q(x)` onto the nullspace of A. This results in a linear
    function.

    .. math::

        2 \\sum_{i=1}^{n}{z_i x_i} ~(\\mathrm{mod}~ 4)  \\forall  x \\in \\mathrm{Ker}(A)

    and the goal is to recover this linear function (equivalently a vector
    :math:`[z_0, ..., z_{n-1}]`). There can be multiple solutions.

    In [1] it is shown that the present circuit solves this problem
    on a quantum computer in constant depth, whereas any corresponding
    solution on a classical computer would require circuits that grow
    logarithmically with :math:`n`. Thus this circuit is an example
    of quantum advantage with shallow circuits.

    **Reference Circuit:**

        .. plot::

           from qiskit.circuit.library import HiddenLinearFunction
           from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
           A = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
           circuit = HiddenLinearFunction(A)
           _generate_circuit_library_visualization(circuit)

    **Reference:**

    [1] S. Bravyi, D. Gosset, R. Koenig, Quantum Advantage with Shallow Circuits, 2017.
    `arXiv:1704.00690 <https://arxiv.org/abs/1704.00690>`_
    """

    def __init__(self, adjacency_matrix: Union[List[List[int]], np.ndarray]) -> None:
        if False:
            return 10
        'Create new HLF circuit.\n\n        Args:\n            adjacency_matrix: a symmetric n-by-n list of 0-1 lists.\n                n will be the number of qubits.\n\n        Raises:\n            CircuitError: If A is not symmetric.\n        '
        adjacency_matrix = np.asarray(adjacency_matrix)
        if not np.allclose(adjacency_matrix, adjacency_matrix.transpose()):
            raise CircuitError('The adjacency matrix must be symmetric.')
        num_qubits = len(adjacency_matrix)
        circuit = QuantumCircuit(num_qubits, name='hlf: %s' % adjacency_matrix)
        circuit.h(range(num_qubits))
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                if adjacency_matrix[i][j]:
                    circuit.cz(i, j)
        for i in range(num_qubits):
            if adjacency_matrix[i][i]:
                circuit.s(i)
        circuit.h(range(num_qubits))
        super().__init__(*circuit.qregs, name=circuit.name)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)