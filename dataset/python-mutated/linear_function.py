"""Linear Function."""
from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.synthesis.linear import check_invertible_binary_matrix
from qiskit.circuit.library.generalized_gates.permutation import PermutationGate

class LinearFunction(Gate):
    """A linear reversible circuit on n qubits.

    Internally, a linear function acting on n qubits is represented
    as a n x n matrix of 0s and 1s in numpy array format.

    A linear function can be synthesized into CX and SWAP gates using the Patel–Markov–Hayes
    algorithm, as implemented in :func:`~qiskit.transpiler.synthesis.cnot_synth`
    based on reference [1].

    For efficiency, the internal n x n matrix is stored in the format expected
    by cnot_synth, which is the big-endian (and not the little-endian) bit-ordering convention.

    **Example:** the circuit

    .. parsed-literal::

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ X ├
             └───┘
        q_2: ─────

    is represented by a 3x3 linear matrix

    .. math::

            \\begin{pmatrix}
                1 & 0 & 0 \\\\
                1 & 1 & 0 \\\\
                0 & 0 & 1
            \\end{pmatrix}


    **References:**

    [1] Ketan N. Patel, Igor L. Markov, and John P. Hayes,
    Optimal synthesis of linear reversible circuits,
    Quantum Inf. Comput. 8(3) (2008).
    `Online at umich.edu. <https://web.eecs.umich.edu/~imarkov/pubs/jour/qic08-cnot.pdf>`_
    """

    def __init__(self, linear, validate_input=False):
        if False:
            print('Hello World!')
        'Create a new linear function.\n\n        Args:\n            linear (list[list] or ndarray[bool] or QuantumCircuit or LinearFunction\n                or PermutationGate or Clifford): data from which a linear function\n                can be constructed. It can be either a nxn matrix (describing the\n                linear transformation), a permutation (which is a special case of\n                a linear function), another linear function, a clifford (when it\n                corresponds to a linear function), or a quantum circuit composed of\n                linear gates (CX and SWAP) and other objects described above, including\n                nested subcircuits.\n\n            validate_input: if True, performs more expensive input validation checks,\n                such as checking that a given n x n matrix is invertible.\n\n        Raises:\n            CircuitError: if the input is invalid:\n                either the input matrix is not square or not invertible,\n                or the input quantum circuit contains non-linear objects\n                (for example, a Hadamard gate, or a Clifford that does\n                not correspond to a linear function).\n        '
        from qiskit.quantum_info import Clifford
        original_circuit = None
        if isinstance(linear, (list, np.ndarray)):
            try:
                linear = np.array(linear, dtype=bool, copy=True)
            except ValueError:
                raise CircuitError('A linear function must be represented by a square matrix.') from None
            if len(linear.shape) != 2 or linear.shape[0] != linear.shape[1]:
                raise CircuitError('A linear function must be represented by a square matrix.')
            if validate_input:
                if not check_invertible_binary_matrix(linear):
                    raise CircuitError('A linear function must be represented by an invertible matrix.')
        elif isinstance(linear, QuantumCircuit):
            original_circuit = linear
            linear = LinearFunction._circuit_to_mat(linear)
        elif isinstance(linear, LinearFunction):
            linear = linear.linear.copy()
        elif isinstance(linear, PermutationGate):
            linear = LinearFunction._permutation_to_mat(linear)
        elif isinstance(linear, Clifford):
            linear = LinearFunction._clifford_to_mat(linear)
        else:
            raise CircuitError('A linear function cannot be successfully constructed.')
        super().__init__(name='linear_function', num_qubits=len(linear), params=[linear, original_circuit])

    @staticmethod
    def _circuit_to_mat(qc: QuantumCircuit):
        if False:
            return 10
        'This creates a nxn matrix corresponding to the given quantum circuit.'
        nq = qc.num_qubits
        mat = np.eye(nq, nq, dtype=bool)
        for instruction in qc.data:
            if instruction.operation.name in ('barrier', 'delay'):
                continue
            if instruction.operation.name == 'cx':
                cb = qc.find_bit(instruction.qubits[0]).index
                tb = qc.find_bit(instruction.qubits[1]).index
                mat[tb, :] = mat[tb, :] ^ mat[cb, :]
                continue
            if instruction.operation.name == 'swap':
                cb = qc.find_bit(instruction.qubits[0]).index
                tb = qc.find_bit(instruction.qubits[1]).index
                mat[[cb, tb]] = mat[[tb, cb]]
                continue
            if getattr(instruction.operation, 'definition', None) is not None:
                other = LinearFunction(instruction.operation.definition)
            else:
                other = LinearFunction(instruction.operation)
            positions = [qc.find_bit(q).index for q in instruction.qubits]
            other = other.extend_with_identity(len(mat), positions)
            mat = np.dot(other.linear.astype(int), mat.astype(int)) % 2
            mat = mat.astype(bool)
        return mat

    @staticmethod
    def _clifford_to_mat(cliff):
        if False:
            print('Hello World!')
        'This creates a nxn matrix corresponding to the given Clifford, when Clifford\n        can be converted to a linear function. This is possible when the clifford has\n        tableau of the form [[A, B], [C, D]], with B = C = 0 and D = A^{-1}^t, and zero\n        phase vector. In this case, the required matrix is A^t.\n        Raises an error otherwise.\n        '
        if cliff.phase.any() or cliff.destab_z.any() or cliff.stab_x.any():
            raise CircuitError('The given clifford does not correspond to a linear function.')
        return np.transpose(cliff.destab_x)

    @staticmethod
    def _permutation_to_mat(perm):
        if False:
            while True:
                i = 10
        'This creates a nxn matrix from a given permutation gate.'
        nq = len(perm.pattern)
        mat = np.zeros((nq, nq), dtype=bool)
        for (i, j) in enumerate(perm.pattern):
            mat[i, j] = True
        return mat

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        'Check if two linear functions represent the same matrix.'
        if not isinstance(other, LinearFunction):
            return False
        return (self.linear == other.linear).all()

    def validate_parameter(self, parameter):
        if False:
            print('Hello World!')
        'Parameter validation'
        return parameter

    def _define(self):
        if False:
            for i in range(10):
                print('nop')
        'Populates self.definition with a decomposition of this gate.'
        self.definition = self.synthesize()

    def synthesize(self):
        if False:
            while True:
                i = 10
        'Synthesizes the linear function into a quantum circuit.\n\n        Returns:\n            QuantumCircuit: A circuit implementing the evolution.\n        '
        from qiskit.synthesis.linear import synth_cnot_count_full_pmh
        return synth_cnot_count_full_pmh(self.linear)

    @property
    def linear(self):
        if False:
            i = 10
            return i + 15
        'Returns the n x n matrix representing this linear function.'
        return self.params[0]

    @property
    def original_circuit(self):
        if False:
            while True:
                i = 10
        'Returns the original circuit used to construct this linear function\n        (including None, when the linear function is not constructed from a circuit).\n        '
        return self.params[1]

    def is_permutation(self) -> bool:
        if False:
            while True:
                i = 10
        'Returns whether this linear function is a permutation,\n        that is whether every row and every column of the n x n matrix\n        has exactly one 1.\n        '
        linear = self.linear
        perm = np.all(np.sum(linear, axis=0) == 1) and np.all(np.sum(linear, axis=1) == 1)
        return perm

    def permutation_pattern(self):
        if False:
            i = 10
            return i + 15
        'This method first checks if a linear function is a permutation and raises a\n        `qiskit.circuit.exceptions.CircuitError` if not. In the case that this linear function\n        is a permutation, returns the permutation pattern.\n        '
        if not self.is_permutation():
            raise CircuitError('The linear function is not a permutation')
        linear = self.linear
        locs = np.where(linear == 1)
        return locs[1]

    def extend_with_identity(self, num_qubits: int, positions: list[int]) -> LinearFunction:
        if False:
            return 10
        "Extend linear function to a linear function over nq qubits,\n        with identities on other subsystems.\n\n        Args:\n            num_qubits: number of qubits of the extended function.\n\n            positions: describes the positions of original qubits in the extended\n                function's qubits.\n\n        Returns:\n            LinearFunction: extended linear function.\n        "
        extended_mat = np.eye(num_qubits, dtype=bool)
        for (i, pos) in enumerate(positions):
            extended_mat[positions, pos] = self.linear[:, i]
        return LinearFunction(extended_mat)

    def mat_str(self):
        if False:
            print('Hello World!')
        'Return string representation of the linear function\n        viewed as a matrix with 0/1 entries.\n        '
        return str(self.linear.astype(int))

    def function_str(self):
        if False:
            i = 10
            return i + 15
        'Return string representation of the linear function\n        viewed as a linear transformation.\n        '
        out = '('
        mat = self.linear
        for row in range(self.num_qubits):
            first_entry = True
            for col in range(self.num_qubits):
                if mat[row, col]:
                    if not first_entry:
                        out += ' + '
                    out += 'x_' + str(col)
                    first_entry = False
            if row != self.num_qubits - 1:
                out += ', '
        out += ')\n'
        return out