"""Synthesize a single qubit gate to a discrete basis set."""
from __future__ import annotations
import numpy as np
from qiskit.circuit.gate import Gate
from .gate_sequence import GateSequence
from .commutator_decompose import commutator_decompose
from .generate_basis_approximations import generate_basic_approximations, _1q_gates, _1q_inverses

class SolovayKitaevDecomposition:
    """The Solovay Kitaev discrete decomposition algorithm.

    This class is called recursively by the transpiler pass, which is why it is separeted.
    See :class:`qiskit.transpiler.passes.SolovayKitaev` for more information.
    """

    def __init__(self, basic_approximations: str | dict[str, np.ndarray] | list[GateSequence] | None=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            basic_approximations: A specification of the basic SU(2) approximations in terms\n                of discrete gates. At each iteration this algorithm, the remaining error is\n                approximated with the closest sequence of gates in this set.\n                If a ``str``, this specifies a ``.npy`` filename from which to load the\n                approximation. If a ``dict``, then this contains\n                ``{gates: effective_SO3_matrix}`` pairs,\n                e.g. ``{"h t": np.array([[0, 0.7071, -0.7071], [0, -0.7071, -0.7071], [-1, 0, 0]]}``.\n                If a list, this contains the same information as the dict, but already converted to\n                :class:`.GateSequence` objects, which contain the SO(3) matrix and gates.\n        '
        if basic_approximations is None:
            basic_approximations = generate_basic_approximations(basis_gates=['h', 't', 'tdg'], depth=10)
        self.basic_approximations = self.load_basic_approximations(basic_approximations)

    def load_basic_approximations(self, data: list | str | dict) -> list[GateSequence]:
        if False:
            while True:
                i = 10
        'Load basic approximations.\n\n        Args:\n            data: If a string, specifies the path to the file from where to load the data.\n                If a dictionary, directly specifies the decompositions as ``{gates: matrix}``.\n                There ``gates`` are the names of the gates producing the SO(3) matrix ``matrix``,\n                e.g. ``{"h t": np.array([[0, 0.7071, -0.7071], [0, -0.7071, -0.7071], [-1, 0, 0]]}``.\n\n        Returns:\n            A list of basic approximations as type ``GateSequence``.\n\n        Raises:\n            ValueError: If the number of gate combinations and associated matrices does not match.\n        '
        if isinstance(data, list):
            return data
        if isinstance(data, str):
            data = np.load(data, allow_pickle=True)
        sequences = []
        for (gatestring, matrix) in data.items():
            sequence = GateSequence()
            sequence.gates = [_1q_gates[element] for element in gatestring.split()]
            sequence.product = np.asarray(matrix)
            sequences.append(sequence)
        return sequences

    def run(self, gate_matrix: np.ndarray, recursion_degree: int, return_dag: bool=False, check_input: bool=True) -> 'QuantumCircuit' | 'DAGCircuit':
        if False:
            for i in range(10):
                print('nop')
        'Run the algorithm.\n\n        Args:\n            gate_matrix: The 2x2 matrix representing the gate. This matrix has to be SU(2)\n                up to global phase.\n            recursion_degree: The recursion degree, called :math:`n` in the paper.\n            return_dag: If ``True`` return a :class:`.DAGCircuit`, else a :class:`.QuantumCircuit`.\n            check_input: If ``True`` check that the input matrix is valid for the decomposition.\n\n        Returns:\n            A one-qubit circuit approximating the ``gate_matrix`` in the specified discrete basis.\n        '
        z = 1 / np.sqrt(np.linalg.det(gate_matrix))
        gate_matrix_su2 = GateSequence.from_matrix(z * gate_matrix)
        global_phase = np.arctan2(np.imag(z), np.real(z))
        decomposition = self._recurse(gate_matrix_su2, recursion_degree, check_input=check_input)
        _remove_identities(decomposition)
        _remove_inverse_follows_gate(decomposition)
        if return_dag:
            out = decomposition.to_dag()
        else:
            out = decomposition.to_circuit()
        out.global_phase = decomposition.global_phase - global_phase
        return out

    def _recurse(self, sequence: GateSequence, n: int, check_input: bool=True) -> GateSequence:
        if False:
            return 10
        'Performs ``n`` iterations of the Solovay-Kitaev algorithm on ``sequence``.\n\n        Args:\n            sequence: ``GateSequence`` to which the Solovay-Kitaev algorithm is applied.\n            n: The number of iterations that the algorithm needs to run.\n            check_input: If ``True`` check that the input matrix represented by ``GateSequence``\n                is valid for the decomposition.\n\n        Returns:\n            GateSequence that approximates ``sequence``.\n\n        Raises:\n            ValueError: If the matrix in ``GateSequence`` does not represent an SO(3)-matrix.\n        '
        if sequence.product.shape != (3, 3):
            raise ValueError('Shape of U must be (3, 3) but is', sequence.shape)
        if n == 0:
            return self.find_basic_approximation(sequence)
        u_n1 = self._recurse(sequence, n - 1, check_input=check_input)
        (v_n, w_n) = commutator_decompose(sequence.dot(u_n1.adjoint()).product, check_input=check_input)
        v_n1 = self._recurse(v_n, n - 1, check_input=check_input)
        w_n1 = self._recurse(w_n, n - 1, check_input=check_input)
        return v_n1.dot(w_n1).dot(v_n1.adjoint()).dot(w_n1.adjoint()).dot(u_n1)

    def find_basic_approximation(self, sequence: GateSequence) -> Gate:
        if False:
            return 10
        'Finds gate in ``self._basic_approximations`` that best represents ``sequence``.\n\n        Args:\n            sequence: The gate to find the approximation to.\n\n        Returns:\n            Gate in basic approximations that is closest to ``sequence``.\n        '

        def key(x):
            if False:
                print('Hello World!')
            return np.linalg.norm(np.subtract(x.product, sequence.product))
        best = min(self.basic_approximations, key=key)
        return best

def _remove_inverse_follows_gate(sequence):
    if False:
        while True:
            i = 10
    index = 0
    while index < len(sequence.gates) - 1:
        curr_gate = sequence.gates[index]
        next_gate = sequence.gates[index + 1]
        if curr_gate.name in _1q_inverses.keys():
            remove = _1q_inverses[curr_gate.name] == next_gate.name
        else:
            remove = curr_gate.inverse() == next_gate
        if remove:
            sequence.remove_cancelling_pair([index, index + 1])
            if index > 0:
                index -= 1
        else:
            index += 1

def _remove_identities(sequence):
    if False:
        return 10
    index = 0
    while index < len(sequence.gates):
        if sequence.gates[index].name == 'id':
            sequence.gates.pop(index)
        else:
            index += 1