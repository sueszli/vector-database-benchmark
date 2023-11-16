"""Fourier checking circuit."""
from typing import List
import math
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .generalized_gates.diagonal import Diagonal

class FourierChecking(QuantumCircuit):
    """Fourier checking circuit.

    The circuit for the Fourier checking algorithm, introduced in [1],
    involves a layer of Hadamards, the function :math:`f`, another layer of
    Hadamards, the function :math:`g`, followed by a final layer of Hadamards.
    The functions :math:`f` and :math:`g` are classical functions realized
    as phase oracles (diagonal operators with {-1, 1} on the diagonal).

    The probability of observing the all-zeros string is :math:`p(f,g)`.
    The algorithm solves the promise Fourier checking problem,
    which decides if f is correlated with the Fourier transform
    of g, by testing if :math:`p(f,g) <= 0.01` or :math:`p(f,g) >= 0.05`,
    promised that one or the other of these is true.

    The functions :math:`f` and :math:`g` are currently implemented
    from their truth tables but could be represented concisely and
    implemented efficiently for special classes of functions.

    Fourier checking is a special case of :math:`k`-fold forrelation [2].

    **Reference:**

    [1] S. Aaronson, BQP and the Polynomial Hierarchy, 2009 (Section 3.2).
    `arXiv:0910.4698 <https://arxiv.org/abs/0910.4698>`_

    [2] S. Aaronson, A. Ambainis, Forrelation: a problem that
    optimally separates quantum from classical computing, 2014.
    `arXiv:1411.5729 <https://arxiv.org/abs/1411.5729>`_
    """

    def __init__(self, f: List[int], g: List[int]) -> None:
        if False:
            while True:
                i = 10
        'Create Fourier checking circuit.\n\n        Args:\n            f: truth table for f, length 2**n list of {1,-1}.\n            g: truth table for g, length 2**n list of {1,-1}.\n\n        Raises:\n            CircuitError: if the inputs f and g are not valid.\n\n        Reference Circuit:\n            .. plot::\n\n               from qiskit.circuit.library import FourierChecking\n               from qiskit.tools.jupyter.library import _generate_circuit_library_visualization\n               f = [1, -1, -1, -1]\n               g = [1, 1, -1, -1]\n               circuit = FourierChecking(f, g)\n               _generate_circuit_library_visualization(circuit)\n        '
        num_qubits = math.log2(len(f))
        if len(f) != len(g) or num_qubits == 0 or (not num_qubits.is_integer()):
            raise CircuitError('The functions f and g must be given as truth tables, each as a list of 2**n entries of {1, -1}.')
        circuit = QuantumCircuit(num_qubits, name=f'fc: {f}, {g}')
        circuit.h(circuit.qubits)
        circuit.compose(Diagonal(f), inplace=True)
        circuit.h(circuit.qubits)
        circuit.compose(Diagonal(g), inplace=True)
        circuit.h(circuit.qubits)
        super().__init__(*circuit.qregs, name=circuit.name)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)