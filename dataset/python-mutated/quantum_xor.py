"""XOR circuit."""
from typing import Optional
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError

class XOR(QuantumCircuit):
    """An n_qubit circuit for bitwise xor-ing the input with some integer ``amount``.

    The ``amount`` is xor-ed in bitstring form with the input.

    This circuit can also represent addition by ``amount`` over the finite field GF(2).
    """

    def __init__(self, num_qubits: int, amount: Optional[int]=None, seed: Optional[int]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Return a circuit implementing bitwise xor.\n\n        Args:\n            num_qubits: the width of circuit.\n            amount: the xor amount in decimal form.\n            seed: random seed in case a random xor is requested.\n\n        Raises:\n            CircuitError: if the xor bitstring exceeds available qubits.\n\n        Reference Circuit:\n            .. plot::\n\n               from qiskit.circuit.library import XOR\n               from qiskit.tools.jupyter.library import _generate_circuit_library_visualization\n               circuit = XOR(5, seed=42)\n               _generate_circuit_library_visualization(circuit)\n        '
        circuit = QuantumCircuit(num_qubits, name='xor')
        if amount is not None:
            if len(bin(amount)[2:]) > num_qubits:
                raise CircuitError("Bits in 'amount' exceed circuit width")
        else:
            rng = np.random.default_rng(seed)
            amount = rng.integers(0, 2 ** num_qubits)
        for i in range(num_qubits):
            bit = amount & 1
            amount = amount >> 1
            if bit == 1:
                circuit.x(i)
        super().__init__(*circuit.qregs, name='xor')
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)