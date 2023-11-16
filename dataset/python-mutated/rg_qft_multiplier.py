"""Compute the product of two qubit registers using QFT."""
from typing import Optional
import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library.standard_gates import PhaseGate
from qiskit.circuit.library.basis_change import QFT
from .multiplier import Multiplier

class RGQFTMultiplier(Multiplier):
    """A QFT multiplication circuit to store product of two input registers out-of-place.

    Multiplication in this circuit is implemented using the procedure of Fig. 3 in [1], where
    weighted sum rotations are implemented as given in Fig. 5 in [1]. QFT is used on the output
    register and is followed by rotations controlled by input registers. The rotations
    transform the state into the product of two input registers in QFT base, which is
    reverted from QFT base using inverse QFT.
    As an example, a circuit that performs a modular QFT multiplication on two 2-qubit
    sized input registers with an output register of 2 qubits, is as follows:

    .. parsed-literal::

          a_0: ────────────────────────────────────────■───────■──────■──────■────────────────
                                                       │       │      │      │
          a_1: ─────────■───────■───────■───────■──────┼───────┼──────┼──────┼────────────────
                        │       │       │       │      │       │      │      │
          b_0: ─────────┼───────┼───────■───────■──────┼───────┼──────■──────■────────────────
                        │       │       │       │      │       │      │      │
          b_1: ─────────■───────■───────┼───────┼──────■───────■──────┼──────┼────────────────
               ┌──────┐ │P(4π)  │       │P(2π)  │      │P(2π)  │      │P(π)  │       ┌───────┐
        out_0: ┤0     ├─■───────┼───────■───────┼──────■───────┼──────■──────┼───────┤0      ├
               │  qft │         │P(2π)          │P(π)          │P(π)         │P(π/2) │  iqft │
        out_1: ┤1     ├─────────■───────────────■──────────────■─────────────■───────┤1      ├
               └──────┘                                                              └───────┘

    **References:**

    [1] Ruiz-Perez et al., Quantum arithmetic with the Quantum Fourier Transform, 2017.
    `arXiv:1411.5949 <https://arxiv.org/pdf/1411.5949.pdf>`_

    """

    def __init__(self, num_state_qubits: int, num_result_qubits: Optional[int]=None, name: str='RGQFTMultiplier') -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            num_state_qubits: The number of qubits in either input register for\n                state :math:`|a\\rangle` or :math:`|b\\rangle`. The two input\n                registers must have the same number of qubits.\n            num_result_qubits: The number of result qubits to limit the output to.\n                If number of result qubits is :math:`n`, multiplication modulo :math:`2^n` is performed\n                to limit the output to the specified number of qubits. Default\n                value is ``2 * num_state_qubits`` to represent any possible\n                result from the multiplication of the two inputs.\n            name: The name of the circuit object.\n\n        '
        super().__init__(num_state_qubits, num_result_qubits, name=name)
        qr_a = QuantumRegister(num_state_qubits, name='a')
        qr_b = QuantumRegister(num_state_qubits, name='b')
        qr_out = QuantumRegister(self.num_result_qubits, name='out')
        self.add_register(qr_a, qr_b, qr_out)
        circuit = QuantumCircuit(*self.qregs, name=name)
        circuit.append(QFT(self.num_result_qubits, do_swaps=False).to_gate(), qr_out[:])
        for j in range(1, num_state_qubits + 1):
            for i in range(1, num_state_qubits + 1):
                for k in range(1, self.num_result_qubits + 1):
                    lam = 2 * np.pi / 2 ** (i + j + k - 2 * num_state_qubits)
                    circuit.append(PhaseGate(lam).control(2), [qr_a[num_state_qubits - j], qr_b[num_state_qubits - i], qr_out[k - 1]])
        circuit.append(QFT(self.num_result_qubits, do_swaps=False).inverse().to_gate(), qr_out[:])
        self.append(circuit.to_gate(), self.qubits)