"""Exact reciprocal rotation."""
from math import isclose
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.generalized_gates import UCRYGate

class ExactReciprocal(QuantumCircuit):
    """Exact reciprocal

    .. math::

        |x\\rangle |0\\rangle \\mapsto \\cos(1/x)|x\\rangle|0\\rangle + \\sin(1/x)|x\\rangle |1\\rangle
    """

    def __init__(self, num_state_qubits: int, scaling: float, neg_vals: bool=False, name: str='1/x') -> None:
        if False:
            while True:
                i = 10
        '\n        Args:\n            num_state_qubits: The number of qubits representing the value to invert.\n            scaling: Scaling factor :math:`s` of the reciprocal function, i.e. to compute\n                :math:`s / x`.\n            neg_vals: Whether :math:`x` might represent negative values. In this case the first\n                qubit is the sign, with :math:`|1\\rangle` for negative and :math:`|0\\rangle` for\n                positive.  For the negative case it is assumed that the remaining string represents\n                :math:`1 - x`. This is because :math:`e^{-2 \\pi i x} = e^{2 \\pi i (1 - x)}` for\n                :math:`x \\in [0,1)`.\n            name: The name of the object.\n\n        .. note::\n\n            It is assumed that the binary string :math:`x` represents a number < 1.\n        '
        qr_state = QuantumRegister(num_state_qubits, 'state')
        qr_flag = QuantumRegister(1, 'flag')
        circuit = QuantumCircuit(qr_state, qr_flag, name=name)
        angles = [0.0]
        nl = 2 ** (num_state_qubits - 1) if neg_vals else 2 ** num_state_qubits
        for i in range(1, nl):
            if isclose(scaling * nl / i, 1, abs_tol=1e-05):
                angles.append(np.pi)
            elif scaling * nl / i < 1:
                angles.append(2 * np.arcsin(scaling * nl / i))
            else:
                angles.append(0.0)
        circuit.compose(UCRYGate(angles), [qr_flag[0]] + qr_state[:len(qr_state) - neg_vals], inplace=True)
        if neg_vals:
            circuit.compose(UCRYGate([-theta for theta in angles]).control(), [qr_state[-1]] + [qr_flag[0]] + qr_state[:-1], inplace=True)
            angles_neg = [0.0]
            for i in range(1, nl):
                if isclose(scaling * -1 / (1 - i / nl), -1, abs_tol=1e-05):
                    angles_neg.append(-np.pi)
                elif np.abs(scaling * -1 / (1 - i / nl)) < 1:
                    angles_neg.append(2 * np.arcsin(scaling * -1 / (1 - i / nl)))
                else:
                    angles_neg.append(0.0)
            circuit.compose(UCRYGate(angles_neg).control(), [qr_state[-1]] + [qr_flag[0]] + qr_state[:-1], inplace=True)
        super().__init__(*circuit.qregs, name=name)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)