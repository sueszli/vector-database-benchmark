"""Compute the weighted sum of qubit states."""
from typing import List, Optional
import numpy as np
from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit
from ..blueprintcircuit import BlueprintCircuit

class WeightedAdder(BlueprintCircuit):
    """A circuit to compute the weighted sum of qubit registers.

    Given :math:`n` qubit basis states :math:`q_0, \\ldots, q_{n-1} \\in \\{0, 1\\}` and non-negative
    integer weights :math:`\\lambda_0, \\ldots, \\lambda_{n-1}`, this circuit performs the operation

    .. math::

        |q_0 \\ldots q_{n-1}\\rangle |0\\rangle_s
        \\mapsto |q_0 \\ldots q_{n-1}\\rangle |\\sum_{j=0}^{n-1} \\lambda_j q_j\\rangle_s

    where :math:`s` is the number of sum qubits required.
    This can be computed as

    .. math::

        s = 1 + \\left\\lfloor \\log_2\\left( \\sum_{j=0}^{n-1} \\lambda_j \\right) \\right\\rfloor

    or :math:`s = 1` if the sum of the weights is 0 (then the expression in the logarithm is
    invalid).

    For qubits in a circuit diagram, the first weight applies to the upper-most qubit.
    For an example where the state of 4 qubits is added into a sum register, the circuit can
    be schematically drawn as

    .. parsed-literal::

                   ┌────────┐
          state_0: ┤0       ├ | state_0 * weights[0]
                   │        │ |
          state_1: ┤1       ├ | + state_1 * weights[1]
                   │        │ |
          state_2: ┤2       ├ | + state_2 * weights[2]
                   │        │ |
          state_3: ┤3       ├ | + state_3 * weights[3]
                   │        │
            sum_0: ┤4       ├ |
                   │  Adder │ |
            sum_1: ┤5       ├ | = sum_0 * 2^0 + sum_1 * 2^1 + sum_2 * 2^2
                   │        │ |
            sum_2: ┤6       ├ |
                   │        │
          carry_0: ┤7       ├
                   │        │
          carry_1: ┤8       ├
                   │        │
        control_0: ┤9       ├
                   └────────┘
    """

    def __init__(self, num_state_qubits: Optional[int]=None, weights: Optional[List[int]]=None, name: str='adder') -> None:
        if False:
            while True:
                i = 10
        'Computes the weighted sum controlled by state qubits.\n\n        Args:\n            num_state_qubits: The number of state qubits.\n            weights: List of weights, one for each state qubit. If none are provided they\n                default to 1 for every qubit.\n            name: The name of the circuit.\n        '
        super().__init__(name=name)
        self._weights = None
        self._num_state_qubits = None
        self.weights = weights
        self.num_state_qubits = num_state_qubits

    @property
    def num_sum_qubits(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'The number of sum qubits in the circuit.\n\n        Returns:\n            The number of qubits needed to represent the weighted sum of the qubits.\n        '
        if sum(self.weights) > 0:
            return int(np.floor(np.log2(sum(self.weights))) + 1)
        return 1

    @property
    def weights(self) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        'The weights for the qubit states.\n\n        Returns:\n            The weight for the qubit states.\n        '
        if self._weights:
            return self._weights
        if self.num_state_qubits:
            return [1] * self.num_state_qubits
        return None

    @weights.setter
    def weights(self, weights: List[int]) -> None:
        if False:
            print('Hello World!')
        'Set the weights for summing the qubit states.\n\n        Args:\n            weights: The new weights.\n\n        Raises:\n            ValueError: If not all weights are close to an integer.\n        '
        if weights:
            for (i, weight) in enumerate(weights):
                if not np.isclose(weight, np.round(weight)):
                    raise ValueError('Non-integer weights are not supported!')
                weights[i] = np.round(weight)
        self._invalidate()
        self._weights = weights
        self._reset_registers()

    @property
    def num_state_qubits(self) -> int:
        if False:
            while True:
                i = 10
        'The number of qubits to be summed.\n\n        Returns:\n            The number of state qubits.\n        '
        return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, num_state_qubits: int) -> None:
        if False:
            return 10
        'Set the number of state qubits.\n\n        Args:\n            num_state_qubits: The new number of state qubits.\n        '
        if self._num_state_qubits is None or num_state_qubits != self._num_state_qubits:
            self._invalidate()
            self._num_state_qubits = num_state_qubits
            self._reset_registers()

    def _reset_registers(self):
        if False:
            return 10
        'Reset the registers.'
        self.qregs = []
        if self.num_state_qubits:
            qr_state = QuantumRegister(self.num_state_qubits, name='state')
            qr_sum = QuantumRegister(self.num_sum_qubits, name='sum')
            self.qregs = [qr_state, qr_sum]
            if self.num_carry_qubits > 0:
                qr_carry = AncillaRegister(self.num_carry_qubits, name='carry')
                self.add_register(qr_carry)
            if self.num_control_qubits > 0:
                qr_control = AncillaRegister(self.num_control_qubits, name='control')
                self.add_register(qr_control)

    @property
    def num_carry_qubits(self) -> int:
        if False:
            while True:
                i = 10
        'The number of carry qubits required to compute the sum.\n\n        Note that this is not necessarily equal to the number of ancilla qubits, these can\n        be queried using ``num_ancilla_qubits``.\n\n        Returns:\n            The number of carry qubits required to compute the sum.\n        '
        return self.num_sum_qubits - 1

    @property
    def num_control_qubits(self) -> int:
        if False:
            return 10
        'The number of additional control qubits required.\n\n        Note that the total number of ancilla qubits can be obtained by calling the\n        method ``num_ancilla_qubits``.\n\n        Returns:\n            The number of additional control qubits required (0 or 1).\n        '
        return int(self.num_sum_qubits > 2)

    def _check_configuration(self, raise_on_failure=True):
        if False:
            print('Hello World!')
        'Check if the current configuration is valid.'
        valid = True
        if self._num_state_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('The number of state qubits has not been set.')
        if self._num_state_qubits != len(self.weights):
            valid = False
            if raise_on_failure:
                raise ValueError('Mismatching number of state qubits and weights.')
        return valid

    def _build(self):
        if False:
            for i in range(10):
                print('nop')
        'If not already built, build the circuit.'
        if self._is_built:
            return
        super()._build()
        num_result_qubits = self.num_state_qubits + self.num_sum_qubits
        circuit = QuantumCircuit(*self.qregs)
        qr_state = circuit.qubits[:self.num_state_qubits]
        qr_sum = circuit.qubits[self.num_state_qubits:num_result_qubits]
        qr_carry = circuit.qubits[num_result_qubits:num_result_qubits + self.num_carry_qubits]
        qr_control = circuit.qubits[num_result_qubits + self.num_carry_qubits:]
        for (i, weight) in enumerate(self.weights):
            if np.isclose(weight, 0):
                continue
            q_state = qr_state[i]
            weight_binary = f'{int(weight):b}'.rjust(self.num_sum_qubits, '0')[::-1]
            for (j, bit) in enumerate(weight_binary):
                if bit == '1':
                    if self.num_sum_qubits == 1:
                        circuit.cx(q_state, qr_sum[j])
                    elif j == 0:
                        circuit.ccx(q_state, qr_sum[j], qr_carry[j])
                        circuit.cx(q_state, qr_sum[j])
                    elif j == self.num_sum_qubits - 1:
                        circuit.cx(q_state, qr_sum[j])
                        circuit.ccx(q_state, qr_carry[j - 1], qr_sum[j])
                    else:
                        circuit.x(qr_sum[j])
                        circuit.x(qr_carry[j - 1])
                        circuit.mcx([q_state, qr_sum[j], qr_carry[j - 1]], qr_carry[j], qr_control, mode='v-chain')
                        circuit.cx(q_state, qr_carry[j])
                        circuit.x(qr_sum[j])
                        circuit.x(qr_carry[j - 1])
                        circuit.cx(q_state, qr_sum[j])
                        circuit.ccx(q_state, qr_carry[j - 1], qr_sum[j])
                elif self.num_sum_qubits == 1:
                    pass
                elif j == 0:
                    pass
                elif j == self.num_sum_qubits - 1:
                    circuit.ccx(q_state, qr_carry[j - 1], qr_sum[j])
                else:
                    circuit.mcx([q_state, qr_sum[j], qr_carry[j - 1]], qr_carry[j], qr_control, mode='v-chain')
                    circuit.ccx(q_state, qr_carry[j - 1], qr_sum[j])
            for j in reversed(range(len(weight_binary))):
                bit = weight_binary[j]
                if bit == '1':
                    if self.num_sum_qubits == 1:
                        pass
                    elif j == 0:
                        circuit.x(qr_sum[j])
                        circuit.ccx(q_state, qr_sum[j], qr_carry[j])
                        circuit.x(qr_sum[j])
                    elif j == self.num_sum_qubits - 1:
                        pass
                    else:
                        circuit.x(qr_carry[j - 1])
                        circuit.mcx([q_state, qr_sum[j], qr_carry[j - 1]], qr_carry[j], qr_control, mode='v-chain')
                        circuit.cx(q_state, qr_carry[j])
                        circuit.x(qr_carry[j - 1])
                elif self.num_sum_qubits == 1:
                    pass
                elif j == 0:
                    pass
                elif j == self.num_sum_qubits - 1:
                    pass
                else:
                    circuit.x(qr_sum[j])
                    circuit.mcx([q_state, qr_sum[j], qr_carry[j - 1]], qr_carry[j], qr_control, mode='v-chain')
                    circuit.x(qr_sum[j])
        self.append(circuit.to_gate(), self.qubits)