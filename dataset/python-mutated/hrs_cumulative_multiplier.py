"""Compute the product of two qubit registers using classical multiplication approach."""
from typing import Optional
from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit
from .multiplier import Multiplier

class HRSCumulativeMultiplier(Multiplier):
    """A multiplication circuit to store product of two input registers out-of-place.

    Circuit uses the approach from [1]. As an example, a multiplier circuit that
    performs a non-modular multiplication on two 3-qubit sized registers with
    the default adder is as follows (where ``Adder`` denotes the
    ``CDKMRippleCarryAdder``):

    .. parsed-literal::

          a_0: ────■─────────────────────────
                   │
          a_1: ────┼─────────■───────────────
                   │         │
          a_2: ────┼─────────┼─────────■─────
               ┌───┴────┐┌───┴────┐┌───┴────┐
          b_0: ┤0       ├┤0       ├┤0       ├
               │        ││        ││        │
          b_1: ┤1       ├┤1       ├┤1       ├
               │        ││        ││        │
          b_2: ┤2       ├┤2       ├┤2       ├
               │        ││        ││        │
        out_0: ┤3       ├┤        ├┤        ├
               │        ││        ││        │
        out_1: ┤4       ├┤3       ├┤        ├
               │  Adder ││  Adder ││  Adder │
        out_2: ┤5       ├┤4       ├┤3       ├
               │        ││        ││        │
        out_3: ┤6       ├┤5       ├┤4       ├
               │        ││        ││        │
        out_4: ┤        ├┤6       ├┤5       ├
               │        ││        ││        │
        out_5: ┤        ├┤        ├┤6       ├
               │        ││        ││        │
        aux_0: ┤7       ├┤7       ├┤7       ├
               └────────┘└────────┘└────────┘

    Multiplication in this circuit is implemented in a classical approach by performing
    a series of shifted additions using one of the input registers while the qubits
    from the other input register act as control qubits for the adders.

    **References:**

    [1] Häner et al., Optimizing Quantum Circuits for Arithmetic, 2018.
    `arXiv:1805.12445 <https://arxiv.org/pdf/1805.12445.pdf>`_

    """

    def __init__(self, num_state_qubits: int, num_result_qubits: Optional[int]=None, adder: Optional[QuantumCircuit]=None, name: str='HRSCumulativeMultiplier') -> None:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            num_state_qubits: The number of qubits in either input register for\n                state :math:`|a\\rangle` or :math:`|b\\rangle`. The two input\n                registers must have the same number of qubits.\n            num_result_qubits: The number of result qubits to limit the output to.\n                If number of result qubits is :math:`n`, multiplication modulo :math:`2^n` is performed\n                to limit the output to the specified number of qubits. Default\n                value is ``2 * num_state_qubits`` to represent any possible\n                result from the multiplication of the two inputs.\n            adder: Half adder circuit to be used for performing multiplication. The\n                CDKMRippleCarryAdder is used as default if no adder is provided.\n            name: The name of the circuit object.\n        Raises:\n            NotImplementedError: If ``num_result_qubits`` is not default and a custom adder is provided.\n        '
        super().__init__(num_state_qubits, num_result_qubits, name=name)
        if self.num_result_qubits != 2 * num_state_qubits and adder is not None:
            raise NotImplementedError('Only default adder is supported for modular multiplication.')
        qr_a = QuantumRegister(num_state_qubits, name='a')
        qr_b = QuantumRegister(num_state_qubits, name='b')
        qr_out = QuantumRegister(self.num_result_qubits, name='out')
        self.add_register(qr_a, qr_b, qr_out)
        if adder is None:
            from qiskit.circuit.library.arithmetic.adders import CDKMRippleCarryAdder
            adder = CDKMRippleCarryAdder(num_state_qubits, kind='half')
        num_helper_qubits = adder.num_ancillas
        if num_helper_qubits > 0:
            qr_h = AncillaRegister(num_helper_qubits, name='helper')
            self.add_register(qr_h)
        circuit = QuantumCircuit(*self.qregs, name=name)
        for i in range(num_state_qubits):
            excess_qubits = max(0, num_state_qubits + i + 1 - self.num_result_qubits)
            if excess_qubits == 0:
                num_adder_qubits = num_state_qubits
                adder_for_current_step = adder
            else:
                num_adder_qubits = num_state_qubits - excess_qubits + 1
                adder_for_current_step = CDKMRippleCarryAdder(num_adder_qubits, kind='fixed')
            controlled_adder = adder_for_current_step.to_gate().control(1)
            qr_list = [qr_a[i]] + qr_b[:num_adder_qubits] + qr_out[i:num_state_qubits + i + 1 - excess_qubits]
            if num_helper_qubits > 0:
                qr_list.extend(qr_h[:])
            circuit.append(controlled_adder, qr_list)
        self.append(circuit.to_gate(), self.qubits)