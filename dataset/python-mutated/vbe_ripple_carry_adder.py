"""Compute the sum of two qubit registers using Classical Addition."""
from __future__ import annotations
from qiskit.circuit.bit import Bit
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from .adder import Adder

class VBERippleCarryAdder(Adder):
    """The VBE ripple carry adder [1].

    This circuit performs inplace addition of two equally-sized quantum registers.
    As an example, a classical adder circuit that performs full addition (i.e. including
    a carry-in bit) on two 2-qubit sized registers is as follows:

    .. parsed-literal::

                  ┌────────┐                       ┌───────────┐┌──────┐
           cin_0: ┤0       ├───────────────────────┤0          ├┤0     ├
                  │        │                       │           ││      │
             a_0: ┤1       ├───────────────────────┤1          ├┤1     ├
                  │        │┌────────┐     ┌──────┐│           ││  Sum │
             a_1: ┤        ├┤1       ├──■──┤1     ├┤           ├┤      ├
                  │        ││        │  │  │      ││           ││      │
             b_0: ┤2 Carry ├┤        ├──┼──┤      ├┤2 Carry_dg ├┤2     ├
                  │        ││        │┌─┴─┐│      ││           │└──────┘
             b_1: ┤        ├┤2 Carry ├┤ X ├┤2 Sum ├┤           ├────────
                  │        ││        │└───┘│      ││           │
          cout_0: ┤        ├┤3       ├─────┤      ├┤           ├────────
                  │        ││        │     │      ││           │
        helper_0: ┤3       ├┤0       ├─────┤0     ├┤3          ├────────
                  └────────┘└────────┘     └──────┘└───────────┘


    Here *Carry* and *Sum* gates correspond to the gates introduced in [1].
    *Carry_dg* correspond to the inverse of the *Carry* gate. Note that
    in this implementation the input register qubits are ordered as all qubits from
    the first input register, followed by all qubits from the second input register.
    This is different ordering as compared to Figure 2 in [1], which leads to a different
    drawing of the circuit.

    **References:**

    [1] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
    `arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>`_

    """

    def __init__(self, num_state_qubits: int, kind: str='full', name: str='VBERippleCarryAdder') -> None:
        if False:
            while True:
                i = 10
        "\n        Args:\n            num_state_qubits: The size of the register.\n            kind: The kind of adder, can be ``'full'`` for a full adder, ``'half'`` for a half\n                adder, or ``'fixed'`` for a fixed-sized adder. A full adder includes both carry-in\n                and carry-out, a half only carry-out, and a fixed-sized adder neither carry-in\n                nor carry-out.\n            name: The name of the circuit.\n\n        Raises:\n            ValueError: If ``num_state_qubits`` is lower than 1.\n        "
        if num_state_qubits < 1:
            raise ValueError('The number of qubits must be at least 1.')
        super().__init__(num_state_qubits, name=name)
        registers: list[QuantumRegister | list[Bit]] = []
        if kind == 'full':
            qr_cin = QuantumRegister(1, name='cin')
            registers.append(qr_cin)
        else:
            qr_cin = QuantumRegister(0)
        qr_a = QuantumRegister(num_state_qubits, name='a')
        qr_b = QuantumRegister(num_state_qubits, name='b')
        registers += [qr_a, qr_b]
        if kind in ['half', 'full']:
            qr_cout = QuantumRegister(1, name='cout')
            registers.append(qr_cout)
        else:
            qr_cout = QuantumRegister(0)
        self.add_register(*registers)
        if num_state_qubits > 1:
            qr_help = AncillaRegister(num_state_qubits - 1, name='helper')
            self.add_register(qr_help)
        else:
            qr_help = AncillaRegister(0)
        carries = qr_cin[:] + qr_help[:] + qr_cout[:]
        qc_carry = QuantumCircuit(4, name='Carry')
        qc_carry.ccx(1, 2, 3)
        qc_carry.cx(1, 2)
        qc_carry.ccx(0, 2, 3)
        carry_gate = qc_carry.to_gate()
        carry_gate_dg = carry_gate.inverse()
        qc_sum = QuantumCircuit(3, name='Sum')
        qc_sum.cx(1, 2)
        qc_sum.cx(0, 2)
        sum_gate = qc_sum.to_gate()
        circuit = QuantumCircuit(*self.qregs, name=name)
        i = 0
        if kind == 'half':
            i += 1
            circuit.ccx(qr_a[0], qr_b[0], carries[0])
        elif kind == 'fixed':
            i += 1
            if num_state_qubits == 1:
                circuit.cx(qr_a[0], qr_b[0])
            else:
                circuit.ccx(qr_a[0], qr_b[0], carries[0])
        for (inp, out) in zip(carries[:-1], carries[1:]):
            circuit.append(carry_gate, [inp, qr_a[i], qr_b[i], out])
            i += 1
        if kind in ['full', 'half']:
            circuit.cx(qr_a[-1], qr_b[-1])
        if len(carries) > 1:
            circuit.append(sum_gate, [carries[-2], qr_a[-1], qr_b[-1]])
        i -= 2
        for (j, (inp, out)) in enumerate(zip(reversed(carries[:-1]), reversed(carries[1:]))):
            if j == 0:
                if kind == 'fixed':
                    i += 1
                else:
                    continue
            circuit.append(carry_gate_dg, [inp, qr_a[i], qr_b[i], out])
            circuit.append(sum_gate, [inp, qr_a[i], qr_b[i]])
            i -= 1
        if kind in ['half', 'fixed'] and num_state_qubits > 1:
            circuit.ccx(qr_a[0], qr_b[0], carries[0])
            circuit.cx(qr_a[0], qr_b[0])
        self.append(circuit.to_gate(), self.qubits)