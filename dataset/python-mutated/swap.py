"""Swap gate."""
from typing import Optional, Union
import numpy
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
_SWAP_ARRAY = numpy.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

@with_gate_array(_SWAP_ARRAY)
class SwapGate(SingletonGate):
    """The SWAP gate.

    This is a symmetric and Clifford gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.swap` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ─X─
              │
        q_1: ─X─

    **Matrix Representation:**

    .. math::

        SWAP =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & 0 & 1 & 0 \\\\
                0 & 1 & 0 & 0 \\\\
                0 & 0 & 0 & 1
            \\end{pmatrix}

    The gate is equivalent to a state swap and is a classical logic gate.

    .. math::

        |a, b\\rangle \\rightarrow |b, a\\rangle
    """

    def __init__(self, label: Optional[str]=None, *, duration=None, unit='dt'):
        if False:
            while True:
                i = 10
        'Create new SWAP gate.'
        super().__init__('swap', 2, [], label=label, duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        if False:
            i = 10
            return i + 15
        '\n        gate swap a,b { cx a,b; cx b,a; cx a,b; }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(CXGate(), [q[0], q[1]], []), (CXGate(), [q[1], q[0]], []), (CXGate(), [q[0], q[1]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def control(self, num_ctrl_qubits: int=1, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None):
        if False:
            print('Hello World!')
        "Return a (multi-)controlled-SWAP gate.\n\n        One control returns a CSWAP (Fredkin) gate.\n\n        Args:\n            num_ctrl_qubits (int): number of control qubits.\n            label (str or None): An optional label for the gate [Default: None]\n            ctrl_state (int or str or None): control state expressed as integer,\n                string (e.g. '110'), or None. If None, use all 1s.\n\n        Returns:\n            ControlledGate: controlled version of this gate.\n        "
        if num_ctrl_qubits == 1:
            gate = CSwapGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        if False:
            return 10
        'Return inverse Swap gate (itself).'
        return SwapGate()

@with_controlled_gate_array(_SWAP_ARRAY, num_ctrl_qubits=1)
class CSwapGate(SingletonControlledGate):
    """Controlled-SWAP gate, also known as the Fredkin gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cswap` and
    :meth:`~qiskit.circuit.QuantumCircuit.fredkin` methods.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ─■─
              │
        q_1: ─X─
              │
        q_2: ─X─


    **Matrix representation:**

    .. math::

        CSWAP\\ q_0, q_1, q_2 =
            I \\otimes I \\otimes |0 \\rangle \\langle 0| +
            SWAP \\otimes |1 \\rangle \\langle 1| =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\\\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\\\
            \\end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_2. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::

            q_0: ─X─
                  │
            q_1: ─X─
                  │
            q_2: ─■─

        .. math::

            CSWAP\\ q_2, q_1, q_0 =
                |0 \\rangle \\langle 0| \\otimes I \\otimes I +
                |1 \\rangle \\langle 1| \\otimes SWAP =
                \\begin{pmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\\\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\\\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\\\
                \\end{pmatrix}

    In the computational basis, this gate swaps the states of
    the two target qubits if the control qubit is in the
    :math:`|1\\rangle` state.

    .. math::
        |0, b, c\\rangle \\rightarrow |0, b, c\\rangle
        |1, b, c\\rangle \\rightarrow |1, c, b\\rangle
    """

    def __init__(self, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        if False:
            for i in range(10):
                print('nop')
        'Create new CSWAP gate.'
        if unit is None:
            unit = 'dt'
        super().__init__('cswap', 3, [], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=SwapGate(label=_base_label), duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        if False:
            return 10
        '\n        gate cswap a,b,c\n        { cx c,b;\n          ccx a,b,c;\n          cx c,b;\n        }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate, CCXGate
        q = QuantumRegister(3, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(CXGate(), [q[2], q[1]], []), (CCXGate(), [q[0], q[1], q[2]], []), (CXGate(), [q[2], q[1]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self):
        if False:
            i = 10
            return i + 15
        'Return inverse CSwap gate (itself).'
        return CSwapGate(ctrl_state=self.ctrl_state)