"""Hadamard gate."""
from math import sqrt, pi
from typing import Optional, Union
import numpy
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
_H_ARRAY = 1 / sqrt(2) * numpy.array([[1, 1], [1, -1]], dtype=numpy.complex128)

@with_gate_array(_H_ARRAY)
class HGate(SingletonGate):
    """Single-qubit Hadamard gate.

    This gate is a \\pi rotation about the X+Z axis, and has the effect of
    changing computation basis from :math:`|0\\rangle,|1\\rangle` to
    :math:`|+\\rangle,|-\\rangle` and vice-versa.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.h` method.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ H ├
             └───┘

    **Matrix Representation:**

    .. math::

        H = \\frac{1}{\\sqrt{2}}
            \\begin{pmatrix}
                1 & 1 \\\\
                1 & -1
            \\end{pmatrix}
    """

    def __init__(self, label: Optional[str]=None, *, duration=None, unit='dt'):
        if False:
            print('Hello World!')
        'Create new H gate.'
        super().__init__('h', 1, [], label=label, duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        if False:
            return 10
        '\n        gate h a { u2(0,pi) a; }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u2 import U2Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U2Gate(0, pi), [q[0]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def control(self, num_ctrl_qubits: int=1, label: Optional[str]=None, ctrl_state: Optional[Union[int, str]]=None):
        if False:
            i = 10
            return i + 15
        "Return a (multi-)controlled-H gate.\n\n        One control qubit returns a CH gate.\n\n        Args:\n            num_ctrl_qubits (int): number of control qubits.\n            label (str or None): An optional label for the gate [Default: None]\n            ctrl_state (int or str or None): control state expressed as integer,\n                string (e.g. '110'), or None. If None, use all 1s.\n\n        Returns:\n            ControlledGate: controlled version of this gate.\n        "
        if num_ctrl_qubits == 1:
            gate = CHGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        if False:
            for i in range(10):
                print('nop')
        'Return inverted H gate (itself).'
        return HGate()

@with_controlled_gate_array(_H_ARRAY, num_ctrl_qubits=1)
class CHGate(SingletonControlledGate):
    """Controlled-Hadamard gate.

    Applies a Hadamard on the target qubit if the control is
    in the :math:`|1\\rangle` state.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ch` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ H ├
             └───┘

    **Matrix Representation:**

    .. math::

        CH\\ q_0, q_1 =
            I \\otimes |0\\rangle\\langle 0| + H \\otimes |1\\rangle\\langle 1| =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & \\frac{1}{\\sqrt{2}} & 0 & \\frac{1}{\\sqrt{2}} \\\\
                0 & 0 & 1 & 0 \\\\
                0 & \\frac{1}{\\sqrt{2}} & 0 & -\\frac{1}{\\sqrt{2}}
            \\end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───┐
            q_0: ┤ H ├
                 └─┬─┘
            q_1: ──■──

        .. math::

            CH\\ q_1, q_0 =
                |0\\rangle\\langle 0| \\otimes I + |1\\rangle\\langle 1| \\otimes H =
                \\begin{pmatrix}
                    1 & 0 & 0 & 0 \\\\
                    0 & 1 & 0 & 0 \\\\
                    0 & 0 & \\frac{1}{\\sqrt{2}} & \\frac{1}{\\sqrt{2}} \\\\
                    0 & 0 & \\frac{1}{\\sqrt{2}} & -\\frac{1}{\\sqrt{2}}
                \\end{pmatrix}
    """

    def __init__(self, label: Optional[str]=None, ctrl_state: Optional[Union[int, str]]=None, *, duration=None, unit='dt', _base_label=None):
        if False:
            i = 10
            return i + 15
        'Create new CH gate.'
        super().__init__('ch', 2, [], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=HGate(label=_base_label), duration=duration, unit=unit, _base_label=_base_label)
    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        if False:
            while True:
                i = 10
        '\n        gate ch a,b {\n            s b;\n            h b;\n            t b;\n            cx a, b;\n            tdg b;\n            h b;\n            sdg b;\n        }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate
        from .t import TGate, TdgGate
        from .s import SGate, SdgGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(SGate(), [q[1]], []), (HGate(), [q[1]], []), (TGate(), [q[1]], []), (CXGate(), [q[0], q[1]], []), (TdgGate(), [q[1]], []), (HGate(), [q[1]], []), (SdgGate(), [q[1]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self):
        if False:
            while True:
                i = 10
        'Return inverted CH gate (itself).'
        return CHGate(ctrl_state=self.ctrl_state)