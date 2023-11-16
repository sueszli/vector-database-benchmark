"""Rotation around the Y axis."""
import math
from math import pi
from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType

class RYGate(Gate):
    """Single-qubit rotation about the Y axis.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ry` method.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────┐
        q_0: ┤ Ry(ϴ) ├
             └───────┘

    **Matrix Representation:**

    .. math::

        \\newcommand{\\th}{\\frac{\\theta}{2}}

        RY(\\theta) = \\exp\\left(-i \\th Y\\right) =
            \\begin{pmatrix}
                \\cos\\left(\\th\\right) & -\\sin\\left(\\th\\right) \\\\
                \\sin\\left(\\th\\right) & \\cos\\left(\\th\\right)
            \\end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, label: Optional[str]=None, *, duration=None, unit='dt'):
        if False:
            return 10
        'Create new RY gate.'
        super().__init__('ry', 1, [theta], label=label, duration=duration, unit=unit)

    def _define(self):
        if False:
            i = 10
            return i + 15
        '\n        gate ry(theta) a { r(theta, pi/2) a; }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .r import RGate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(RGate(self.params[0], pi / 2), [q[0]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def control(self, num_ctrl_qubits: int=1, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None):
        if False:
            return 10
        "Return a (multi-)controlled-RY gate.\n\n        Args:\n            num_ctrl_qubits (int): number of control qubits.\n            label (str or None): An optional label for the gate [Default: None]\n            ctrl_state (int or str or None): control state expressed as integer,\n                string (e.g. '110'), or None. If None, use all 1s.\n\n        Returns:\n            ControlledGate: controlled version of this gate.\n        "
        if num_ctrl_qubits == 1:
            gate = CRYGate(self.params[0], label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        if False:
            print('Hello World!')
        'Return inverted RY gate.\n\n        :math:`RY(\\lambda)^{\\dagger} = RY(-\\lambda)`\n        '
        return RYGate(-self.params[0])

    def __array__(self, dtype=None):
        if False:
            for i in range(10):
                print('nop')
        'Return a numpy.array for the RY gate.'
        cos = math.cos(self.params[0] / 2)
        sin = math.sin(self.params[0] / 2)
        return numpy.array([[cos, -sin], [sin, cos]], dtype=dtype)

    def power(self, exponent: float):
        if False:
            for i in range(10):
                print('nop')
        'Raise gate to a power.'
        (theta,) = self.params
        return RYGate(exponent * theta)

class CRYGate(ControlledGate):
    """Controlled-RY gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cry` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ────■────
             ┌───┴───┐
        q_1: ┤ Ry(ϴ) ├
             └───────┘

    **Matrix representation:**

    .. math::

        \\newcommand{\\th}{\\frac{\\theta}{2}}

        CRY(\\theta)\\ q_0, q_1 =
            I \\otimes |0\\rangle\\langle 0| + RY(\\theta) \\otimes |1\\rangle\\langle 1| =
            \\begin{pmatrix}
                1 & 0         & 0 & 0 \\\\
                0 & \\cos\\left(\\th\\right) & 0 & -\\sin\\left(\\th\\right) \\\\
                0 & 0         & 1 & 0 \\\\
                0 & \\sin\\left(\\th\\right) & 0 & \\cos\\left(\\th\\right)
            \\end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───────┐
            q_0: ┤ Ry(ϴ) ├
                 └───┬───┘
            q_1: ────■────

        .. math::

            \\newcommand{\\th}{\\frac{\\theta}{2}}

            CRY(\\theta)\\ q_1, q_0 =
            |0\\rangle\\langle 0| \\otimes I + |1\\rangle\\langle 1| \\otimes RY(\\theta) =
                \\begin{pmatrix}
                    1 & 0 & 0 & 0 \\\\
                    0 & 1 & 0 & 0 \\\\
                    0 & 0 & \\cos\\left(\\th\\right) & -\\sin\\left(\\th\\right) \\\\
                    0 & 0 & \\sin\\left(\\th\\right) & \\cos\\left(\\th\\right)
                \\end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        if False:
            print('Hello World!')
        'Create new CRY gate.'
        super().__init__('cry', 2, [theta], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=RYGate(theta, label=_base_label), duration=duration, unit=unit)

    def _define(self):
        if False:
            print('Hello World!')
        '\n        gate cry(lambda) a,b\n        { u3(lambda/2,0,0) b; cx a,b;\n          u3(-lambda/2,0,0) b; cx a,b;\n        }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(RYGate(self.params[0] / 2), [q[1]], []), (CXGate(), [q[0], q[1]], []), (RYGate(-self.params[0] / 2), [q[1]], []), (CXGate(), [q[0], q[1]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self):
        if False:
            for i in range(10):
                print('nop')
        'Return inverse CRY gate (i.e. with the negative rotation angle).'
        return CRYGate(-self.params[0], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None):
        if False:
            for i in range(10):
                print('nop')
        'Return a numpy.array for the CRY gate.'
        half_theta = float(self.params[0]) / 2
        cos = math.cos(half_theta)
        sin = math.sin(half_theta)
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0], [0, cos, 0, -sin], [0, 0, 1, 0], [0, sin, 0, cos]], dtype=dtype)
        else:
            return numpy.array([[cos, 0, -sin, 0], [0, 1, 0, 0], [sin, 0, cos, 0], [0, 0, 0, 1]], dtype=dtype)