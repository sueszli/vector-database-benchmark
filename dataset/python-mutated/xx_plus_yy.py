"""Two-qubit XX+YY gate."""
import math
from cmath import exp
from math import pi
from typing import Optional
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType

class XXPlusYYGate(Gate):
    """XX+YY interaction gate.

    A 2-qubit parameterized XX+YY interaction, also known as an XY gate. Its action is to induce
    a coherent rotation by some angle between :math:`|01\\rangle` and :math:`|10\\rangle`.

    **Circuit Symbol:**

    .. parsed-literal::

             ┌───────────────┐
        q_0: ┤0              ├
             │  (XX+YY)(θ,β) │
        q_1: ┤1              ├
             └───────────────┘

    **Matrix Representation:**

    .. math::

        \\newcommand{\\th}{\\frac{\\theta}{2}}

        R_{XX+YY}(\\theta, \\beta)\\ q_0, q_1 =
          RZ_0(-\\beta) \\cdot \\exp\\left(-i \\frac{\\theta}{2} \\frac{XX+YY}{2}\\right) \\cdot RZ_0(\\beta) =
            \\begin{pmatrix}
                1 & 0                     & 0                    & 0  \\\\
                0 & \\cos\\left(\\th\\right)             & -i\\sin\\left(\\th\\right)e^{-i\\beta} & 0  \\\\
                0 & -i\\sin\\left(\\th\\right)e^{i\\beta} & \\cos\\left(\\th\\right)            & 0  \\\\
                0 & 0                     & 0                    & 1
            \\end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In the above example we apply the gate
        on (q_0, q_1) which results in adding the (optional) phase defined
        by :math:`beta` on q_0. Instead, if we apply it on (q_1, q_0), the
        phase is added on q_1. If :math:`beta` is set to its default value
        of :math:`0`, the gate is equivalent in big and little endian.

        .. parsed-literal::

                 ┌───────────────┐
            q_0: ┤1              ├
                 │  (XX+YY)(θ,β) │
            q_1: ┤0              ├
                 └───────────────┘

    .. math::

        \\newcommand{\\th}{\\frac{\\theta}{2}}

        R_{XX+YY}(\\theta, \\beta)\\ q_0, q_1 =
          RZ_1(-\\beta) \\cdot \\exp\\left(-i \\frac{\\theta}{2} \\frac{XX+YY}{2}\\right) \\cdot RZ_1(\\beta) =
            \\begin{pmatrix}
                1 & 0                     & 0                    & 0  \\\\
                0 & \\cos\\left(\\th\\right)             & -i\\sin\\left(\\th\\right)e^{i\\beta} & 0  \\\\
                0 & -i\\sin\\left(\\th\\right)e^{-i\\beta} & \\cos\\left(\\th\\right)            & 0  \\\\
                0 & 0                     & 0                    & 1
            \\end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, beta: ParameterValueType=0, label: Optional[str]='(XX+YY)', *, duration=None, unit='dt'):
        if False:
            for i in range(10):
                print('nop')
        'Create new XX+YY gate.\n\n        Args:\n            theta: The rotation angle.\n            beta: The phase angle.\n            label: The label of the gate.\n        '
        super().__init__('xx_plus_yy', 2, [theta, beta], label=label, duration=duration, unit=unit)

    def _define(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        gate xx_plus_yy(theta, beta) a, b {\n            rz(beta) b;\n            rz(-pi/2) a;\n            sx a;\n            rz(pi/2) a;\n            s b;\n            cx a, b;\n            ry(theta/2) a;\n            ry(theta/2) b;\n            cx a, b;\n            sdg b;\n            rz(-pi/2) a;\n            sxdg a;\n            rz(pi/2) a;\n            rz(-beta) b;\n        }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate
        from .s import SGate, SdgGate
        from .sx import SXGate, SXdgGate
        from .rz import RZGate
        from .ry import RYGate
        theta = self.params[0]
        beta = self.params[1]
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(RZGate(beta), [q[0]], []), (RZGate(-pi / 2), [q[1]], []), (SXGate(), [q[1]], []), (RZGate(pi / 2), [q[1]], []), (SGate(), [q[0]], []), (CXGate(), [q[1], q[0]], []), (RYGate(-theta / 2), [q[1]], []), (RYGate(-theta / 2), [q[0]], []), (CXGate(), [q[1], q[0]], []), (SdgGate(), [q[0]], []), (RZGate(-pi / 2), [q[1]], []), (SXdgGate(), [q[1]], []), (RZGate(pi / 2), [q[1]], []), (RZGate(-beta), [q[0]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self):
        if False:
            while True:
                i = 10
        'Return inverse XX+YY gate (i.e. with the negative rotation angle and same phase angle).'
        return XXPlusYYGate(-self.params[0], self.params[1])

    def __array__(self, dtype=complex):
        if False:
            print('Hello World!')
        'Return a numpy.array for the XX+YY gate.'
        import numpy
        half_theta = float(self.params[0]) / 2
        beta = float(self.params[1])
        cos = math.cos(half_theta)
        sin = math.sin(half_theta)
        return numpy.array([[1, 0, 0, 0], [0, cos, -1j * sin * exp(-1j * beta), 0], [0, -1j * sin * exp(1j * beta), cos, 0], [0, 0, 0, 1]], dtype=dtype)

    def power(self, exponent: float):
        if False:
            while True:
                i = 10
        'Raise gate to a power.'
        (theta, beta) = self.params
        return XXPlusYYGate(exponent * theta, beta)