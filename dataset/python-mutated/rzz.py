"""Two-qubit ZZ-rotation gate."""
from cmath import exp
from typing import Optional
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType

class RZZGate(Gate):
    """A parametric 2-qubit :math:`Z \\otimes Z` interaction (rotation about ZZ).

    This gate is symmetric, and is maximally entangling at :math:`\\theta = \\pi/2`.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rzz` method.

    **Circuit Symbol:**

    .. parsed-literal::

        q_0: ───■────
                │zz(θ)
        q_1: ───■────

    **Matrix Representation:**

    .. math::

        \\newcommand{\\th}{\\frac{\\theta}{2}}

        R_{ZZ}(\\theta) = \\exp\\left(-i \\th Z{\\otimes}Z\\right) =
            \\begin{pmatrix}
                e^{-i \\th} & 0 & 0 & 0 \\\\
                0 & e^{i \\th} & 0 & 0 \\\\
                0 & 0 & e^{i \\th} & 0 \\\\
                0 & 0 & 0 & e^{-i \\th}
            \\end{pmatrix}

    This is a direct sum of RZ rotations, so this gate is equivalent to a
    uniformly controlled (multiplexed) RZ gate:

    .. math::

        R_{ZZ}(\\theta) =
            \\begin{pmatrix}
                RZ(\\theta) & 0 \\\\
                0 & RZ(-\\theta)
            \\end{pmatrix}

    **Examples:**

        .. math::

            R_{ZZ}(\\theta = 0) = I

        .. math::

            R_{ZZ}(\\theta = 2\\pi) = -I

        .. math::

            R_{ZZ}(\\theta = \\pi) = - Z \\otimes Z

        .. math::

            R_{ZZ}\\left(\\theta = \\frac{\\pi}{2}\\right) = \\frac{1}{\\sqrt{2}}
                                    \\begin{pmatrix}
                                        1-i & 0 & 0 & 0 \\\\
                                        0 & 1+i & 0 & 0 \\\\
                                        0 & 0 & 1+i & 0 \\\\
                                        0 & 0 & 0 & 1-i
                                    \\end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, label: Optional[str]=None, *, duration=None, unit='dt'):
        if False:
            while True:
                i = 10
        'Create new RZZ gate.'
        super().__init__('rzz', 2, [theta], label=label, duration=duration, unit=unit)

    def _define(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        gate rzz(theta) a, b { cx a, b; u1(theta) b; cx a, b; }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate
        from .rz import RZGate
        q = QuantumRegister(2, 'q')
        theta = self.params[0]
        qc = QuantumCircuit(q, name=self.name)
        rules = [(CXGate(), [q[0], q[1]], []), (RZGate(theta), [q[1]], []), (CXGate(), [q[0], q[1]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self):
        if False:
            print('Hello World!')
        'Return inverse RZZ gate (i.e. with the negative rotation angle).'
        return RZZGate(-self.params[0])

    def __array__(self, dtype=None):
        if False:
            i = 10
            return i + 15
        'Return a numpy.array for the RZZ gate.'
        import numpy
        itheta2 = 1j * float(self.params[0]) / 2
        return numpy.array([[exp(-itheta2), 0, 0, 0], [0, exp(itheta2), 0, 0], [0, 0, exp(itheta2), 0], [0, 0, 0, exp(-itheta2)]], dtype=dtype)

    def power(self, exponent: float):
        if False:
            return 10
        'Raise gate to a power.'
        (theta,) = self.params
        return RZZGate(exponent * theta)