"""T and Tdg gate."""
import math
from math import pi
from typing import Optional
import numpy
from qiskit.circuit.singleton import SingletonGate, stdlib_singleton_key
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array

@with_gate_array([[1, 0], [0, (1 + 1j) / math.sqrt(2)]])
class TGate(SingletonGate):
    """Single qubit T gate (Z**0.25).

    It induces a :math:`\\pi/4` phase, and is sometimes called the pi/8 gate
    (because of how the RZ(\\pi/4) matrix looks like).

    This is a non-Clifford gate and a fourth-root of Pauli-Z.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.t` method.

    **Matrix Representation:**

    .. math::

        T = \\begin{pmatrix}
                1 & 0 \\\\
                0 & e^{i\\pi/4}
            \\end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ T ├
             └───┘

    Equivalent to a :math:`\\pi/4` radian rotation about the Z axis.
    """

    def __init__(self, label: Optional[str]=None, *, duration=None, unit='dt'):
        if False:
            for i in range(10):
                print('nop')
        'Create new T gate.'
        super().__init__('t', 1, [], label=label, duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        if False:
            while True:
                i = 10
        '\n        gate t a { u1(pi/4) a; }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate(pi / 4), [q[0]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self):
        if False:
            for i in range(10):
                print('nop')
        'Return inverse T gate (i.e. Tdg).'
        return TdgGate()

    def power(self, exponent: float):
        if False:
            return 10
        'Raise gate to a power.'
        return PhaseGate(0.25 * numpy.pi * exponent)

@with_gate_array([[1, 0], [0, (1 - 1j) / math.sqrt(2)]])
class TdgGate(SingletonGate):
    """Single qubit T-adjoint gate (~Z**0.25).

    It induces a :math:`-\\pi/4` phase.

    This is a non-Clifford gate and a fourth-root of Pauli-Z.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.tdg` method.

    **Matrix Representation:**

    .. math::

        Tdg = \\begin{pmatrix}
                1 & 0 \\\\
                0 & e^{-i\\pi/4}
            \\end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌─────┐
        q_0: ┤ Tdg ├
             └─────┘

    Equivalent to a :math:`-\\pi/4` radian rotation about the Z axis.
    """

    def __init__(self, label: Optional[str]=None, *, duration=None, unit='dt'):
        if False:
            while True:
                i = 10
        'Create new Tdg gate.'
        super().__init__('tdg', 1, [], label=label, duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        if False:
            i = 10
            return i + 15
        '\n        gate tdg a { u1(pi/4) a; }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate(-pi / 4), [q[0]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self):
        if False:
            return 10
        'Return inverse Tdg gate (i.e. T).'
        return TGate()

    def power(self, exponent: float):
        if False:
            for i in range(10):
                print('nop')
        'Raise gate to a power.'
        return PhaseGate(-0.25 * numpy.pi * exponent)