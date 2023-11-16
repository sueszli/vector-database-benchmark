"""iSWAP gate."""
from typing import Optional
import numpy as np
from qiskit.circuit.singleton import SingletonGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array
from .xx_plus_yy import XXPlusYYGate

@with_gate_array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
class iSwapGate(SingletonGate):
    """iSWAP gate.

    A 2-qubit XX+YY interaction.
    This is a Clifford and symmetric gate. Its action is to swap two qubit
    states and phase the :math:`|01\\rangle` and :math:`|10\\rangle`
    amplitudes by i.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.iswap` method.

    **Circuit Symbol:**

    .. parsed-literal::

        q_0: ─⨂─
              │
        q_1: ─⨂─

    **Reference Implementation:**

    .. parsed-literal::

             ┌───┐┌───┐     ┌───┐
        q_0: ┤ S ├┤ H ├──■──┤ X ├─────
             ├───┤└───┘┌─┴─┐└─┬─┘┌───┐
        q_1: ┤ S ├─────┤ X ├──■──┤ H ├
             └───┘     └───┘     └───┘

    **Matrix Representation:**

    .. math::

        iSWAP = R_{XX+YY}\\left(-\\frac{\\pi}{2}\\right)
          = \\exp\\left(i \\frac{\\pi}{4} \\left(X{\\otimes}X+Y{\\otimes}Y\\right)\\right) =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & 0 & i & 0 \\\\
                0 & i & 0 & 0 \\\\
                0 & 0 & 0 & 1
            \\end{pmatrix}

    This gate is equivalent to a SWAP up to a diagonal.

    .. math::

         iSWAP =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & 0 & 1 & 0 \\\\
                0 & 1 & 0 & 0 \\\\
                0 & 0 & 0 & 1
            \\end{pmatrix}
         .  \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & i & 0 & 0 \\\\
                0 & 0 & i & 0 \\\\
                0 & 0 & 0 & 1
            \\end{pmatrix}
    """

    def __init__(self, label: Optional[str]=None, *, duration=None, unit='dt'):
        if False:
            return 10
        'Create new iSwap gate.'
        super().__init__('iswap', 2, [], label=label, duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        if False:
            return 10
        '\n        gate iswap a,b {\n            s q[0];\n            s q[1];\n            h q[0];\n            cx q[0],q[1];\n            cx q[1],q[0];\n            h q[1];\n        }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .h import HGate
        from .s import SGate
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(SGate(), [q[0]], []), (SGate(), [q[1]], []), (HGate(), [q[0]], []), (CXGate(), [q[0], q[1]], []), (CXGate(), [q[1], q[0]], []), (HGate(), [q[1]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def power(self, exponent: float):
        if False:
            return 10
        'Raise gate to a power.'
        return XXPlusYYGate(-np.pi * exponent)