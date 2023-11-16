"""Z, CZ and CCZ gates."""
from math import pi
from typing import Optional, Union
import numpy
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from .p import PhaseGate
_Z_ARRAY = [[1, 0], [0, -1]]

@with_gate_array(_Z_ARRAY)
class ZGate(SingletonGate):
    """The single-qubit Pauli-Z gate (:math:`\\sigma_z`).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.z` method.

    **Matrix Representation:**

    .. math::

        Z = \\begin{pmatrix}
                1 & 0 \\\\
                0 & -1
            \\end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ Z ├
             └───┘

    Equivalent to a :math:`\\pi` radian rotation about the Z axis.

    .. note::

        A global phase difference exists between the definitions of
        :math:`RZ(\\pi)` and :math:`Z`.

        .. math::

            RZ(\\pi) = \\begin{pmatrix}
                        -i & 0 \\\\
                        0 & i
                      \\end{pmatrix}
                    = -i Z

    The gate is equivalent to a phase flip.

    .. math::

        |0\\rangle \\rightarrow |0\\rangle \\\\
        |1\\rangle \\rightarrow -|1\\rangle
    """

    def __init__(self, label: Optional[str]=None, *, duration=None, unit='dt'):
        if False:
            while True:
                i = 10
        'Create new Z gate.'
        super().__init__('z', 1, [], label=label, duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        if False:
            i = 10
            return i + 15
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate(pi), [q[0]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def control(self, num_ctrl_qubits: int=1, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None):
        if False:
            while True:
                i = 10
        "Return a (multi-)controlled-Z gate.\n\n        One control returns a CZ gate.\n\n        Args:\n            num_ctrl_qubits (int): number of control qubits.\n            label (str or None): An optional label for the gate [Default: None]\n            ctrl_state (int or str or None): control state expressed as integer,\n                string (e.g. '110'), or None. If None, use all 1s.\n\n        Returns:\n            ControlledGate: controlled version of this gate.\n        "
        if num_ctrl_qubits == 1:
            gate = CZGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        if False:
            i = 10
            return i + 15
        'Return inverted Z gate (itself).'
        return ZGate()

    def power(self, exponent: float):
        if False:
            return 10
        'Raise gate to a power.'
        return PhaseGate(numpy.pi * exponent)

@with_controlled_gate_array(_Z_ARRAY, num_ctrl_qubits=1)
class CZGate(SingletonControlledGate):
    """Controlled-Z gate.

    This is a Clifford and symmetric gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cz` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ─■─
              │
        q_1: ─■─

    **Matrix representation:**

    .. math::

        CZ\\ q_0, q_1 =
            I \\otimes |0\\rangle\\langle 0| + Z \\otimes |1\\rangle\\langle 1| =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & 1 & 0 & 0 \\\\
                0 & 0 & 1 & 0 \\\\
                0 & 0 & 0 & -1
            \\end{pmatrix}

    In the computational basis, this gate flips the phase of
    the target qubit if the control qubit is in the :math:`|1\\rangle` state.
    """

    def __init__(self, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        if False:
            for i in range(10):
                print('nop')
        'Create new CZ gate.'
        super().__init__('cz', 2, [], label=label, num_ctrl_qubits=1, ctrl_state=ctrl_state, base_gate=ZGate(label=_base_label), duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        if False:
            i = 10
            return i + 15
        '\n        gate cz a,b { h b; cx a,b; h b; }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .h import HGate
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(HGate(), [q[1]], []), (CXGate(), [q[0], q[1]], []), (HGate(), [q[1]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self):
        if False:
            for i in range(10):
                print('nop')
        'Return inverted CZ gate (itself).'
        return CZGate(ctrl_state=self.ctrl_state)

@with_controlled_gate_array(_Z_ARRAY, num_ctrl_qubits=2, cached_states=(3,))
class CCZGate(SingletonControlledGate):
    """CCZ gate.

    This is a symmetric gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ccz` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ─■─
              │
        q_1: ─■─
              │
        q_2: ─■─

    **Matrix representation:**

    .. math::

        CCZ\\ q_0, q_1, q_2 =
            I \\otimes I \\otimes |0\\rangle\\langle 0| + CZ \\otimes |1\\rangle\\langle 1| =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\\\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
            \\end{pmatrix}

    In the computational basis, this gate flips the phase of
    the target qubit if the control qubits are in the :math:`|11\\rangle` state.
    """

    def __init__(self, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        if False:
            return 10
        'Create new CCZ gate.'
        super().__init__('ccz', 3, [], label=label, num_ctrl_qubits=2, ctrl_state=ctrl_state, base_gate=ZGate(label=_base_label), duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=2)

    def _define(self):
        if False:
            return 10
        '\n        gate ccz a,b,c { h c; ccx a,b,c; h c; }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .h import HGate
        from .x import CCXGate
        q = QuantumRegister(3, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(HGate(), [q[2]], []), (CCXGate(), [q[0], q[1], q[2]], []), (HGate(), [q[2]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self):
        if False:
            i = 10
            return i + 15
        'Return inverted CCZ gate (itself).'
        return CCZGate(ctrl_state=self.ctrl_state)