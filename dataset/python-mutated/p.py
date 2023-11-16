"""Phase Gate."""
from __future__ import annotations
from cmath import exp
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType

class PhaseGate(Gate):
    """Single-qubit rotation about the Z axis.

    This is a diagonal gate. It can be implemented virtually in hardware
    via framechanges (i.e. at zero error and duration).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.p` method.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────┐
        q_0: ┤ P(λ) ├
             └──────┘

    **Matrix Representation:**

    .. math::

        P(\\lambda) =
            \\begin{pmatrix}
                1 & 0 \\\\
                0 & e^{i\\lambda}
            \\end{pmatrix}

    **Examples:**

        .. math::

            P(\\lambda = \\pi) = Z

        .. math::

            P(\\lambda = \\pi/2) = S

        .. math::

            P(\\lambda = \\pi/4) = T

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.RZGate`:
        This gate is equivalent to RZ up to a phase factor.

            .. math::

                P(\\lambda) = e^{i{\\lambda}/2} RZ(\\lambda)

        Reference for virtual Z gate implementation:
        `1612.00858 <https://arxiv.org/abs/1612.00858>`_
    """

    def __init__(self, theta: ParameterValueType, label: str | None=None, *, duration=None, unit='dt'):
        if False:
            while True:
                i = 10
        'Create new Phase gate.'
        super().__init__('p', 1, [theta], label=label, duration=duration, unit=unit)

    def _define(self):
        if False:
            i = 10
            return i + 15
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u import UGate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        qc.append(UGate(0, 0, self.params[0]), [0])
        self.definition = qc

    def control(self, num_ctrl_qubits: int=1, label: str | None=None, ctrl_state: str | int | None=None):
        if False:
            print('Hello World!')
        "Return a (multi-)controlled-Phase gate.\n\n        Args:\n            num_ctrl_qubits (int): number of control qubits.\n            label (str or None): An optional label for the gate [Default: None]\n            ctrl_state (int or str or None): control state expressed as integer,\n                string (e.g. '110'), or None. If None, use all 1s.\n\n        Returns:\n            ControlledGate: controlled version of this gate.\n        "
        if num_ctrl_qubits == 1:
            gate = CPhaseGate(self.params[0], label=label, ctrl_state=ctrl_state)
        elif ctrl_state is None and num_ctrl_qubits > 1:
            gate = MCPhaseGate(self.params[0], num_ctrl_qubits, label=label)
        else:
            return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        if False:
            for i in range(10):
                print('nop')
        'Return inverted Phase gate (:math:`Phase(\\lambda)^{\\dagger} = Phase(-\\lambda)`)'
        return PhaseGate(-self.params[0])

    def __array__(self, dtype=None):
        if False:
            i = 10
            return i + 15
        'Return a numpy.array for the Phase gate.'
        lam = float(self.params[0])
        return numpy.array([[1, 0], [0, exp(1j * lam)]], dtype=dtype)

    def power(self, exponent: float):
        if False:
            for i in range(10):
                print('nop')
        'Raise gate to a power.'
        (theta,) = self.params
        return PhaseGate(exponent * theta)

class CPhaseGate(ControlledGate):
    """Controlled-Phase gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the control state.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cp` method.

    **Circuit symbol:**

    .. parsed-literal::


        q_0: ─■──
              │λ
        q_1: ─■──


    **Matrix representation:**

    .. math::

        CPhase =
            I \\otimes |0\\rangle\\langle 0| + P \\otimes |1\\rangle\\langle 1| =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & 1 & 0 & 0 \\\\
                0 & 0 & 1 & 0 \\\\
                0 & 0 & 0 & e^{i\\lambda}
            \\end{pmatrix}

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CRZGate`:
        Due to the global phase difference in the matrix definitions
        of Phase and RZ, CPhase and CRZ are different gates with a relative
        phase difference.
    """

    def __init__(self, theta: ParameterValueType, label: str | None=None, ctrl_state: str | int | None=None, *, duration=None, unit='dt', _base_label=None):
        if False:
            i = 10
            return i + 15
        'Create new CPhase gate.'
        super().__init__('cp', 2, [theta], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=PhaseGate(theta, label=_base_label), duration=duration, unit=unit)

    def _define(self):
        if False:
            i = 10
            return i + 15
        '\n        gate cphase(lambda) a,b\n        { phase(lambda/2) a; cx a,b;\n          phase(-lambda/2) b; cx a,b;\n          phase(lambda/2) b;\n        }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        qc.p(self.params[0] / 2, 0)
        qc.cx(0, 1)
        qc.p(-self.params[0] / 2, 1)
        qc.cx(0, 1)
        qc.p(self.params[0] / 2, 1)
        self.definition = qc

    def control(self, num_ctrl_qubits: int=1, label: str | None=None, ctrl_state: str | int | None=None):
        if False:
            print('Hello World!')
        "Controlled version of this gate.\n\n        Args:\n            num_ctrl_qubits (int): number of control qubits.\n            label (str or None): An optional label for the gate [Default: None]\n            ctrl_state (int or str or None): control state expressed as integer,\n                string (e.g. '110'), or None. If None, use all 1s.\n\n        Returns:\n            ControlledGate: controlled version of this gate.\n        "
        if ctrl_state is None:
            gate = MCPhaseGate(self.params[0], num_ctrl_qubits=num_ctrl_qubits + 1, label=label)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        if False:
            print('Hello World!')
        'Return inverted CPhase gate (:math:`CPhase(\\lambda)^{\\dagger} = CPhase(-\\lambda)`)'
        return CPhaseGate(-self.params[0], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None):
        if False:
            for i in range(10):
                print('nop')
        'Return a numpy.array for the CPhase gate.'
        eith = exp(1j * float(self.params[0]))
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, eith]], dtype=dtype)
        return numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, eith, 0], [0, 0, 0, 1]], dtype=dtype)

    def power(self, exponent: float):
        if False:
            for i in range(10):
                print('nop')
        'Raise gate to a power.'
        (theta,) = self.params
        return CPhaseGate(exponent * theta)

class MCPhaseGate(ControlledGate):
    """Multi-controlled-Phase gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the state of the control qubits.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.mcp` method.

    **Circuit symbol:**

    .. parsed-literal::

            q_0: ───■────
                    │
                    .
                    │
        q_(n-1): ───■────
                 ┌──┴───┐
            q_n: ┤ P(λ) ├
                 └──────┘

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CPhaseGate`:
        The singly-controlled-version of this gate.
    """

    def __init__(self, lam: ParameterValueType, num_ctrl_qubits: int, label: str | None=None, *, duration=None, unit='dt', _base_label=None):
        if False:
            while True:
                i = 10
        'Create new MCPhase gate.'
        super().__init__('mcphase', num_ctrl_qubits + 1, [lam], num_ctrl_qubits=num_ctrl_qubits, label=label, base_gate=PhaseGate(lam, label=_base_label), duration=duration, unit=unit)

    def _define(self):
        if False:
            return 10
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(self.num_qubits, 'q')
        qc = QuantumCircuit(q, name=self.name)
        if self.num_ctrl_qubits == 0:
            qc.p(self.params[0], 0)
        if self.num_ctrl_qubits == 1:
            qc.cp(self.params[0], 0, 1)
        else:
            from .u3 import _gray_code_chain
            scaled_lam = self.params[0] / 2 ** (self.num_ctrl_qubits - 1)
            bottom_gate = CPhaseGate(scaled_lam)
            for (operation, qubits, clbits) in _gray_code_chain(q, self.num_ctrl_qubits, bottom_gate):
                qc._append(operation, qubits, clbits)
        self.definition = qc

    def control(self, num_ctrl_qubits: int=1, label: str | None=None, ctrl_state: str | int | None=None):
        if False:
            return 10
        "Controlled version of this gate.\n\n        Args:\n            num_ctrl_qubits (int): number of control qubits.\n            label (str or None): An optional label for the gate [Default: None]\n            ctrl_state (int or str or None): control state expressed as integer,\n                string (e.g. '110'), or None. If None, use all 1s.\n\n        Returns:\n            ControlledGate: controlled version of this gate.\n        "
        if ctrl_state is None:
            gate = MCPhaseGate(self.params[0], num_ctrl_qubits=num_ctrl_qubits + self.num_ctrl_qubits, label=label)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        if False:
            while True:
                i = 10
        'Return inverted MCU1 gate (:math:`MCU1(\\lambda)^{\\dagger} = MCU1(-\\lambda)`)'
        return MCPhaseGate(-self.params[0], self.num_ctrl_qubits)