"""Two-pulse single-qubit gate."""
import math
from cmath import exp
from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister

class U3Gate(Gate):
    """Generic single-qubit rotation gate with 3 Euler angles.

    .. warning::

       This gate is deprecated. Instead, the following replacements should be used

       .. math::

           U3(\\theta, \\phi, \\lambda) =  U(\\theta, \\phi, \\lambda)

       .. code-block:: python

          circuit = QuantumCircuit(1)
          circuit.u(theta, phi, lambda)

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤ U3(ϴ,φ,λ) ├
             └───────────┘

    **Matrix Representation:**

    .. math::

        \\newcommand{\\th}{\\frac{\\theta}{2}}

        U3(\\theta, \\phi, \\lambda) =
            \\begin{pmatrix}
                \\cos\\left(\\th\\right)          & -e^{i\\lambda}\\sin\\left(\\th\\right) \\\\
                e^{i\\phi}\\sin\\left(\\th\\right) & e^{i(\\phi+\\lambda)}\\cos\\left(\\th\\right)
            \\end{pmatrix}

    .. note::

        The matrix representation shown here differs from the `OpenQASM 2.0 specification
        <https://doi.org/10.48550/arXiv.1707.03429>`_ by a global phase of
        :math:`e^{i(\\phi+\\lambda)/2}`.

    **Examples:**

    .. math::

        U3(\\theta, \\phi, \\lambda) = e^{-i \\frac{\\pi + \\theta}{2}} P(\\phi + \\pi) \\sqrt{X}
        P(\\theta + \\pi) \\sqrt{X} P(\\lambda)

    .. math::

        U3\\left(\\theta, -\\frac{\\pi}{2}, \\frac{\\pi}{2}\\right) = RX(\\theta)

    .. math::

        U3(\\theta, 0, 0) = RY(\\theta)
    """

    def __init__(self, theta: ParameterValueType, phi: ParameterValueType, lam: ParameterValueType, label: Optional[str]=None, *, duration=None, unit='dt'):
        if False:
            while True:
                i = 10
        'Create new U3 gate.'
        super().__init__('u3', 1, [theta, phi, lam], label=label, duration=duration, unit=unit)

    def inverse(self):
        if False:
            print('Hello World!')
        'Return inverted U3 gate.\n\n        :math:`U3(\\theta,\\phi,\\lambda)^{\\dagger} =U3(-\\theta,-\\lambda,-\\phi)`)\n        '
        return U3Gate(-self.params[0], -self.params[2], -self.params[1])

    def control(self, num_ctrl_qubits: int=1, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None):
        if False:
            for i in range(10):
                print('nop')
        "Return a (multi-)controlled-U3 gate.\n\n        Args:\n            num_ctrl_qubits (int): number of control qubits.\n            label (str or None): An optional label for the gate [Default: None]\n            ctrl_state (int or str or None): control state expressed as integer,\n                string (e.g. '110'), or None. If None, use all 1s.\n\n        Returns:\n            ControlledGate: controlled version of this gate.\n        "
        if num_ctrl_qubits == 1:
            gate = CU3Gate(*self.params, label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def _define(self):
        if False:
            return 10
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        qc.u(self.params[0], self.params[1], self.params[2], 0)
        self.definition = qc

    def __array__(self, dtype=complex):
        if False:
            while True:
                i = 10
        'Return a Numpy.array for the U3 gate.'
        (theta, phi, lam) = self.params
        (theta, phi, lam) = (float(theta), float(phi), float(lam))
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        return numpy.array([[cos, -exp(1j * lam) * sin], [exp(1j * phi) * sin, exp(1j * (phi + lam)) * cos]], dtype=dtype)

class CU3Gate(ControlledGate):
    """Controlled-U3 gate (3-parameter two-qubit gate).

    This is a controlled version of the U3 gate (generic single qubit rotation).
    It is restricted to 3 parameters, and so cannot cover generic two-qubit
    controlled gates).

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──────■──────
             ┌─────┴─────┐
        q_1: ┤ U3(ϴ,φ,λ) ├
             └───────────┘

    **Matrix representation:**

    .. math::

        \\newcommand{\\th}{\\frac{\\theta}{2}}

        CU3(\\theta, \\phi, \\lambda)\\ q_0, q_1 =
            I \\otimes |0\\rangle\\langle 0| +
            U3(\\theta,\\phi,\\lambda) \\otimes |1\\rangle\\langle 1| =
            \\begin{pmatrix}
                1 & 0                   & 0 & 0 \\\\
                0 & \\cos(\\th)           & 0 & -e^{i\\lambda}\\sin(\\th) \\\\
                0 & 0                   & 1 & 0 \\\\
                0 & e^{i\\phi}\\sin(\\th)  & 0 & e^{i(\\phi+\\lambda)}\\cos(\\th)
            \\end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───────────┐
            q_0: ┤ U3(ϴ,φ,λ) ├
                 └─────┬─────┘
            q_1: ──────■──────

        .. math::

            CU3(\\theta, \\phi, \\lambda)\\ q_1, q_0 =
                |0\\rangle\\langle 0| \\otimes I +
                |1\\rangle\\langle 1| \\otimes U3(\\theta,\\phi,\\lambda) =
                \\begin{pmatrix}
                    1 & 0   & 0                  & 0 \\\\
                    0 & 1   & 0                  & 0 \\\\
                    0 & 0   & \\cos(\\th)          & -e^{i\\lambda}\\sin(\\th) \\\\
                    0 & 0   & e^{i\\phi}\\sin(\\th) & e^{i(\\phi+\\lambda)}\\cos(\\th)
                \\end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, phi: ParameterValueType, lam: ParameterValueType, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        if False:
            return 10
        'Create new CU3 gate.'
        super().__init__('cu3', 2, [theta, phi, lam], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=U3Gate(theta, phi, lam, label=_base_label), duration=duration, unit=unit)

    def _define(self):
        if False:
            while True:
                i = 10
        '\n        gate cu3(theta,phi,lambda) c, t\n        { u1((lambda+phi)/2) c;\n          u1((lambda-phi)/2) t;\n          cx c,t;\n          u3(-theta/2,0,-(phi+lambda)/2) t;\n          cx c,t;\n          u3(theta/2,phi,0) t;\n        }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate((self.params[2] + self.params[1]) / 2), [q[0]], []), (U1Gate((self.params[2] - self.params[1]) / 2), [q[1]], []), (CXGate(), [q[0], q[1]], []), (U3Gate(-self.params[0] / 2, 0, -(self.params[1] + self.params[2]) / 2), [q[1]], []), (CXGate(), [q[0], q[1]], []), (U3Gate(self.params[0] / 2, self.params[1], 0), [q[1]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self):
        if False:
            return 10
        'Return inverted CU3 gate.\n\n        :math:`CU3(\\theta,\\phi,\\lambda)^{\\dagger} =CU3(-\\theta,-\\phi,-\\lambda)`)\n        '
        return CU3Gate(-self.params[0], -self.params[2], -self.params[1], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=complex):
        if False:
            while True:
                i = 10
        'Return a numpy.array for the CU3 gate.'
        (theta, phi, lam) = self.params
        (theta, phi, lam) = (float(theta), float(phi), float(lam))
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0], [0, cos, 0, -exp(1j * lam) * sin], [0, 0, 1, 0], [0, exp(1j * phi) * sin, 0, exp(1j * (phi + lam)) * cos]], dtype=dtype)
        else:
            return numpy.array([[cos, 0, -exp(1j * lam) * sin, 0], [0, 1, 0, 0], [exp(1j * phi) * sin, 0, exp(1j * (phi + lam)) * cos, 0], [0, 0, 0, 1]], dtype=dtype)

def _generate_gray_code(num_bits):
    if False:
        for i in range(10):
            print('nop')
    'Generate the gray code for ``num_bits`` bits.'
    if num_bits <= 0:
        raise ValueError('Cannot generate the gray code for less than 1 bit.')
    result = [0]
    for i in range(num_bits):
        result += [x + 2 ** i for x in reversed(result)]
    return [format(x, '0%sb' % num_bits) for x in result]

def _gray_code_chain(q, num_ctrl_qubits, gate):
    if False:
        for i in range(10):
            print('nop')
    'Apply the gate to the last qubit in the register ``q``, controlled on all\n    preceding qubits. This function uses the gray code to propagate down to the last qubit.\n\n    Ported and adapted from Aqua (github.com/Qiskit/qiskit-aqua),\n    commit 769ca8d, file qiskit/aqua/circuits/gates/multi_control_u1_gate.py.\n    '
    from .x import CXGate
    rule = []
    (q_controls, q_target) = (q[:num_ctrl_qubits], q[num_ctrl_qubits])
    gray_code = _generate_gray_code(num_ctrl_qubits)
    last_pattern = None
    for pattern in gray_code:
        if '1' not in pattern:
            continue
        if last_pattern is None:
            last_pattern = pattern
        lm_pos = list(pattern).index('1')
        comp = [i != j for (i, j) in zip(pattern, last_pattern)]
        if True in comp:
            pos = comp.index(True)
        else:
            pos = None
        if pos is not None:
            if pos != lm_pos:
                rule.append((CXGate(), [q_controls[pos], q_controls[lm_pos]], []))
            else:
                indices = [i for (i, x) in enumerate(pattern) if x == '1']
                for idx in indices[1:]:
                    rule.append((CXGate(), [q_controls[idx], q_controls[lm_pos]], []))
        if pattern.count('1') % 2 == 0:
            rule.append((gate.inverse(), [q_controls[lm_pos], q_target], []))
        else:
            rule.append((gate, [q_controls[lm_pos], q_target], []))
        last_pattern = pattern
    return rule