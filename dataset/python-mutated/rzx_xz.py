"""
RZX based template for CX - RXGate - CX
.. parsed-literal::
     ┌───┐         ┌───┐┌─────────┐┌─────────┐┌─────────┐┌──────────┐»
q_0: ┤ X ├─────────┤ X ├┤ RZ(π/2) ├┤ RX(π/2) ├┤ RZ(π/2) ├┤0         ├»
     └─┬─┘┌───────┐└─┬─┘└─────────┘└─────────┘└─────────┘│  RZX(-ϴ) │»
q_1: ──■──┤ RX(ϴ) ├──■───────────────────────────────────┤1         ├»
          └───────┘                                      └──────────┘»
«     ┌─────────┐┌─────────┐┌─────────┐
«q_0: ┤ RZ(π/2) ├┤ RX(π/2) ├┤ RZ(π/2) ├
«     └─────────┘└─────────┘└─────────┘
«q_1: ─────────────────────────────────
«
"""
from __future__ import annotations
import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType

def rzx_xz(theta: ParameterValueType | None=None):
    if False:
        print('Hello World!')
    'Template for CX - RXGate - CX.'
    if theta is None:
        theta = Parameter('ϴ')
    qc = QuantumCircuit(2)
    qc.cx(1, 0)
    qc.rx(theta, 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    qc.rx(np.pi / 2, 0)
    qc.rz(np.pi / 2, 0)
    qc.rzx(-1 * theta, 0, 1)
    qc.rz(np.pi / 2, 0)
    qc.rx(np.pi / 2, 0)
    qc.rz(np.pi / 2, 0)
    return qc