"""
RZX based template for CX - RZGate - CX
.. parsed-literal::
                                                                            »
q_0: ──■─────────────■──────────────────────────────────────────────────────»
     ┌─┴─┐┌───────┐┌─┴─┐┌────────┐┌─────────┐┌─────────┐┌─────────┐┌───────┐»
q_1: ┤ X ├┤ RZ(ϴ) ├┤ X ├┤ RZ(-ϴ) ├┤ RZ(π/2) ├┤ RX(π/2) ├┤ RZ(π/2) ├┤ RX(ϴ) ├»
     └───┘└───────┘└───┘└────────┘└─────────┘└─────────┘└─────────┘└───────┘»
«     ┌──────────┐
«q_0: ┤0         ├─────────────────────────────────
«     │  RZX(-ϴ) │┌─────────┐┌─────────┐┌─────────┐
«q_1: ┤1         ├┤ RZ(π/2) ├┤ RX(π/2) ├┤ RZ(π/2) ├
«     └──────────┘└─────────┘└─────────┘└─────────┘
"""
from __future__ import annotations
import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType

def rzx_zz3(theta: ParameterValueType | None=None):
    if False:
        while True:
            i = 10
    'Template for CX - RZGate - CX.'
    if theta is None:
        theta = Parameter('ϴ')
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.rz(theta, 1)
    qc.cx(0, 1)
    qc.rz(-1 * theta, 1)
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)
    qc.rx(theta, 1)
    qc.rzx(-1 * theta, 0, 1)
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)
    return qc