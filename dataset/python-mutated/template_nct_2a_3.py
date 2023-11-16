"""
Template 2a_3:
.. parsed-literal::
    q_0: ──■────■──
           │    │
    q_1: ──■────■──
         ┌─┴─┐┌─┴─┐
    q_2: ┤ X ├┤ X ├
         └───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_2a_3():
    if False:
        return 10
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.ccx(0, 1, 2)
    return qc