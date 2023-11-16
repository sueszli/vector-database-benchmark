"""
Template 4b_1:
.. parsed-literal::
    q_0: ───────■─────────■──
                │         │
    q_1: ──■────┼────■────┼──
           │    │    │    │
    q_2: ──■────■────■────■──
         ┌─┴─┐┌─┴─┐┌─┴─┐┌─┴─┐
    q_3: ┤ X ├┤ X ├┤ X ├┤ X ├
         └───┘└───┘└───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_4b_1():
    if False:
        while True:
            i = 10
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(4)
    qc.ccx(1, 2, 3)
    qc.ccx(0, 2, 3)
    qc.ccx(1, 2, 3)
    qc.ccx(0, 2, 3)
    return qc