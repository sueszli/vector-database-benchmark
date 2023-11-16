"""
Template 4b_2:
.. parsed-literal::
    q_0: ──■─────────■───────
           │         │
    q_1: ──■────■────■────■──
         ┌─┴─┐┌─┴─┐┌─┴─┐┌─┴─┐
    q_2: ┤ X ├┤ X ├┤ X ├┤ X ├
         └───┘└───┘└───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_4b_2():
    if False:
        print('Hello World!')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.cx(1, 2)
    qc.ccx(0, 1, 2)
    qc.cx(1, 2)
    return qc