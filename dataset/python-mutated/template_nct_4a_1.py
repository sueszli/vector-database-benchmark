"""
Template 4a_1:
.. parsed-literal::
    q_0: ───────■─────────■──
                │         │
    q_1: ──■────┼────■────┼──
           │    │    │    │
    q_2: ──■────■────■────■──
           │  ┌─┴─┐  │  ┌─┴─┐
    q_3: ──┼──┤ X ├──┼──┤ X ├
         ┌─┴─┐└───┘┌─┴─┐└───┘
    q_4: ┤ X ├─────┤ X ├─────
         └───┘     └───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_4a_1():
    if False:
        print('Hello World!')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(5)
    qc.ccx(1, 2, 4)
    qc.ccx(0, 2, 3)
    qc.ccx(1, 2, 4)
    qc.ccx(0, 2, 3)
    return qc