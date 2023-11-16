"""
Template 4a_2:
.. parsed-literal::
    q_0: ──■─────────■───────
           │         │
    q_1: ──■────■────■────■──
           │  ┌─┴─┐  │  ┌─┴─┐
    q_2: ──┼──┤ X ├──┼──┤ X ├
         ┌─┴─┐└───┘┌─┴─┐└───┘
    q_3: ┤ X ├─────┤ X ├─────
         └───┘     └───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_4a_2():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 3)
    qc.cx(1, 2)
    qc.ccx(0, 1, 3)
    qc.cx(1, 2)
    return qc