"""
Template 5a_2:
.. parsed-literal::
    q_0: ──■─────────■─────────■──
           │  ┌───┐  │  ┌───┐  │
    q_1: ──■──┤ X ├──■──┤ X ├──┼──
         ┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐
    q_2: ┤ X ├─────┤ X ├─────┤ X ├
         └───┘     └───┘     └───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_5a_2():
    if False:
        print('Hello World!')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.x(1)
    qc.ccx(0, 1, 2)
    qc.x(1)
    qc.cx(0, 2)
    return qc