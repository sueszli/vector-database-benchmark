"""
Template 4a_3:
.. parsed-literal::
    q_0: ──■────■────■────■──
           │  ┌─┴─┐  │  ┌─┴─┐
    q_1: ──┼──┤ X ├──┼──┤ X ├
         ┌─┴─┐└───┘┌─┴─┐└───┘
    q_2: ┤ X ├─────┤ X ├─────
         └───┘     └───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_4a_3():
    if False:
        print('Hello World!')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 1)
    return qc