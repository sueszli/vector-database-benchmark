"""
Template 2a_2:
.. parsed-literal::
    q_0: ──■────■──
         ┌─┴─┐┌─┴─┐
    q_1: ┤ X ├┤ X ├
         └───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_2a_2():
    if False:
        return 10
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(0, 1)
    return qc