"""
Template 2a_1:
.. parsed-literal::
             ┌───┐┌───┐
        q_0: ┤ X ├┤ X ├
             └───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_2a_1():
    if False:
        while True:
            i = 10
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.x(0)
    return qc