"""
Template 5a_4:
.. parsed-literal::
              ┌───┐     ┌───┐
    q_0: ──■──┤ X ├──■──┤ X ├
         ┌─┴─┐└───┘┌─┴─┐├───┤
    q_1: ┤ X ├─────┤ X ├┤ X ├
         └───┘     └───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_5a_4():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.x(0)
    qc.cx(0, 1)
    qc.x(0)
    qc.x(1)
    return qc