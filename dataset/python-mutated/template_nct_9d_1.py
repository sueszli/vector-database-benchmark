"""
Template 9d_1:
.. parsed-literal::
                   ┌───┐          ┌───┐          ┌───┐
    q_0: ──■───────┤ X ├───────■──┤ X ├───────■──┤ X ├
         ┌─┴─┐┌───┐└─┬─┘┌───┐┌─┴─┐└─┬─┘┌───┐┌─┴─┐└─┬─┘
    q_1: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
         └───┘└───┘     └───┘└───┘     └───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_9d_1():
    if False:
        i = 10
        return i + 15
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.x(1)
    qc.cx(1, 0)
    qc.x(1)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.x(1)
    qc.cx(0, 1)
    qc.cx(1, 0)
    return qc