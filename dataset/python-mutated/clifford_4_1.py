"""
Clifford template 4_1:
.. parsed-literal::

                  ┌───┐
        q_0: ──■──┤ X ├──■───X─
             ┌─┴─┐└─┬─┘┌─┴─┐ │
        q_1: ┤ X ├──■──┤ X ├─X─
             └───┘     └───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_4_1():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cx(0, 1)
    qc.swap(0, 1)
    return qc