"""
Clifford template 6_2:
.. parsed-literal::

             ┌───┐
        q_0: ┤ S ├──■───────────■───■─
             ├───┤┌─┴─┐┌─────┐┌─┴─┐ │
        q_1: ┤ S ├┤ X ├┤ SDG ├┤ X ├─■─
             └───┘└───┘└─────┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_6_2():
    if False:
        print('Hello World!')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.s(0)
    qc.s(1)
    qc.cx(0, 1)
    qc.sdg(1)
    qc.cx(0, 1)
    qc.cz(0, 1)
    return qc