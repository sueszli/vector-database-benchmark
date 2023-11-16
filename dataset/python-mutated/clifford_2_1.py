"""
Clifford template 2_1:
.. parsed-literal::

        q_0: ─■──■─
              │  │
        q_1: ─■──■─
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_2_1():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.cz(0, 1)
    return qc