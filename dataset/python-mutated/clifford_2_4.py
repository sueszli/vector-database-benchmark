"""
Clifford template 2_4:
.. parsed-literal::

        q_0: ─X──X─
              │  │
        q_1: ─X──X─
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_2_4():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.swap(0, 1)
    qc.swap(1, 0)
    return qc