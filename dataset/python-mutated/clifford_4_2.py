"""
Clifford template 4_2:
.. parsed-literal::

        q_0: ───────■────────■─
             ┌───┐┌─┴─┐┌───┐ │
        q_1: ┤ H ├┤ X ├┤ H ├─■─
             └───┘└───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_4_2():
    if False:
        while True:
            i = 10
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.h(1)
    qc.cx(0, 1)
    qc.h(1)
    qc.cz(0, 1)
    return qc