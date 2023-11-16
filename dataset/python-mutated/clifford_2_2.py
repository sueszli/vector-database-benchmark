"""
Clifford template 2_2:
.. parsed-literal::

        q_0: ──■────■──
             ┌─┴─┐┌─┴─┐
        q_1: ┤ X ├┤ X ├
             └───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_2_2():
    if False:
        while True:
            i = 10
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(0, 1)
    return qc