"""
Clifford template 6_1:
.. parsed-literal::

             ┌───┐     ┌───┐┌───┐
        q_0: ┤ H ├──■──┤ H ├┤ X ├
             ├───┤┌─┴─┐├───┤└─┬─┘
        q_1: ┤ H ├┤ X ├┤ H ├──■──
             └───┘└───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_6_1():
    if False:
        while True:
            i = 10
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)
    qc.h(0)
    qc.h(1)
    qc.cx(1, 0)
    return qc