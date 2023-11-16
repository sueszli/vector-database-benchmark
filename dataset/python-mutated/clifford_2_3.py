"""
Clifford template 2_3:
.. parsed-literal::
             ┌───┐┌───┐
        q_0: ┤ H ├┤ H ├
             └───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_2_3():
    if False:
        i = 10
        return i + 15
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.h(0)
    return qc