"""
Clifford template 3_1:
.. parsed-literal::

             ┌───┐┌───┐┌───┐
        q_0: ┤ S ├┤ S ├┤ Z ├
             └───┘└───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_3_1():
    if False:
        i = 10
        return i + 15
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(1)
    qc.s(0)
    qc.s(0)
    qc.z(0)
    return qc