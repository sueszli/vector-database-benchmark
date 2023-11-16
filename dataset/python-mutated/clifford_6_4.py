"""
Clifford template 6_4:
.. parsed-literal::

             ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐
        q_0: ┤ S ├┤ H ├┤ S ├┤ H ├┤ S ├┤ H ├
             └───┘└───┘└───┘└───┘└───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_6_4():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(1)
    qc.s(0)
    qc.h(0)
    qc.s(0)
    qc.h(0)
    qc.s(0)
    qc.h(0)
    return qc