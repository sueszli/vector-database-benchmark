"""
Clifford template 8_1:
.. parsed-literal::

                       ┌───┐ ┌───┐ ┌───┐┌─────┐
        q_0: ──■───────┤ X ├─┤ S ├─┤ X ├┤ SDG ├
             ┌─┴─┐┌───┐└─┬─┘┌┴───┴┐└─┬─┘└┬───┬┘
        q_1: ┤ X ├┤ H ├──■──┤ SDG ├──■───┤ H ├─
             └───┘└───┘     └─────┘      └───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_8_1():
    if False:
        while True:
            i = 10
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.h(1)
    qc.cx(1, 0)
    qc.s(0)
    qc.sdg(1)
    qc.cx(1, 0)
    qc.sdg(0)
    qc.h(1)
    return qc