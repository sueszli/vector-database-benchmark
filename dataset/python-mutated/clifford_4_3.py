"""
Clifford template 4_3:
.. parsed-literal::

             ┌───┐     ┌─────┐
        q_0: ┤ S ├──■──┤ SDG ├──■──
             └───┘┌─┴─┐└─────┘┌─┴─┐
        q_1: ─────┤ X ├───────┤ X ├
                  └───┘       └───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_4_3():
    if False:
        print('Hello World!')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.s(0)
    qc.cx(0, 1)
    qc.sdg(0)
    qc.cx(0, 1)
    return qc