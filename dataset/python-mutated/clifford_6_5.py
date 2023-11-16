"""
Clifford template 6_5:
.. parsed-literal::

                      ┌───┐
        q_0: ─■───■───┤ S ├───■───────
              │ ┌─┴─┐┌┴───┴┐┌─┴─┐┌───┐
        q_1: ─■─┤ X ├┤ SDG ├┤ X ├┤ S ├
                └───┘└─────┘└───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_6_5():
    if False:
        i = 10
        return i + 15
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.cx(0, 1)
    qc.s(0)
    qc.sdg(1)
    qc.cx(0, 1)
    qc.s(1)
    return qc