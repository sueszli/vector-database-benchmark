"""
Clifford template 8_2:
.. parsed-literal::

                             ┌───┐
        q_0: ──■─────────■───┤ S ├───■────────────
             ┌─┴─┐┌───┐┌─┴─┐┌┴───┴┐┌─┴─┐┌───┐┌───┐
        q_1: ┤ X ├┤ H ├┤ X ├┤ SDG ├┤ X ├┤ S ├┤ H ├
             └───┘└───┘└───┘└─────┘└───┘└───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_8_2():
    if False:
        return 10
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.h(1)
    qc.cx(0, 1)
    qc.s(0)
    qc.sdg(1)
    qc.cx(0, 1)
    qc.s(1)
    qc.h(1)
    return qc