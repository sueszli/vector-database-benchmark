"""
Clifford template 8_3:
.. parsed-literal::

        q_0: ─────────────────■───────────────────────■──
             ┌───┐┌───┐┌───┐┌─┴─┐┌─────┐┌───┐┌─────┐┌─┴─┐
        q_1: ┤ S ├┤ H ├┤ S ├┤ X ├┤ SDG ├┤ H ├┤ SDG ├┤ X ├
             └───┘└───┘└───┘└───┘└─────┘└───┘└─────┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_8_3():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.s(1)
    qc.h(1)
    qc.s(1)
    qc.cx(0, 1)
    qc.sdg(1)
    qc.h(1)
    qc.sdg(1)
    qc.cx(0, 1)
    return qc