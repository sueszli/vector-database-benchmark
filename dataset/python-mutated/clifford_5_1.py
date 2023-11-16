"""
Clifford template 5_1:
.. parsed-literal::

        q_0: ──■─────────■─────────■──
             ┌─┴─┐     ┌─┴─┐       │
        q_1: ┤ X ├──■──┤ X ├──■────┼──
             └───┘┌─┴─┐└───┘┌─┴─┐┌─┴─┐
        q_2: ─────┤ X ├─────┤ X ├┤ X ├
                  └───┘     └───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_5_1():
    if False:
        i = 10
        return i + 15
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 2)
    return qc