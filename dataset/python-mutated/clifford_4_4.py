"""
Clifford template 4_4:
.. parsed-literal::

             ┌───┐   ┌─────┐
        q_0: ┤ S ├─■─┤ SDG ├─■─
             └───┘ │ └─────┘ │
        q_1: ──────■─────────■─
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_4_4():
    if False:
        i = 10
        return i + 15
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.s(0)
    qc.cz(0, 1)
    qc.sdg(0)
    qc.cz(0, 1)
    return qc