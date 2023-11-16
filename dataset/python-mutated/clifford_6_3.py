"""
Clifford template 6_3:
.. parsed-literal::

                   ┌───┐     ┌───┐
        q_0: ─X──■─┤ H ├──■──┤ X ├─────
              │  │ └───┘┌─┴─┐└─┬─┘┌───┐
        q_1: ─X──■──────┤ X ├──■──┤ H ├
                        └───┘     └───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def clifford_6_3():
    if False:
        i = 10
        return i + 15
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(2)
    qc.swap(0, 1)
    qc.cz(0, 1)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.h(1)
    return qc