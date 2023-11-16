"""
Template 9d_3:
.. parsed-literal::
    q_0: ──■────■───────────────────■─────────────────
           │    │  ┌───┐          ┌─┴─┐          ┌───┐
    q_1: ──■────┼──┤ X ├───────■──┤ X ├───────■──┤ X ├
         ┌─┴─┐┌─┴─┐└─┬─┘┌───┐┌─┴─┐└─┬─┘┌───┐┌─┴─┐└─┬─┘
    q_2: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
         └───┘└───┘     └───┘└───┘     └───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_9d_3():
    if False:
        i = 10
        return i + 15
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.cx(0, 2)
    qc.cx(2, 1)
    qc.x(2)
    qc.cx(1, 2)
    qc.ccx(0, 2, 1)
    qc.x(2)
    qc.cx(1, 2)
    qc.cx(2, 1)
    return qc