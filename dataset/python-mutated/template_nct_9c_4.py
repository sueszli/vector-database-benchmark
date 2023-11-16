"""
Template 9c_4:
.. parsed-literal::
    q_0: ──■────■─────────■──────────────■────────────
         ┌─┴─┐  │  ┌───┐┌─┴─┐     ┌───┐  │       ┌───┐
    q_1: ┤ X ├──■──┤ X ├┤ X ├─────┤ X ├──■───────┤ X ├
         └─┬─┘┌─┴─┐└───┘└─┬─┘┌───┐└─┬─┘┌─┴─┐┌───┐└─┬─┘
    q_2: ──■──┤ X ├───────■──┤ X ├──■──┤ X ├┤ X ├──■──
              └───┘          └───┘     └───┘└───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_9c_4():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(3)
    qc.ccx(0, 2, 1)
    qc.ccx(0, 1, 2)
    qc.x(1)
    qc.ccx(0, 2, 1)
    qc.x(2)
    qc.cx(2, 1)
    qc.ccx(0, 1, 2)
    qc.x(2)
    qc.cx(2, 1)
    return qc