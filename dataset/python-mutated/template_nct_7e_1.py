"""
Template 7e_1:
.. parsed-literal::
         ┌───┐                    ┌───┐
    q_0: ┤ X ├──■─────────■────■──┤ X ├──■──
         └───┘┌─┴─┐       │  ┌─┴─┐└───┘  │
    q_1: ─────┤ X ├───────┼──┤ X ├───────┼──
              └─┬─┘┌───┐┌─┴─┐└─┬─┘     ┌─┴─┐
    q_2: ───────■──┤ X ├┤ X ├──■───────┤ X ├
                   └───┘└───┘          └───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_7e_1():
    if False:
        while True:
            i = 10
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.ccx(0, 2, 1)
    qc.x(2)
    qc.cx(0, 2)
    qc.ccx(0, 2, 1)
    qc.x(0)
    qc.cx(0, 2)
    return qc