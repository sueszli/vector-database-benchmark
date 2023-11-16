"""
Template 7c_1:
.. parsed-literal::
         ┌───┐                    ┌───┐
    q_0: ┤ X ├──■─────────■────■──┤ X ├──■──
         └───┘┌─┴─┐       │  ┌─┴─┐└───┘  │
    q_1: ─────┤ X ├──■────■──┤ X ├───────■──
              └─┬─┘┌─┴─┐┌─┴─┐└─┬─┘     ┌─┴─┐
    q_2: ───────■──┤ X ├┤ X ├──■───────┤ X ├
                   └───┘└───┘          └───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_7c_1():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.ccx(0, 2, 1)
    qc.cx(1, 2)
    qc.ccx(0, 1, 2)
    qc.ccx(0, 2, 1)
    qc.x(0)
    qc.ccx(0, 1, 2)
    return qc