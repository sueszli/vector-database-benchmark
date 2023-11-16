"""
Template 7a_1:
.. parsed-literal::
         ┌───┐                    ┌───┐
    q_0: ┤ X ├──■─────────■────■──┤ X ├──■──
         └─┬─┘┌─┴─┐       │  ┌─┴─┐└─┬─┘  │
    q_1: ──■──┤ X ├──■────■──┤ X ├──■────■──
              └───┘┌─┴─┐┌─┴─┐└───┘     ┌─┴─┐
    q_2: ──────────┤ X ├┤ X ├──────────┤ X ├
                   └───┘└───┘          └───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_7a_1():
    if False:
        print('Hello World!')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(3)
    qc.cx(1, 0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.ccx(0, 1, 2)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.ccx(0, 1, 2)
    return qc