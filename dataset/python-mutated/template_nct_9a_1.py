"""
Template 9a_1:
.. parsed-literal::
         ┌───┐     ┌───┐          ┌───┐
    q_0: ┤ X ├──■──┤ X ├──■────■──┤ X ├──■──
         └─┬─┘┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐
    q_1: ──■──┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├
              └─┬─┘  │  ├───┤└─┬─┘┌───┐└─┬─┘
    q_2: ───────■────■──┤ X ├──■──┤ X ├──■──
                        └───┘     └───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_9a_1():
    if False:
        i = 10
        return i + 15
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(3)
    qc.cx(1, 0)
    qc.ccx(0, 2, 1)
    qc.ccx(1, 2, 0)
    qc.x(2)
    qc.cx(0, 1)
    qc.ccx(0, 2, 1)
    qc.cx(1, 0)
    qc.x(2)
    qc.ccx(0, 2, 1)
    return qc