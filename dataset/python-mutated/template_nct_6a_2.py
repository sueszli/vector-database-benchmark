"""
Template 6a_2:
.. parsed-literal::
    q_0: ──■────■────■────■────■────■──
           │  ┌─┴─┐  │  ┌─┴─┐  │  ┌─┴─┐
    q_1: ──■──┤ X ├──■──┤ X ├──■──┤ X ├
         ┌─┴─┐└─┬─┘┌─┴─┐└─┬─┘┌─┴─┐└─┬─┘
    q_2: ┤ X ├──■──┤ X ├──■──┤ X ├──■──
         └───┘     └───┘     └───┘
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_nct_6a_2():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns:\n        QuantumCircuit: template as a quantum circuit.\n    '
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.ccx(0, 2, 1)
    qc.ccx(0, 1, 2)
    qc.ccx(0, 2, 1)
    qc.ccx(0, 1, 2)
    qc.ccx(0, 2, 1)
    return qc