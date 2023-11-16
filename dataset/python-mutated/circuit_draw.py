"""
Example showing how to draw a quantum circuit using Qiskit.
"""
from qiskit import QuantumCircuit

def build_bell_circuit():
    if False:
        for i in range(10):
            print('nop')
    'Returns a circuit putting 2 qubits in the Bell state.'
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc
bell_circuit = build_bell_circuit()
print(bell_circuit)