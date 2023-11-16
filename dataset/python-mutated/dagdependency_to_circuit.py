"""Helper function for converting a dag dependency to a circuit"""
from qiskit.circuit import QuantumCircuit, CircuitInstruction

def dagdependency_to_circuit(dagdependency):
    if False:
        while True:
            i = 10
    'Build a ``QuantumCircuit`` object from a ``DAGDependency``.\n\n    Args:\n        dagdependency (DAGDependency): the input dag.\n\n    Return:\n        QuantumCircuit: the circuit representing the input dag dependency.\n    '
    name = dagdependency.name or None
    circuit = QuantumCircuit(dagdependency.qubits, dagdependency.clbits, *dagdependency.qregs.values(), *dagdependency.cregs.values(), name=name)
    circuit.metadata = dagdependency.metadata
    circuit.calibrations = dagdependency.calibrations
    for node in dagdependency.topological_nodes():
        circuit._append(CircuitInstruction(node.op.copy(), node.qargs, node.cargs))
    return circuit