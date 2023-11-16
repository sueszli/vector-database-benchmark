"""Helper function for converting a circuit to a dag dependency"""
from qiskit.dagcircuit.dagdependency import DAGDependency

def circuit_to_dagdependency(circuit, create_preds_and_succs=True):
    if False:
        return 10
    'Build a ``DAGDependency`` object from a :class:`~.QuantumCircuit`.\n\n    Args:\n        circuit (QuantumCircuit): the input circuit.\n        create_preds_and_succs (bool): whether to construct lists of\n            predecessors and successors for every node.\n\n    Return:\n        DAGDependency: the DAG representing the input circuit as a dag dependency.\n    '
    dagdependency = DAGDependency()
    dagdependency.name = circuit.name
    dagdependency.metadata = circuit.metadata
    dagdependency.add_qubits(circuit.qubits)
    dagdependency.add_clbits(circuit.clbits)
    for register in circuit.qregs:
        dagdependency.add_qreg(register)
    for register in circuit.cregs:
        dagdependency.add_creg(register)
    for instruction in circuit.data:
        dagdependency.add_op_node(instruction.operation, instruction.qubits, instruction.clbits)
    if create_preds_and_succs:
        dagdependency._add_predecessors()
        dagdependency._add_successors()
    dagdependency.calibrations = circuit.calibrations
    return dagdependency