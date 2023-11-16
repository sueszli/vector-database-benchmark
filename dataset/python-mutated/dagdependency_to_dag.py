"""Helper function for converting a dag dependency to a dag circuit"""
from qiskit.dagcircuit.dagcircuit import DAGCircuit

def dagdependency_to_dag(dagdependency):
    if False:
        print('Hello World!')
    'Build a ``DAGCircuit`` object from a ``DAGDependency``.\n\n    Args:\n        dag dependency (DAGDependency): the input dag.\n\n    Return:\n        DAGCircuit: the DAG representing the input circuit.\n    '
    dagcircuit = DAGCircuit()
    dagcircuit.name = dagdependency.name
    dagcircuit.metadata = dagdependency.metadata
    dagcircuit.add_qubits(dagdependency.qubits)
    dagcircuit.add_clbits(dagdependency.clbits)
    for register in dagdependency.qregs.values():
        dagcircuit.add_qreg(register)
    for register in dagdependency.cregs.values():
        dagcircuit.add_creg(register)
    for node in dagdependency.topological_nodes():
        inst = node.op.copy()
        dagcircuit.apply_operation_back(inst, node.qargs, node.cargs)
    dagcircuit.global_phase = dagdependency.global_phase
    dagcircuit.calibrations = dagdependency.calibrations
    return dagcircuit