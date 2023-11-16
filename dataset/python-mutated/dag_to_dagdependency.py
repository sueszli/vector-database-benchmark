"""Helper function for converting a dag circuit to a dag dependency"""
from qiskit.dagcircuit.dagdependency import DAGDependency

def dag_to_dagdependency(dag, create_preds_and_succs=True):
    if False:
        while True:
            i = 10
    'Build a ``DAGDependency`` object from a ``DAGCircuit``.\n\n    Args:\n        dag (DAGCircuit): the input dag.\n        create_preds_and_succs (bool): whether to construct lists of\n            predecessors and successors for every node.\n\n    Return:\n        DAGDependency: the DAG representing the input circuit as a dag dependency.\n    '
    dagdependency = DAGDependency()
    dagdependency.name = dag.name
    dagdependency.metadata = dag.metadata
    dagdependency.add_qubits(dag.qubits)
    dagdependency.add_clbits(dag.clbits)
    for register in dag.qregs.values():
        dagdependency.add_qreg(register)
    for register in dag.cregs.values():
        dagdependency.add_creg(register)
    for node in dag.topological_op_nodes():
        inst = node.op.copy()
        dagdependency.add_op_node(inst, node.qargs, node.cargs)
    if create_preds_and_succs:
        dagdependency._add_predecessors()
        dagdependency._add_successors()
    dagdependency.global_phase = dag.global_phase
    dagdependency.calibrations = dag.calibrations
    return dagdependency