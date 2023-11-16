"""Check if the gates follow the right direction with respect to the coupling map."""
from qiskit.circuit import ControlFlowOp
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import AnalysisPass

class CheckGateDirection(AnalysisPass):
    """Check if the two-qubit gates follow the right direction with
    respect to the coupling map.
    """

    def __init__(self, coupling_map, target=None):
        if False:
            return 10
        'CheckGateDirection initializer.\n\n        Args:\n            coupling_map (CouplingMap): Directed graph representing a coupling map.\n            target (Target): The backend target to use for this pass. If this is specified\n                it will be used instead of the coupling map\n        '
        super().__init__()
        self.coupling_map = coupling_map
        self.target = target

    def _coupling_map_visit(self, dag, wire_map, edges=None):
        if False:
            return 10
        if edges is None:
            edges = self.coupling_map.get_edges()
        for node in dag.op_nodes(include_directives=False):
            if isinstance(node.op, ControlFlowOp):
                for block in node.op.blocks:
                    inner_wire_map = {inner: wire_map[outer] for (outer, inner) in zip(node.qargs, block.qubits)}
                    if not self._coupling_map_visit(circuit_to_dag(block), inner_wire_map, edges):
                        return False
            elif len(node.qargs) == 2 and (wire_map[node.qargs[0]], wire_map[node.qargs[1]]) not in edges:
                return False
        return True

    def _target_visit(self, dag, wire_map):
        if False:
            return 10
        for node in dag.op_nodes(include_directives=False):
            if isinstance(node.op, ControlFlowOp):
                for block in node.op.blocks:
                    inner_wire_map = {inner: wire_map[outer] for (outer, inner) in zip(node.qargs, block.qubits)}
                    if not self._target_visit(circuit_to_dag(block), inner_wire_map):
                        return False
            elif len(node.qargs) == 2 and (not self.target.instruction_supported(node.op.name, (wire_map[node.qargs[0]], wire_map[node.qargs[1]]))):
                return False
        return True

    def run(self, dag):
        if False:
            i = 10
            return i + 15
        'Run the CheckGateDirection pass on `dag`.\n\n        If `dag` is mapped and the direction is correct the property\n        `is_direction_mapped` is set to True (or to False otherwise).\n\n        Args:\n            dag (DAGCircuit): DAG to check.\n        '
        wire_map = {bit: i for (i, bit) in enumerate(dag.qubits)}
        self.property_set['is_direction_mapped'] = self._coupling_map_visit(dag, wire_map) if self.target is None else self._target_visit(dag, wire_map)