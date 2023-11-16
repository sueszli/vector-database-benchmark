"""Analysis pass to find commutation relations between DAG nodes."""
from collections import defaultdict
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.circuit.commutation_checker import CommutationChecker

class CommutationAnalysis(AnalysisPass):
    """Analysis pass to find commutation relations between DAG nodes.

    ``property_set['commutation_set']`` is a dictionary that describes
    the commutation relations on a given wire, all the gates on a wire
    are grouped into a set of gates that commute.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.comm_checker = CommutationChecker()

    def run(self, dag):
        if False:
            return 10
        'Run the CommutationAnalysis pass on `dag`.\n\n        Run the pass on the DAG, and write the discovered commutation relations\n        into the ``property_set``.\n        '
        self.property_set['commutation_set'] = defaultdict(list)
        for wire in dag.wires:
            self.property_set['commutation_set'][wire] = []
        for node in dag.topological_op_nodes():
            for (_, _, edge_wire) in dag.edges(node):
                self.property_set['commutation_set'][node, edge_wire] = -1
        for wire in dag.wires:
            for current_gate in dag.nodes_on_wire(wire):
                current_comm_set = self.property_set['commutation_set'][wire]
                if not current_comm_set:
                    current_comm_set.append([current_gate])
                if current_gate not in current_comm_set[-1]:
                    does_commute = True
                    for prev_gate in current_comm_set[-1]:
                        does_commute = isinstance(current_gate, DAGOpNode) and isinstance(prev_gate, DAGOpNode) and self.comm_checker.commute(current_gate.op, current_gate.qargs, current_gate.cargs, prev_gate.op, prev_gate.qargs, prev_gate.cargs)
                        if not does_commute:
                            break
                    if does_commute:
                        current_comm_set[-1].append(current_gate)
                    else:
                        current_comm_set.append([current_gate])
                temp_len = len(current_comm_set)
                self.property_set['commutation_set'][current_gate, wire] = temp_len - 1