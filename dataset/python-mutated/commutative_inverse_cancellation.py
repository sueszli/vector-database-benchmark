"""Cancel pairs of inverse gates exploiting commutation relations."""
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.commutation_checker import CommutationChecker

class CommutativeInverseCancellation(TransformationPass):
    """Cancel pairs of inverse gates exploiting commutation relations."""

    def _skip_node(self, node):
        if False:
            print('Hello World!')
        'Returns True if we should skip this node for the analysis.'
        if not isinstance(node, DAGOpNode):
            return True
        if getattr(node.op, '_directive', False) or node.name in {'measure', 'reset', 'delay'}:
            return True
        if getattr(node.op, 'condition', None):
            return True
        if node.op.is_parameterized():
            return True
        return False

    def run(self, dag: DAGCircuit):
        if False:
            i = 10
            return i + 15
        '\n        Run the CommutativeInverseCancellation pass on `dag`.\n\n        Args:\n            dag: the directed acyclic graph to run on.\n\n        Returns:\n            DAGCircuit: Transformed DAG.\n        '
        topo_sorted_nodes = list(dag.topological_op_nodes())
        circ_size = len(topo_sorted_nodes)
        removed = [False for _ in range(circ_size)]
        cc = CommutationChecker()
        for idx1 in range(0, circ_size):
            if self._skip_node(topo_sorted_nodes[idx1]):
                continue
            matched_idx2 = -1
            for idx2 in range(idx1 - 1, -1, -1):
                if removed[idx2]:
                    continue
                if not self._skip_node(topo_sorted_nodes[idx2]) and topo_sorted_nodes[idx2].qargs == topo_sorted_nodes[idx1].qargs and (topo_sorted_nodes[idx2].cargs == topo_sorted_nodes[idx1].cargs) and (topo_sorted_nodes[idx2].op == topo_sorted_nodes[idx1].op.inverse()):
                    matched_idx2 = idx2
                    break
                if not cc.commute(topo_sorted_nodes[idx1].op, topo_sorted_nodes[idx1].qargs, topo_sorted_nodes[idx1].cargs, topo_sorted_nodes[idx2].op, topo_sorted_nodes[idx2].qargs, topo_sorted_nodes[idx2].cargs):
                    break
            if matched_idx2 != -1:
                removed[idx1] = True
                removed[matched_idx2] = True
        for idx in range(circ_size):
            if removed[idx]:
                dag.remove_op_node(topo_sorted_nodes[idx])
        return dag