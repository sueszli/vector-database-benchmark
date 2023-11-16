"""Combine consecutive Cliffords over the same qubits."""
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.quantum_info.operators import Clifford

class OptimizeCliffords(TransformationPass):
    """Combine consecutive Cliffords over the same qubits.
    This serves as an example of extra capabilities enabled by storing
    Cliffords natively on the circuit.
    """

    @control_flow.trivial_recurse
    def run(self, dag):
        if False:
            print('Hello World!')
        'Run the OptimizeCliffords pass on `dag`.\n\n        Args:\n            dag (DAGCircuit): the DAG to be optimized.\n\n        Returns:\n            DAGCircuit: the optimized DAG.\n        '
        blocks = []
        prev_node = None
        cur_block = []
        for node in dag.topological_op_nodes():
            if isinstance(node.op, Clifford):
                if prev_node is None:
                    blocks.append(cur_block)
                    cur_block = [node]
                elif prev_node.qargs == node.qargs:
                    cur_block.append(node)
                else:
                    blocks.append(cur_block)
                    cur_block = [node]
                prev_node = node
            else:
                if cur_block:
                    blocks.append(cur_block)
                prev_node = None
                cur_block = []
        if cur_block:
            blocks.append(cur_block)
        for cur_nodes in blocks:
            if len(cur_nodes) <= 1:
                continue
            wire_pos_map = {qb: ix for (ix, qb) in enumerate(cur_nodes[0].qargs)}
            cliff = cur_nodes[0].op
            for (i, node) in enumerate(cur_nodes):
                if i > 0:
                    cliff = Clifford.compose(node.op, cliff, front=True)
            dag.replace_block_with_op(cur_nodes, cliff, wire_pos_map, cycle_check=False)
        return dag