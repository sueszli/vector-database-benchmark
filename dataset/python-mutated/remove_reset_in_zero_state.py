"""Remove reset gate when the qubit is in zero state."""
from qiskit.circuit import Reset
from qiskit.dagcircuit import DAGInNode
from qiskit.transpiler.basepasses import TransformationPass

class RemoveResetInZeroState(TransformationPass):
    """Remove reset gate when the qubit is in zero state."""

    def run(self, dag):
        if False:
            for i in range(10):
                print('nop')
        'Run the RemoveResetInZeroState pass on `dag`.\n\n        Args:\n            dag (DAGCircuit): the DAG to be optimized.\n\n        Returns:\n            DAGCircuit: the optimized DAG.\n        '
        resets = dag.op_nodes(Reset)
        for reset in resets:
            predecessor = next(dag.predecessors(reset))
            if isinstance(predecessor, DAGInNode):
                dag.remove_op_node(reset)
        return dag