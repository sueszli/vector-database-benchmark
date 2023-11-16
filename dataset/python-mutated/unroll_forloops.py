""" UnrollForLoops transpilation pass """
from qiskit.circuit import ForLoopOp, ContinueLoopOp, BreakLoopOp, IfElseOp
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.converters import circuit_to_dag

class UnrollForLoops(TransformationPass):
    """``UnrollForLoops`` transpilation pass unrolls for-loops when possible."""

    def __init__(self, max_target_depth=-1):
        if False:
            return 10
        'Things like ``for x in {0, 3, 4} {rx(x) qr[1];}`` will turn into\n        ``rx(0) qr[1]; rx(3) qr[1]; rx(4) qr[1];``.\n\n        .. note::\n            The ``UnrollForLoops`` unrolls only one level of block depth. No inner loop will\n            be considered by ``max_target_depth``.\n\n        Args:\n            max_target_depth (int): Optional. Checks if the unrolled block is over a particular\n                subcircuit depth. To disable the check, use ``-1`` (Default).\n        '
        super().__init__()
        self.max_target_depth = max_target_depth

    @control_flow.trivial_recurse
    def run(self, dag):
        if False:
            for i in range(10):
                print('nop')
        'Run the UnrollForLoops pass on ``dag``.\n\n        Args:\n            dag (DAGCircuit): the directed acyclic graph to run on.\n\n        Returns:\n            DAGCircuit: Transformed DAG.\n        '
        for forloop_op in dag.op_nodes(ForLoopOp):
            (indexset, loop_param, body) = forloop_op.op.params
            if 0 < self.max_target_depth < len(indexset) * body.depth():
                continue
            if _body_contains_continue_or_break(body):
                continue
            unrolled_dag = circuit_to_dag(body).copy_empty_like()
            for index_value in indexset:
                bound_body = body.assign_parameters({loop_param: index_value}) if loop_param else body
                unrolled_dag.compose(circuit_to_dag(bound_body), inplace=True)
            dag.substitute_node_with_dag(forloop_op, unrolled_dag)
        return dag

def _body_contains_continue_or_break(circuit):
    if False:
        while True:
            i = 10
    'Checks if a circuit contains ``continue``s or ``break``s. Conditional bodies are inspected.'
    for inst in circuit.data:
        operation = inst.operation
        if isinstance(operation, (ContinueLoopOp, BreakLoopOp)):
            return True
        if isinstance(operation, IfElseOp):
            for block in operation.params:
                if _body_contains_continue_or_break(block):
                    return True
    return False