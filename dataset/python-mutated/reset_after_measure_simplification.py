"""Replace resets after measure with a conditional XGate."""
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.circuit.library.standard_gates.x import XGate
from qiskit.circuit.reset import Reset
from qiskit.circuit.measure import Measure
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode

class ResetAfterMeasureSimplification(TransformationPass):
    """This pass replaces reset after measure with a conditional X gate.

    This optimization is suitable for use on IBM Quantum systems where the
    reset operation is performed by a measurement followed by a conditional
    x-gate. It might not be desirable on other backends if reset is implemented
    differently.
    """

    @control_flow.trivial_recurse
    def run(self, dag):
        if False:
            i = 10
            return i + 15
        'Run the pass on a dag.'
        for node in dag.op_nodes(Measure):
            succ = next(dag.quantum_successors(node))
            if isinstance(succ, DAGOpNode) and isinstance(succ.op, Reset):
                new_x = XGate().c_if(node.cargs[0], 1)
                new_dag = DAGCircuit()
                new_dag.add_qubits(node.qargs)
                new_dag.add_clbits(node.cargs)
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
                new_dag.apply_operation_back(new_x, node.qargs)
                dag.remove_op_node(succ)
                dag.substitute_node_with_dag(node, new_dag)
        return dag