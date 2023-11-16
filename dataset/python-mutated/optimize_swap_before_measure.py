"""Remove the swaps followed by measurement (and adapt the measurement)."""
from qiskit.circuit import Measure
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGOutNode

class OptimizeSwapBeforeMeasure(TransformationPass):
    """Remove the swaps followed by measurement (and adapt the measurement).

    Transpiler pass to remove swaps in front of measurements by re-targeting
    the classical bit of the measure instruction.
    """

    @control_flow.trivial_recurse
    def run(self, dag):
        if False:
            print('Hello World!')
        'Run the OptimizeSwapBeforeMeasure pass on `dag`.\n\n        Args:\n            dag (DAGCircuit): the DAG to be optimized.\n\n        Returns:\n            DAGCircuit: the optimized DAG.\n        '
        swaps = dag.op_nodes(SwapGate)
        for swap in swaps[::-1]:
            if getattr(swap.op, 'condition', None) is not None:
                continue
            final_successor = []
            for successor in dag.successors(swap):
                final_successor.append(isinstance(successor, DAGOutNode) or (isinstance(successor, DAGOpNode) and isinstance(successor.op, Measure)))
            if all(final_successor):
                swap_qargs = swap.qargs
                measure_layer = DAGCircuit()
                for qreg in dag.qregs.values():
                    measure_layer.add_qreg(qreg)
                for creg in dag.cregs.values():
                    measure_layer.add_creg(creg)
                for successor in list(dag.successors(swap)):
                    if isinstance(successor, DAGOpNode) and isinstance(successor.op, Measure):
                        dag.remove_op_node(successor)
                        old_measure_qarg = successor.qargs[0]
                        new_measure_qarg = swap_qargs[swap_qargs.index(old_measure_qarg) - 1]
                        measure_layer.apply_operation_back(Measure(), (new_measure_qarg,), (successor.cargs[0],), check=False)
                dag.compose(measure_layer)
                dag.remove_op_node(swap)
        return dag