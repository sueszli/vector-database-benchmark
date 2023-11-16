"""Remove diagonal gates (including diagonal 2Q gates) before a measurement."""
from qiskit.circuit import Measure
from qiskit.circuit.library.standard_gates import RZGate, ZGate, TGate, SGate, TdgGate, SdgGate, U1Gate, CZGate, CRZGate, CU1Gate, RZZGate
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow

class RemoveDiagonalGatesBeforeMeasure(TransformationPass):
    """Remove diagonal gates (including diagonal 2Q gates) before a measurement.

    Transpiler pass to remove diagonal gates (like RZ, T, Z, etc) before
    a measurement. Including diagonal 2Q gates.
    """

    @control_flow.trivial_recurse
    def run(self, dag):
        if False:
            while True:
                i = 10
        'Run the RemoveDiagonalGatesBeforeMeasure pass on `dag`.\n\n        Args:\n            dag (DAGCircuit): the DAG to be optimized.\n\n        Returns:\n            DAGCircuit: the optimized DAG.\n        '
        diagonal_1q_gates = (RZGate, ZGate, TGate, SGate, TdgGate, SdgGate, U1Gate)
        diagonal_2q_gates = (CZGate, CRZGate, CU1Gate, RZZGate)
        nodes_to_remove = set()
        for measure in dag.op_nodes(Measure):
            predecessor = next(dag.quantum_predecessors(measure))
            if isinstance(predecessor, DAGOpNode) and isinstance(predecessor.op, diagonal_1q_gates):
                nodes_to_remove.add(predecessor)
            if isinstance(predecessor, DAGOpNode) and isinstance(predecessor.op, diagonal_2q_gates):
                successors = dag.quantum_successors(predecessor)
                if all((isinstance(s, DAGOpNode) and isinstance(s.op, Measure) for s in successors)):
                    nodes_to_remove.add(predecessor)
        for node_to_remove in nodes_to_remove:
            dag.remove_op_node(node_to_remove)
        return dag