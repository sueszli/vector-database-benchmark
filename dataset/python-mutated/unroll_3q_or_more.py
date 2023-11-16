"""Recursively expands 3q+ gates until the circuit only contains 2q or 1q gates."""
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.exceptions import QiskitError
from qiskit.circuit import ControlFlowOp
from qiskit.converters.circuit_to_dag import circuit_to_dag

class Unroll3qOrMore(TransformationPass):
    """Recursively expands 3q+ gates until the circuit only contains 2q or 1q gates."""

    def __init__(self, target=None, basis_gates=None):
        if False:
            return 10
        'Initialize the Unroll3qOrMore pass\n\n        Args:\n            target (Target): The target object representing the compilation\n                target. If specified any multi-qubit instructions in the\n                circuit when the pass is run that are supported by the target\n                device will be left in place. If both this and ``basis_gates``\n                are specified only the target will be checked.\n            basis_gates (list): A list of basis gate names that the target\n                device supports. If specified any gate names in the circuit\n                which are present in this list will not be unrolled. If both\n                this and ``target`` are specified only the target will be used\n                for checking which gates are supported.\n        '
        super().__init__()
        self.target = target
        self.basis_gates = None
        if basis_gates is not None:
            self.basis_gates = set(basis_gates)

    def run(self, dag):
        if False:
            return 10
        'Run the Unroll3qOrMore pass on `dag`.\n\n        Args:\n            dag(DAGCircuit): input dag\n        Returns:\n            DAGCircuit: output dag with maximum node degrees of 2\n        Raises:\n            QiskitError: if a 3q+ gate is not decomposable\n        '
        for node in dag.multi_qubit_ops():
            if dag.has_calibration_for(node):
                continue
            if isinstance(node.op, ControlFlowOp):
                node.op = control_flow.map_blocks(self.run, node.op)
                continue
            if self.target is not None:
                if node.name in self.target:
                    continue
            elif self.basis_gates is not None and node.name in self.basis_gates:
                continue
            rule = node.op.definition.data
            if not rule:
                if rule == []:
                    dag.remove_op_node(node)
                    continue
                raise QiskitError('Cannot unroll all 3q or more gates. No rule to expand instruction %s.' % node.op.name)
            decomposition = circuit_to_dag(node.op.definition, copy_operations=False)
            decomposition = self.run(decomposition)
            dag.substitute_node_with_dag(node, decomposition)
        return dag