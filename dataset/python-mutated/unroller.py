"""Unroll a circuit to a given basis."""
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.exceptions import QiskitError
from qiskit.circuit import ControlledGate, ControlFlowOp
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.utils.deprecation import deprecate_func

class Unroller(TransformationPass):
    """Unroll a circuit to a given basis.

    Unroll (expand) non-basis, non-opaque instructions recursively
    to a desired basis, using decomposition rules defined for each instruction.
    """

    @deprecate_func(since='0.45.0', additional_msg='This has been replaced by the `BasisTranslator` pass and is going to be removed in Qiskit 1.0.')
    def __init__(self, basis=None, target=None):
        if False:
            print('Hello World!')
        "Unroller initializer.\n\n        Args:\n            basis (list[str] or None): Target basis names to unroll to, e.g. `['u3', 'cx']` . If\n                None, does not unroll any gate.\n            target (Target):  The :class:`~.Target` representing the target backend, if both\n                ``basis`` and this are specified then this argument will take\n                precedence and ``basis`` will be ignored.\n        "
        super().__init__()
        self.basis = basis
        self.target = target

    def run(self, dag):
        if False:
            while True:
                i = 10
        'Run the Unroller pass on `dag`.\n\n        Args:\n            dag (DAGCircuit): input dag\n\n        Raises:\n            QiskitError: if unable to unroll given the basis due to undefined\n            decomposition rules (such as a bad basis) or excessive recursion.\n\n        Returns:\n            DAGCircuit: output unrolled dag\n        '
        if self.basis is None and self.target is None:
            return dag
        basic_insts = ['measure', 'reset', 'barrier', 'snapshot', 'delay']
        for node in dag.op_nodes():
            if getattr(node.op, '_directive', False):
                continue
            run_qubits = None
            if self.target is not None:
                run_qubits = tuple((dag.find_bit(x).index for x in node.qargs))
                if self.target.instruction_supported(node.op.name, qargs=run_qubits) or node.op.name == 'barrier':
                    if isinstance(node.op, ControlledGate) and node.op._open_ctrl:
                        pass
                    else:
                        continue
            else:
                if node.name in basic_insts:
                    continue
                if node.name in self.basis:
                    if isinstance(node.op, ControlledGate) and node.op._open_ctrl:
                        pass
                    else:
                        continue
            if isinstance(node.op, ControlFlowOp):
                node.op = control_flow.map_blocks(self.run, node.op)
                continue
            try:
                phase = node.op.definition.global_phase
                rule = node.op.definition.data
            except (TypeError, AttributeError) as err:
                raise QiskitError(f"Error decomposing node of instruction '{node.name}': {err}. Unable to define instruction '{node.name}' in the given basis.") from err
            while rule and len(rule) == 1 and (len(node.qargs) == len(rule[0].qubits) == 1):
                if self.target is not None:
                    if self.target.instruction_supported(rule[0].operation.name, run_qubits):
                        dag.global_phase += phase
                        dag.substitute_node(node, rule[0].operation, inplace=True)
                        break
                elif rule[0].operation.name in self.basis:
                    dag.global_phase += phase
                    dag.substitute_node(node, rule[0].operation, inplace=True)
                    break
                try:
                    phase += rule[0].operation.definition.global_phase
                    rule = rule[0].operation.definition.data
                except (TypeError, AttributeError) as err:
                    raise QiskitError(f"Error decomposing node of instruction '{node.name}': {err}. Unable to define instruction '{rule[0].operation.name}' in the basis.") from err
            else:
                if not rule:
                    if rule == []:
                        dag.remove_op_node(node)
                        dag.global_phase += phase
                        continue
                    raise QiskitError('Cannot unroll the circuit to the given basis, %s. No rule to expand instruction %s.' % (str(self.basis), node.op.name))
                decomposition = circuit_to_dag(node.op.definition)
                unrolled_dag = self.run(decomposition)
                dag.substitute_node_with_dag(node, unrolled_dag)
        return dag