"""Cancel the redundant (self-adjoint) gates through commutation relations."""
from collections import defaultdict
import numpy as np
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes.optimization.commutation_analysis import CommutationAnalysis
from qiskit.dagcircuit import DAGCircuit, DAGInNode, DAGOutNode
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.library.standard_gates.rx import RXGate
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit import ControlFlowOp
_CUTOFF_PRECISION = 1e-05

class CommutativeCancellation(TransformationPass):
    """Cancel the redundant (self-adjoint) gates through commutation relations.

    Pass for cancelling self-inverse gates/rotations. The cancellation utilizes
    the commutation relations in the circuit. Gates considered include::

        H, X, Y, Z, CX, CY, CZ
    """

    def __init__(self, basis_gates=None, target=None):
        if False:
            while True:
                i = 10
        "\n        CommutativeCancellation initializer.\n\n        Args:\n            basis_gates (list[str]): Basis gates to consider, e.g.\n                ``['u3', 'cx']``. For the effects of this pass, the basis is\n                the set intersection between the ``basis_gates`` parameter\n                and the gates in the dag.\n            target (Target): The :class:`~.Target` representing the target backend, if both\n                ``basis_gates`` and ``target`` are specified then this argument will take\n                precedence and ``basis_gates`` will be ignored.\n        "
        super().__init__()
        if basis_gates:
            self.basis = set(basis_gates)
        else:
            self.basis = set()
        if target is not None:
            self.basis = set(target.operation_names)
        self._var_z_map = {'rz': RZGate, 'p': PhaseGate, 'u1': U1Gate}
        self.requires.append(CommutationAnalysis())

    def run(self, dag):
        if False:
            print('Hello World!')
        'Run the CommutativeCancellation pass on `dag`.\n\n        Args:\n            dag (DAGCircuit): the DAG to be optimized.\n\n        Returns:\n            DAGCircuit: the optimized DAG.\n\n        Raises:\n            TranspilerError: when the 1-qubit rotation gates are not found\n        '
        var_z_gate = None
        z_var_gates = [gate for gate in dag.count_ops().keys() if gate in self._var_z_map]
        if z_var_gates:
            var_z_gate = self._var_z_map[next(iter(z_var_gates))]
        else:
            z_var_gates = [gate for gate in self.basis if gate in self._var_z_map]
            if z_var_gates:
                var_z_gate = self._var_z_map[next(iter(z_var_gates))]
        q_gate_list = ['cx', 'cy', 'cz', 'h', 'y']
        cancellation_sets = defaultdict(lambda : [])
        for wire in dag.wires:
            wire_commutation_set = self.property_set['commutation_set'][wire]
            for (com_set_idx, com_set) in enumerate(wire_commutation_set):
                if isinstance(com_set[0], (DAGInNode, DAGOutNode)):
                    continue
                for node in com_set:
                    num_qargs = len(node.qargs)
                    if num_qargs == 1 and node.name in q_gate_list:
                        cancellation_sets[node.name, wire, com_set_idx].append(node)
                    if num_qargs == 1 and node.name in ['p', 'z', 'u1', 'rz', 't', 's']:
                        cancellation_sets['z_rotation', wire, com_set_idx].append(node)
                    if num_qargs == 1 and node.name in ['rx', 'x']:
                        cancellation_sets['x_rotation', wire, com_set_idx].append(node)
                    elif num_qargs == 2 and node.qargs[0] == wire:
                        second_qarg = node.qargs[1]
                        q2_key = (node.name, wire, second_qarg, com_set_idx, self.property_set['commutation_set'][node, second_qarg])
                        cancellation_sets[q2_key].append(node)
        for cancel_set_key in cancellation_sets:
            if cancel_set_key[0] == 'z_rotation' and var_z_gate is None:
                continue
            set_len = len(cancellation_sets[cancel_set_key])
            if set_len > 1 and cancel_set_key[0] in q_gate_list:
                gates_to_cancel = cancellation_sets[cancel_set_key]
                for c_node in gates_to_cancel[:set_len // 2 * 2]:
                    dag.remove_op_node(c_node)
            elif set_len > 1 and cancel_set_key[0] in ['z_rotation', 'x_rotation']:
                run = cancellation_sets[cancel_set_key]
                run_qarg = run[0].qargs[0]
                total_angle = 0.0
                total_phase = 0.0
                for current_node in run:
                    if getattr(current_node.op, 'condition', None) is not None or len(current_node.qargs) != 1 or current_node.qargs[0] != run_qarg:
                        raise TranspilerError('internal error')
                    if current_node.name in ['p', 'u1', 'rz', 'rx']:
                        current_angle = float(current_node.op.params[0])
                    elif current_node.name in ['z', 'x']:
                        current_angle = np.pi
                    elif current_node.name == 't':
                        current_angle = np.pi / 4
                    elif current_node.name == 's':
                        current_angle = np.pi / 2
                    total_angle = current_angle + total_angle
                    if current_node.op.definition:
                        total_phase += current_node.op.definition.global_phase
                if cancel_set_key[0] == 'z_rotation':
                    new_op = var_z_gate(total_angle)
                elif cancel_set_key[0] == 'x_rotation':
                    new_op = RXGate(total_angle)
                new_op_phase = 0
                if np.mod(total_angle, 2 * np.pi) > _CUTOFF_PRECISION:
                    new_qarg = QuantumRegister(1, 'q')
                    new_dag = DAGCircuit()
                    new_dag.add_qreg(new_qarg)
                    new_dag.apply_operation_back(new_op, [new_qarg[0]])
                    dag.substitute_node_with_dag(run[0], new_dag)
                    if new_op.definition:
                        new_op_phase = new_op.definition.global_phase
                dag.global_phase = total_phase - new_op_phase
                for current_node in run[1:]:
                    dag.remove_op_node(current_node)
                if np.mod(total_angle, 2 * np.pi) < _CUTOFF_PRECISION:
                    dag.remove_op_node(run[0])
        dag = self._handle_control_flow_ops(dag)
        return dag

    def _handle_control_flow_ops(self, dag):
        if False:
            i = 10
            return i + 15
        '\n        This is similar to transpiler/passes/utils/control_flow.py except that the\n        commutation analysis is redone for the control flow blocks.\n        '
        pass_manager = PassManager([CommutationAnalysis(), self])
        for node in dag.op_nodes(ControlFlowOp):
            mapped_blocks = []
            for block in node.op.blocks:
                new_circ = pass_manager.run(block)
                mapped_blocks.append(new_circ)
            node.op = node.op.replace_blocks(mapped_blocks)
        return dag