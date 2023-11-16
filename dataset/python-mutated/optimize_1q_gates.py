"""Optimize chains of single-qubit u1, u2, u3 gates by combining them into a single gate."""
from itertools import groupby
import numpy as np
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.u import UGate
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.library.standard_gates.u2 import U2Gate
from qiskit.circuit.library.standard_gates.u3 import U3Gate
from qiskit.circuit import ParameterExpression
from qiskit.circuit.gate import Gate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info.synthesis import Quaternion
from qiskit._accelerate.optimize_1q_gates import compose_u3_rust
_CHOP_THRESHOLD = 1e-15

class Optimize1qGates(TransformationPass):
    """Optimize chains of single-qubit u1, u2, u3 gates by combining them into a single gate."""

    def __init__(self, basis=None, eps=1e-15, target=None):
        if False:
            i = 10
            return i + 15
        "Optimize1qGates initializer.\n\n        Args:\n            basis (list[str]): Basis gates to consider, e.g. `['u3', 'cx']`. For the effects\n                of this pass, the basis is the set intersection between the `basis` parameter and\n                the set `{'u1','u2','u3', 'u', 'p'}`.\n            eps (float): EPS to check against\n            target (Target): The :class:`~.Target` representing the target backend, if both\n                ``basis`` and ``target`` are specified then this argument will take\n                precedence and ``basis`` will be ignored.\n        "
        super().__init__()
        self.basis = set(basis) if basis else {'u1', 'u2', 'u3'}
        self.eps = eps
        self.target = target

    def run(self, dag):
        if False:
            print('Hello World!')
        'Run the Optimize1qGates pass on `dag`.\n\n        Args:\n            dag (DAGCircuit): the DAG to be optimized.\n\n        Returns:\n            DAGCircuit: the optimized DAG.\n\n        Raises:\n            TranspilerError: if ``YZY`` and ``ZYZ`` angles do not give same rotation matrix.\n        '
        use_u = 'u' in self.basis
        use_p = 'p' in self.basis
        runs = dag.collect_runs(['u1', 'u2', 'u3', 'u', 'p'])
        runs = _split_runs_on_parameters(runs)
        for run in runs:
            run_qubits = None
            if self.target is not None:
                run_qubits = tuple((dag.find_bit(x).index for x in run[0].qargs))
                if self.target.instruction_supported('p', run_qubits):
                    right_name = 'p'
                else:
                    right_name = 'u1'
            elif use_p:
                right_name = 'p'
            else:
                right_name = 'u1'
            right_parameters = (0, 0, 0)
            right_global_phase = 0
            for current_node in run:
                left_name = current_node.name
                if getattr(current_node.op, 'condition', None) is not None or len(current_node.qargs) != 1 or left_name not in ['p', 'u1', 'u2', 'u3', 'u', 'id']:
                    raise TranspilerError('internal error')
                if left_name in ('u1', 'p'):
                    left_parameters = (0, 0, current_node.op.params[0])
                elif left_name == 'u2':
                    left_parameters = (np.pi / 2, current_node.op.params[0], current_node.op.params[1])
                elif left_name in ('u3', 'u'):
                    left_parameters = tuple(current_node.op.params)
                else:
                    if use_p:
                        left_name = 'p'
                    else:
                        left_name = 'u1'
                    left_parameters = (0, 0, 0)
                if current_node.op.definition is not None and current_node.op.definition.global_phase:
                    right_global_phase += current_node.op.definition.global_phase
                try:
                    left_parameters = tuple((float(x) for x in left_parameters))
                except TypeError:
                    pass
                name_tuple = (left_name, right_name)
                if name_tuple in (('u1', 'u1'), ('p', 'p')):
                    right_parameters = (0, 0, right_parameters[2] + left_parameters[2])
                elif name_tuple in (('u1', 'u2'), ('p', 'u2')):
                    right_parameters = (np.pi / 2, right_parameters[1] + left_parameters[2], right_parameters[2])
                elif name_tuple in (('u2', 'u1'), ('u2', 'p')):
                    right_name = 'u2'
                    right_parameters = (np.pi / 2, left_parameters[1], right_parameters[2] + left_parameters[2])
                elif name_tuple in (('u1', 'u3'), ('u1', 'u'), ('p', 'u3'), ('p', 'u')):
                    right_parameters = (right_parameters[0], right_parameters[1] + left_parameters[2], right_parameters[2])
                elif name_tuple in (('u3', 'u1'), ('u', 'u1'), ('u3', 'p'), ('u', 'p')):
                    if use_u:
                        right_name = 'u'
                    else:
                        right_name = 'u3'
                    right_parameters = (left_parameters[0], left_parameters[1], right_parameters[2] + left_parameters[2])
                elif name_tuple == ('u2', 'u2'):
                    if use_u:
                        right_name = 'u'
                    else:
                        right_name = 'u3'
                    right_parameters = (np.pi - left_parameters[2] - right_parameters[1], left_parameters[1] + np.pi / 2, right_parameters[2] + np.pi / 2)
                elif name_tuple[1] == 'nop':
                    right_name = left_name
                    right_parameters = left_parameters
                else:
                    if use_u:
                        right_name = 'u'
                    else:
                        right_name = 'u3'
                    right_parameters = Optimize1qGates.compose_u3(left_parameters[0], left_parameters[1], left_parameters[2], right_parameters[0], right_parameters[1], right_parameters[2])
                if not isinstance(right_parameters[0], ParameterExpression) and abs(np.mod(right_parameters[0], 2 * np.pi)) < self.eps and (right_name != 'u1') and (right_name != 'p'):
                    if use_p:
                        right_name = 'p'
                    else:
                        right_name = 'u1'
                    right_parameters = (0, 0, right_parameters[1] + right_parameters[2] + right_parameters[0])
                if right_name in ('u3', 'u'):
                    if not isinstance(right_parameters[0], ParameterExpression):
                        right_angle = right_parameters[0] - np.pi / 2
                        if abs(right_angle) < self.eps:
                            right_angle = 0
                        if abs(np.mod(right_angle, 2 * np.pi)) < self.eps:
                            right_name = 'u2'
                            right_parameters = (np.pi / 2, right_parameters[1], right_parameters[2] + (right_parameters[0] - np.pi / 2))
                        right_angle = right_parameters[0] + np.pi / 2
                        if abs(right_angle) < self.eps:
                            right_angle = 0
                        if abs(np.mod(right_angle, 2 * np.pi)) < self.eps:
                            right_name = 'u2'
                            right_parameters = (np.pi / 2, right_parameters[1] + np.pi, right_parameters[2] - np.pi + (right_parameters[0] + np.pi / 2))
                if not isinstance(right_parameters[2], ParameterExpression) and right_name in ('u1', 'p') and (abs(np.mod(right_parameters[2], 2 * np.pi)) < self.eps):
                    right_name = 'nop'
            if self.target is not None:
                if right_name == 'u2' and (not self.target.instruction_supported('u2', run_qubits)):
                    if self.target.instruction_supported('u', run_qubits):
                        right_name = 'u'
                    else:
                        right_name = 'u3'
                if right_name in ('u1', 'p') and (not self.target.instruction_supported(right_name, run_qubits)):
                    if self.target.instruction_supported('u', run_qubits):
                        right_name = 'u'
                    else:
                        right_name = 'u3'
            else:
                if right_name == 'u2' and 'u2' not in self.basis:
                    if use_u:
                        right_name = 'u'
                    else:
                        right_name = 'u3'
                if right_name in ('u1', 'p') and right_name not in self.basis:
                    if use_u:
                        right_name = 'u'
                    else:
                        right_name = 'u3'
            new_op = Gate(name='', num_qubits=1, params=[])
            if right_name == 'u1':
                new_op = U1Gate(right_parameters[2])
            if right_name == 'p':
                new_op = PhaseGate(right_parameters[2])
            if right_name == 'u2':
                new_op = U2Gate(right_parameters[1], right_parameters[2])
            if right_name == 'u':
                if 'u' in self.basis:
                    new_op = UGate(*right_parameters)
            if right_name == 'u3':
                if 'u3' in self.basis:
                    new_op = U3Gate(*right_parameters)
                else:
                    raise TranspilerError('It was not possible to use the basis %s' % self.basis)
            dag.global_phase += right_global_phase
            if right_name != 'nop':
                dag.substitute_node(run[0], new_op, inplace=True)
            for current_node in run[1:]:
                dag.remove_op_node(current_node)
            if right_name == 'nop':
                dag.remove_op_node(run[0])
        return dag

    @staticmethod
    def compose_u3(theta1, phi1, lambda1, theta2, phi2, lambda2):
        if False:
            for i in range(10):
                print('nop')
        "Return a triple theta, phi, lambda for the product.\n\n        u3(theta, phi, lambda)\n           = u3(theta1, phi1, lambda1).u3(theta2, phi2, lambda2)\n           = Rz(phi1).Ry(theta1).Rz(lambda1+phi2).Ry(theta2).Rz(lambda2)\n           = Rz(phi1).Rz(phi').Ry(theta').Rz(lambda').Rz(lambda2)\n           = u3(theta', phi1 + phi', lambda2 + lambda')\n\n        Return theta, phi, lambda.\n        "
        (theta, phi, lamb) = compose_u3_rust(theta1, phi1, lambda1, theta2, phi2, lambda2)
        return (theta, phi, lamb)

    @staticmethod
    def yzy_to_zyz(xi, theta1, theta2, eps=1e-09):
        if False:
            i = 10
            return i + 15
        'Express a Y.Z.Y single qubit gate as a Z.Y.Z gate.\n\n        Solve the equation\n\n        .. math::\n\n        Ry(theta1).Rz(xi).Ry(theta2) = Rz(phi).Ry(theta).Rz(lambda)\n\n        for theta, phi, and lambda.\n\n        Return a solution theta, phi, and lambda.\n        '
        quaternion_yzy = Quaternion.from_euler([theta1, xi, theta2], 'yzy')
        euler = quaternion_yzy.to_zyz()
        quaternion_zyz = Quaternion.from_euler(euler, 'zyz')
        out_angles = (euler[1], euler[0], euler[2])
        abs_inner = abs(quaternion_zyz.data.dot(quaternion_yzy.data))
        if not np.allclose(abs_inner, 1, eps):
            raise TranspilerError('YZY and ZYZ angles do not give same rotation matrix.')
        out_angles = tuple((0 if np.abs(angle) < _CHOP_THRESHOLD else angle for angle in out_angles))
        return out_angles

def _split_runs_on_parameters(runs):
    if False:
        print('Hello World!')
    'Finds runs containing parameterized gates and splits them into sequential\n    runs excluding the parameterized gates.\n    '
    out = []
    for run in runs:
        groups = groupby(run, lambda x: x.op.is_parameterized() and x.op.name in ('u3', 'u'))
        for (group_is_parameterized, gates) in groups:
            if not group_is_parameterized:
                out.append(list(gates))
    return out