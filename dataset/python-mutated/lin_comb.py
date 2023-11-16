"""The module to compute the state gradient with the linear combination method."""
from collections.abc import Iterable
from copy import deepcopy
from functools import partial
from itertools import product
from typing import List, Optional, Tuple, Union, Callable
import scipy
import numpy as np
from qiskit.circuit import Gate, Instruction
from qiskit.circuit import CircuitInstruction, QuantumCircuit, QuantumRegister, ParameterVector, ParameterExpression, Parameter
from qiskit.circuit.parametertable import ParameterReferences, ParameterTable
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.library import SGate, SdgGate, XGate
from qiskit.circuit.library.standard_gates import CXGate, CYGate, CZGate, IGate, RXGate, RXXGate, RYGate, RYYGate, RZGate, RZXGate, RZZGate, PhaseGate, UGate, ZGate
from qiskit.quantum_info import partial_trace
from qiskit.utils.deprecation import deprecate_func
from ...operator_base import OperatorBase
from ...list_ops.list_op import ListOp
from ...list_ops.composed_op import ComposedOp
from ...list_ops.summed_op import SummedOp
from ...operator_globals import Z, I, Y, One, Zero
from ...primitive_ops.primitive_op import PrimitiveOp
from ...state_fns.state_fn import StateFn
from ...state_fns.circuit_state_fn import CircuitStateFn
from ...state_fns.dict_state_fn import DictStateFn
from ...state_fns.vector_state_fn import VectorStateFn
from ...state_fns.sparse_vector_state_fn import SparseVectorStateFn
from ...exceptions import OpflowError
from .circuit_gradient import CircuitGradient
from ...converters import PauliBasisChange

class LinComb(CircuitGradient):
    """Deprecated: Compute the state gradient d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω respectively the gradients of the
    sampling probabilities of the basis states of
    a state |ψ(ω)〉w.r.t. ω.
    This method employs a linear combination of unitaries,
    see e.g. https://arxiv.org/pdf/1811.11184.pdf
    """
    SUPPORTED_GATES = {'rx', 'ry', 'rz', 'rzx', 'rzz', 'ryy', 'rxx', 'p', 'u', 'controlledgate', 'cx', 'cy', 'cz', 'ccx', 'swap', 'iswap', 't', 's', 'sdg', 'x', 'y', 'z'}

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, aux_meas_op: OperatorBase=Z):
        if False:
            while True:
                i = 10
        '\n        Args:\n            aux_meas_op: The operator that the auxiliary qubit is measured with respect to.\n                For ``aux_meas_op = Z`` we compute 2Re[(dω⟨ψ(ω)|)O(θ)|ψ(ω)〉],\n                for ``aux_meas_op = -Y`` we compute 2Im[(dω⟨ψ(ω)|)O(θ)|ψ(ω)〉], and\n                for ``aux_meas_op = Z - 1j * Y`` we compute 2(dω⟨ψ(ω)|)O(θ)|ψ(ω)〉.\n        Raises:\n            ValueError: If the provided auxiliary measurement operator is not supported.\n        '
        super().__init__()
        if aux_meas_op not in [Z, -Y, Z - 1j * Y]:
            raise ValueError('This auxiliary measurement operator is currently not supported. Please choose either Z, -Y, or Z - 1j * Y. ')
        self._aux_meas_op = aux_meas_op

    def convert(self, operator: OperatorBase, params: Union[ParameterExpression, ParameterVector, List[ParameterExpression], Tuple[ParameterExpression, ParameterExpression], List[Tuple[ParameterExpression, ParameterExpression]]]) -> OperatorBase:
        if False:
            while True:
                i = 10
        'Convert ``operator`` into an operator that represents the gradient w.r.t. ``params``.\n\n        Args:\n            operator: The operator we are taking the gradient of: ⟨ψ(ω)|O(θ)|ψ(ω)〉\n            params: The parameters we are taking the gradient wrt: ω\n                    If a ParameterExpression, ParameterVector or List[ParameterExpression] is given,\n                    then the 1st order derivative of the operator is calculated.\n                    If a Tuple[ParameterExpression, ParameterExpression] or\n                    List[Tuple[ParameterExpression, ParameterExpression]]\n                    is given, then the 2nd order derivative of the operator is calculated.\n        Returns:\n            An operator corresponding to the gradient resp. Hessian. The order is in accordance with\n            the order of the given parameters.\n        '
        return self._prepare_operator(operator, params)

    def _prepare_operator(self, operator: OperatorBase, params: Union[ParameterExpression, ParameterVector, List[ParameterExpression], Tuple[ParameterExpression, ParameterExpression], List[Tuple[ParameterExpression, ParameterExpression]]]) -> OperatorBase:
        if False:
            i = 10
            return i + 15
        'Traverse ``operator`` to get back the adapted operator representing the gradient.\n\n        Args:\n            operator: The operator we are taking the gradient of: ⟨ψ(ω)|O(θ)|ψ(ω)〉.\n            params: The parameters we are taking the gradient wrt: ω.\n                If a ``ParameterExpression```, ``ParameterVector`` or ``List[ParameterExpression]``\n                is given, then the 1st order derivative of the operator is calculated.\n                If a ``Tuple[ParameterExpression, ParameterExpression]`` or\n                ``List[Tuple[ParameterExpression, ParameterExpression]]``\n                is given, then the 2nd order derivative of the operator is calculated.\n        Returns:\n            The adapted operator.\n            Measurement operators are attached with an additional Z term acting\n            on an additional working qubit.\n            Quantum states - which must be given as circuits - are adapted. An additional\n            working qubit controls intercepting gates.\n            See e.g. [1].\n\n        Raises:\n            ValueError: If ``operator`` does not correspond to an expectation value.\n            TypeError: If the ``StateFn`` corresponding to the quantum state could not be extracted\n                       from ``operator``.\n            OpflowError: If third or higher order gradients are requested.\n\n        References:\n            [1]: Evaluating analytic gradients on quantum hardware\n                 Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and Nathan Killoran\n                 Phys. Rev. A 99, 032331 – Published 21 March 2019\n\n        '
        if isinstance(operator, ComposedOp):
            if not isinstance(operator[0], StateFn) or not operator[0].is_measurement:
                raise ValueError('The given operator does not correspond to an expectation value')
            if not isinstance(operator[-1], StateFn) or operator[-1].is_measurement:
                raise ValueError('The given operator does not correspond to an expectation value')
            if operator[0].is_measurement:
                meas = deepcopy(operator.oplist[0])
                meas = meas.primitive * meas.coeff
                if len(operator.oplist) == 2:
                    state_op = operator[1]
                    if not isinstance(state_op, StateFn):
                        raise TypeError('The StateFn representing the quantum state could not be extracted.')
                    if isinstance(params, (ParameterExpression, ParameterVector)) or (isinstance(params, list) and all((isinstance(param, ParameterExpression) for param in params))):
                        return self._gradient_states(state_op, meas_op=2 * meas, target_params=params)
                    elif isinstance(params, tuple) or (isinstance(params, list) and all((isinstance(param, tuple) for param in params))):
                        return self._hessian_states(state_op, meas_op=4 * (I ^ meas), target_params=params)
                    else:
                        raise OpflowError('The linear combination gradient does only support the computation of 1st gradients and 2nd order gradients.')
                else:
                    state_op = deepcopy(operator)
                    state_op.oplist.pop(0)
                    if not isinstance(state_op, StateFn):
                        raise TypeError('The StateFn representing the quantum state could not be extracted.')
                    if isinstance(params, (ParameterExpression, ParameterVector)) or (isinstance(params, list) and all((isinstance(param, ParameterExpression) for param in params))):
                        return state_op.traverse(partial(self._gradient_states, meas_op=2 * meas, target_params=params))
                    elif isinstance(params, tuple) or (isinstance(params, list) and all((isinstance(param, tuple) for param in params))):
                        return state_op.traverse(partial(self._hessian_states, meas_op=4 * I ^ meas, target_params=params))
                    raise OpflowError('The linear combination gradient only supports the computation of 1st and 2nd order gradients.')
            else:
                return operator.traverse(partial(self._prepare_operator, params=params))
        elif isinstance(operator, ListOp):
            return operator.traverse(partial(self._prepare_operator, params=params))
        elif isinstance(operator, StateFn):
            if operator.is_measurement:
                return operator.traverse(partial(self._prepare_operator, params=params))
            elif isinstance(params, (ParameterExpression, ParameterVector)) or (isinstance(params, list) and all((isinstance(param, ParameterExpression) for param in params))):
                return self._gradient_states(operator, target_params=params)
            elif isinstance(params, tuple) or (isinstance(params, list) and all((isinstance(param, tuple) for param in params))):
                return self._hessian_states(operator, target_params=params)
            else:
                raise OpflowError('The linear combination gradient does only support the computation of 1st gradients and 2nd order gradients.')
        elif isinstance(operator, PrimitiveOp):
            return operator
        return operator

    @staticmethod
    def _grad_combo_fn(x, state_op):
        if False:
            while True:
                i = 10

        def get_result(item):
            if False:
                print('Hello World!')
            if isinstance(item, (DictStateFn, SparseVectorStateFn)):
                item = item.primitive
            if isinstance(item, VectorStateFn):
                item = item.primitive.data
            if isinstance(item, dict):
                prob_dict = {}
                for (key, val) in item.items():
                    prob_counts = val * np.conj(val)
                    if int(key[0]) == 1:
                        prob_counts *= -1
                    suffix = key[1:]
                    prob_dict[suffix] = prob_dict.get(suffix, 0) + prob_counts
                for key in prob_dict:
                    prob_dict[key] *= 2
                return prob_dict
            elif isinstance(item, scipy.sparse.spmatrix):
                trace = _z_exp(item)
                return trace
            elif isinstance(item, Iterable):
                lin_comb_op = 2 * Z ^ (I ^ state_op.num_qubits)
                lin_comb_op = lin_comb_op.to_matrix()
                outer = np.outer(item, item.conj())
                return list(np.diag(partial_trace(lin_comb_op.dot(outer), [state_op.num_qubits]).data))
            else:
                raise TypeError('The state result should be either a DictStateFn or a VectorStateFn.')
        if not isinstance(x, Iterable):
            return get_result(x)
        elif len(x) == 1:
            return get_result(x[0])
        else:
            result = []
            for item in x:
                result.append(get_result(item))
            return result

    @staticmethod
    def _hess_combo_fn(x, state_op):
        if False:
            while True:
                i = 10

        def get_result(item):
            if False:
                print('Hello World!')
            if isinstance(item, DictStateFn):
                item = item.primitive
            if isinstance(item, VectorStateFn):
                item = item.primitive.data
            if isinstance(item, Iterable):
                lin_comb_op = 4 * (I ^ state_op.num_qubits + 1) ^ Z
                lin_comb_op = lin_comb_op.to_matrix()
                return list(np.diag(partial_trace(lin_comb_op.dot(np.outer(item, np.conj(item))), [0, 1]).data))
            elif isinstance(item, scipy.sparse.spmatrix):
                trace = _z_exp(item)
                return trace
            elif isinstance(item, dict):
                prob_dict = {}
                for (key, val) in item.values():
                    prob_counts = val * np.conj(val)
                    if int(key[-1]) == 1:
                        prob_counts *= -1
                    prefix = key[:-2]
                    prob_dict[prefix] = prob_dict.get(prefix, 0) + prob_counts
                for key in prob_dict:
                    prob_dict[key] *= 4
                return prob_dict
            else:
                raise TypeError('The state result should be either a DictStateFn or a VectorStateFn.')
        if not isinstance(x, Iterable):
            return get_result(x)
        elif len(x) == 1:
            return get_result(x[0])
        else:
            result = []
            for item in x:
                result.append(get_result(item))
            return result

    @staticmethod
    def _gate_gradient_dict(gate: Gate) -> List[Tuple[List[complex], List[Instruction]]]:
        if False:
            for i in range(10):
                print('nop')
        "Given a parameterized gate U(theta) with derivative\n        dU(theta)/dtheta = sum_ia_iU(theta)V_i.\n        This function returns a:=[a_0, ...] and V=[V_0, ...]\n        Suppose U takes multiple parameters, i.e., U(theta^0, ... theta^k).\n        The returned coefficients and gates are ordered accordingly.\n        Only parameterized Qiskit gates are supported.\n\n        Args:\n            gate: The gate for which the derivative is being computed.\n\n        Returns:\n            The coefficients and the gates used for the metric computation for each parameter of\n            the respective gates ``[([a^0], [V^0]) ..., ([a^k], [V^k])]``.\n\n        Raises:\n            OpflowError: If the input gate is controlled by another state but '|1>^{\\otimes k}'\n            TypeError: If the input gate is not a supported parameterized gate.\n        "
        if isinstance(gate, PhaseGate):
            return [([0.5j, -0.5j], [IGate(), CZGate()])]
        if isinstance(gate, UGate):
            return [([-0.5j], [CZGate()]), ([-0.5j], [CZGate()]), ([-0.5j], [CZGate()])]
        if isinstance(gate, RXGate):
            return [([-0.5j], [CXGate()])]
        if isinstance(gate, RYGate):
            return [([-0.5j], [CYGate()])]
        if isinstance(gate, RZGate):
            return [([-0.5j], [CZGate()])]
        if isinstance(gate, RXXGate):
            cxx_circ = QuantumCircuit(3)
            cxx_circ.cx(0, 1)
            cxx_circ.cx(0, 2)
            cxx = cxx_circ.to_instruction()
            return [([-0.5j], [cxx])]
        if isinstance(gate, RYYGate):
            cyy_circ = QuantumCircuit(3)
            cyy_circ.cy(0, 1)
            cyy_circ.cy(0, 2)
            cyy = cyy_circ.to_instruction()
            return [([-0.5j], [cyy])]
        if isinstance(gate, RZZGate):
            czz_circ = QuantumCircuit(3)
            czz_circ.cz(0, 1)
            czz_circ.cz(0, 2)
            czz = czz_circ.to_instruction()
            return [([-0.5j], [czz])]
        if isinstance(gate, RZXGate):
            czx_circ = QuantumCircuit(3)
            czx_circ.cx(0, 2)
            czx_circ.cz(0, 1)
            czx = czx_circ.to_instruction()
            return [([-0.5j], [czx])]
        if isinstance(gate, ControlledGate):
            if gate.ctrl_state != 2 ** gate.num_ctrl_qubits - 1:
                raise OpflowError('Function only support controlled gates with control state `1` on all control qubits.')
            base_coeffs_gates = LinComb._gate_gradient_dict(gate.base_gate)
            coeffs_gates = []
            proj_gates_controlled = [[(-1) ** p.count(ZGate()), p] for p in product([IGate(), ZGate()], repeat=gate.num_ctrl_qubits)]
            for (base_coeffs, base_gates) in base_coeffs_gates:
                coeffs = []
                gates = []
                for (phase, proj_gates) in proj_gates_controlled:
                    coeffs.extend([phase * c / 2 ** gate.num_ctrl_qubits for c in base_coeffs])
                    for base_gate in base_gates:
                        controlled_circ = QuantumCircuit(gate.num_ctrl_qubits + gate.num_qubits)
                        for (i, proj_gate) in enumerate(proj_gates):
                            if isinstance(proj_gate, ZGate):
                                controlled_circ.cz(0, i + 1)
                        if not isinstance(base_gate, IGate):
                            controlled_circ.append(base_gate, [0, range(gate.num_ctrl_qubits + 1, gate.num_ctrl_qubits + gate.num_qubits)])
                        gates.append(controlled_circ.to_instruction())
                c_g = (coeffs, gates)
                coeffs_gates.append(c_g)
            return coeffs_gates
        raise TypeError(f'Unrecognized parameterized gate, {gate}')

    @staticmethod
    def apply_grad_gate(circuit, gate, param_index, grad_gate, grad_coeff, qr_superpos, open_ctrl=False, trim_after_grad_gate=False):
        if False:
            for i in range(10):
                print('nop')
        'Util function to apply a gradient gate for the linear combination of unitaries method.\n        Replaces the ``gate`` instance in ``circuit`` with ``grad_gate`` using ``qr_superpos`` as\n        superposition qubit. Also adds the appropriate sign-fix gates on the superposition qubit.\n\n        Args:\n            circuit (QuantumCircuit): The circuit in which to do the replacements.\n            gate (Gate): The gate instance to replace.\n            param_index (int): The index of the parameter in ``gate``.\n            grad_gate (Gate): A controlled gate encoding the gradient of ``gate``.\n            grad_coeff (float): A coefficient to the gradient component. Might not be one if the\n                gradient contains multiple summed terms.\n            qr_superpos (QuantumRegister): A ``QuantumRegister`` of size 1 contained in ``circuit``\n                that is used as control for ``grad_gate``.\n            open_ctrl (bool): If True use an open control for ``grad_gate`` instead of closed.\n            trim_after_grad_gate (bool): If True remove all gates after the ``grad_gate``. Can\n                be used to reduce the circuit depth in e.g. computing an overlap of gradients.\n\n        Returns:\n            QuantumCircuit: A copy of the original circuit with the gradient gate added.\n\n        Raises:\n            RuntimeError: If ``gate`` is not in ``circuit``.\n        '
        qr_superpos_qubits = tuple(qr_superpos)
        out = QuantumCircuit(*circuit.qregs)
        out._data = circuit._data.copy()
        out._parameter_table = ParameterTable({param: values.copy() for (param, values) in circuit._parameter_table.items()})
        (gate_idx, gate_qubits) = (None, None)
        for (i, instruction) in enumerate(out._data):
            if instruction.operation is gate:
                (gate_idx, gate_qubits) = (i, instruction.qubits)
                break
        if gate_idx is None:
            raise RuntimeError('The specified gate could not be found in the circuit data.')
        replacement = []
        sign = np.sign(grad_coeff)
        is_complex = np.iscomplex(grad_coeff)
        if sign < 0 and is_complex:
            replacement.append(CircuitInstruction(SdgGate(), qr_superpos_qubits, ()))
        elif sign < 0:
            replacement.append(CircuitInstruction(ZGate(), qr_superpos_qubits, ()))
        elif is_complex:
            replacement.append(CircuitInstruction(SGate(), qr_superpos_qubits, ()))
        if open_ctrl:
            replacement += [CircuitInstruction(XGate(), qr_superpos_qubits, [])]
        if isinstance(gate, UGate) and param_index == 0:
            theta = gate.params[2]
            (rz_plus, rz_minus) = (RZGate(theta), RZGate(-theta))
            replacement += [CircuitInstruction(rz_plus, (qubit,), ()) for qubit in gate_qubits]
            replacement += [CircuitInstruction(RXGate(np.pi / 2), (qubit,), ()) for qubit in gate_qubits]
            replacement.append(CircuitInstruction(grad_gate, qr_superpos_qubits + gate_qubits, []))
            replacement += [CircuitInstruction(RXGate(-np.pi / 2), (qubit,), ()) for qubit in gate_qubits]
            replacement += [CircuitInstruction(rz_minus, (qubit,), ()) for qubit in gate_qubits]
            if isinstance(theta, ParameterExpression):
                out._update_parameter_table(CircuitInstruction(rz_plus, (gate_qubits[0],), ()))
                out._update_parameter_table(CircuitInstruction(rz_minus, (gate_qubits[0],), ()))
            if open_ctrl:
                replacement.append(CircuitInstruction(XGate(), qr_superpos_qubits, ()))
            if not trim_after_grad_gate:
                replacement.append(CircuitInstruction(gate, gate_qubits, ()))
        elif isinstance(gate, UGate) and param_index == 1:
            replacement.append(CircuitInstruction(gate, gate_qubits, ()))
            replacement.append(CircuitInstruction(grad_gate, qr_superpos_qubits + gate_qubits, ()))
            if open_ctrl:
                replacement.append(CircuitInstruction(XGate(), qr_superpos_qubits, ()))
        else:
            replacement.append(CircuitInstruction(grad_gate, qr_superpos_qubits + gate_qubits, ()))
            if open_ctrl:
                replacement.append(CircuitInstruction(XGate(), qr_superpos_qubits, ()))
            if not trim_after_grad_gate:
                replacement.append(CircuitInstruction(gate, gate_qubits, ()))
        if trim_after_grad_gate:
            out._data[gate_idx:] = replacement
            table = ParameterTable()
            for instruction in out._data:
                for (idx, param_expression) in enumerate(instruction.operation.params):
                    if isinstance(param_expression, ParameterExpression):
                        for param in param_expression.parameters:
                            if param not in table.keys():
                                table[param] = ParameterReferences(((instruction.operation, idx),))
                            else:
                                table[param].add((instruction.operation, idx))
            out._parameter_table = table
        else:
            out._data[gate_idx:gate_idx + 1] = replacement
        return out

    def _aux_meas_basis_trafo(self, aux_meas_op: OperatorBase, state: StateFn, state_op: StateFn, combo_fn: Callable) -> ListOp:
        if False:
            i = 10
            return i + 15
        '\n        This function applies the necessary basis transformation to measure the quantum state in\n        a different basis -- given by the auxiliary measurement operator ``aux_meas_op``.\n\n        Args:\n            aux_meas_op: The auxiliary measurement operator defines the necessary measurement basis.\n            state: This operator represents the gradient or Hessian before the basis transformation.\n            state_op: The operator representing the quantum state for which we compute the gradient\n                or Hessian.\n            combo_fn: This ``combo_fn`` defines whether the target is a gradient or Hessian.\n\n\n        Returns:\n            Operator representing the gradient or Hessian.\n\n        Raises:\n            ValueError: If ``aux_meas_op`` is neither ``Z`` nor ``-Y`` nor ``Z - 1j * Y``.\n\n        '
        if aux_meas_op == Z - 1j * Y:
            state_z = ListOp([state], combo_fn=partial(combo_fn, state_op=state_op))
            pbc = PauliBasisChange(replacement_fn=PauliBasisChange.measurement_replacement_fn)
            pbc = pbc.convert(-Y ^ (I ^ state.num_qubits - 1))
            state_y = pbc[-1] @ state
            state_y = ListOp([state_y], combo_fn=partial(combo_fn, state_op=state_op))
            return state_z - 1j * state_y
        elif aux_meas_op == -Y:
            pbc = PauliBasisChange(replacement_fn=PauliBasisChange.measurement_replacement_fn)
            pbc = pbc.convert(aux_meas_op ^ (I ^ state.num_qubits - 1))
            state = pbc[-1] @ state
            return -1 * ListOp([state], combo_fn=partial(combo_fn, state_op=state_op))
        elif aux_meas_op == Z:
            return ListOp([state], combo_fn=partial(combo_fn, state_op=state_op))
        else:
            raise ValueError(f'The auxiliary measurement operator passed {aux_meas_op} is not supported. Only Y, Z, or Z - 1j * Y are valid.')

    def _gradient_states(self, state_op: StateFn, meas_op: Optional[OperatorBase]=None, target_params: Optional[Union[Parameter, List[Parameter]]]=None, open_ctrl: bool=False, trim_after_grad_gate: bool=False) -> ListOp:
        if False:
            i = 10
            return i + 15
        'Generate the gradient states.\n\n        Args:\n            state_op: The operator representing the quantum state for which we compute the gradient.\n            meas_op: The operator representing the observable for which we compute the gradient.\n            target_params: The parameters we are taking the gradient wrt: ω\n            open_ctrl: If True use an open control for ``grad_gate`` instead of closed.\n            trim_after_grad_gate: If True remove all gates after the ``grad_gate``. Can\n                be used to reduce the circuit depth in e.g. computing an overlap of gradients.\n\n        Returns:\n            ListOp of StateFns as quantum circuits which are the states w.r.t. which we compute the\n            gradient. If a parameter appears multiple times, one circuit is created per\n            parameterized gates to compute the product rule.\n\n        Raises:\n            QiskitError: If one of the circuits could not be constructed.\n            TypeError: If the operators is of unsupported type.\n            ValueError: If the auxiliary operator preparation fails.\n        '
        unrolled = self._transpile_to_supported_operations(state_op.primitive, self.SUPPORTED_GATES)
        qr_superpos = QuantumRegister(1)
        state_qc = QuantumCircuit(*state_op.primitive.qregs, qr_superpos)
        state_qc.h(qr_superpos)
        state_qc.compose(unrolled, inplace=True)
        if not isinstance(target_params, (list, np.ndarray)):
            target_params = [target_params]
        oplist = []
        for param in target_params:
            if param not in state_qc.parameters:
                oplist += [~Zero @ One]
            else:
                param_gates = state_qc._parameter_table[param]
                sub_oplist = []
                for (gate, idx) in param_gates:
                    (grad_coeffs, grad_gates) = self._gate_gradient_dict(gate)[idx]
                    for (grad_coeff, grad_gate) in zip(grad_coeffs, grad_gates):
                        grad_circuit = self.apply_grad_gate(state_qc, gate, idx, grad_gate, grad_coeff, qr_superpos, open_ctrl, trim_after_grad_gate)
                        grad_circuit.h(qr_superpos)
                        coeff = np.sqrt(np.abs(grad_coeff)) * state_op.coeff
                        state = CircuitStateFn(grad_circuit, coeff=coeff)
                        param_expression = gate.params[idx]
                        if isinstance(meas_op, OperatorBase):
                            state = StateFn(self._aux_meas_op ^ meas_op, is_measurement=True) @ state
                        else:
                            state = self._aux_meas_basis_trafo(self._aux_meas_op, state, state_op, self._grad_combo_fn)
                        if param_expression != param:
                            param_grad = param_expression.gradient(param)
                            state *= param_grad
                        sub_oplist += [state]
                oplist += [SummedOp(sub_oplist) if len(sub_oplist) > 1 else sub_oplist[0]]
        return ListOp(oplist) if len(oplist) > 1 else oplist[0]

    def _hessian_states(self, state_op: StateFn, meas_op: Optional[OperatorBase]=None, target_params: Optional[Union[Tuple[ParameterExpression, ParameterExpression], List[Tuple[ParameterExpression, ParameterExpression]]]]=None) -> OperatorBase:
        if False:
            for i in range(10):
                print('nop')
        'Generate the operator states whose evaluation returns the Hessian (items).\n\n        Args:\n            state_op: The operator representing the quantum state for which we compute the Hessian.\n            meas_op: The operator representing the observable for which we compute the gradient.\n            target_params: The parameters we are computing the Hessian wrt: ω\n\n        Returns:\n            Operators which give the Hessian. If a parameter appears multiple times, one circuit is\n            created per parameterized gates to compute the product rule.\n\n        Raises:\n            QiskitError: If one of the circuits could not be constructed.\n            TypeError: If ``operator`` is of unsupported type.\n            ValueError: If the auxiliary operator preparation fails.\n        '
        if not isinstance(target_params, list):
            target_params = [target_params]
        if not all((isinstance(params, tuple) for params in target_params)):
            raise TypeError('Please define in the parameters for which the Hessian is evaluated either as parameter tuple or a list of parameter tuples')
        qr_add0 = QuantumRegister(1, 's0')
        qr_add1 = QuantumRegister(1, 's1')
        state_qc = QuantumCircuit(*state_op.primitive.qregs, qr_add0, qr_add1)
        state_qc.h(qr_add0)
        state_qc.h(qr_add1)
        state_qc.compose(state_op.primitive, inplace=True)
        oplist = []
        for (param_a, param_b) in target_params:
            if param_a not in state_qc.parameters or param_b not in state_qc.parameters:
                oplist += [~Zero @ One]
            else:
                sub_oplist = []
                param_gates_a = state_qc._parameter_table[param_a]
                param_gates_b = state_qc._parameter_table[param_b]
                for (gate_a, idx_a) in param_gates_a:
                    (grad_coeffs_a, grad_gates_a) = self._gate_gradient_dict(gate_a)[idx_a]
                    for (grad_coeff_a, grad_gate_a) in zip(grad_coeffs_a, grad_gates_a):
                        grad_circuit = self.apply_grad_gate(state_qc, gate_a, idx_a, grad_gate_a, grad_coeff_a, qr_add0)
                        for (gate_b, idx_b) in param_gates_b:
                            (grad_coeffs_b, grad_gates_b) = self._gate_gradient_dict(gate_b)[idx_b]
                            for (grad_coeff_b, grad_gate_b) in zip(grad_coeffs_b, grad_gates_b):
                                hessian_circuit = self.apply_grad_gate(grad_circuit, gate_b, idx_b, grad_gate_b, grad_coeff_b, qr_add1)
                                hessian_circuit.h(qr_add0)
                                hessian_circuit.cz(qr_add1[0], qr_add0[0])
                                hessian_circuit.h(qr_add1)
                                coeff = state_op.coeff
                                coeff *= np.sqrt(np.abs(grad_coeff_a) * np.abs(grad_coeff_b))
                                state = CircuitStateFn(hessian_circuit, coeff=coeff)
                                if meas_op is not None:
                                    state = StateFn(self._aux_meas_op ^ meas_op, is_measurement=True) @ state
                                else:
                                    state = self._aux_meas_basis_trafo(self._aux_meas_op, state, state_op, self._hess_combo_fn)
                                param_grad = 1
                                for (gate, idx, param) in zip([gate_a, gate_b], [idx_a, idx_b], [param_a, param_b]):
                                    param_expression = gate.params[idx]
                                    if param_expression != param:
                                        param_grad *= param_expression.gradient(param)
                                if param_grad != 1:
                                    state *= param_grad
                                sub_oplist += [state]
                oplist += [SummedOp(sub_oplist) if len(sub_oplist) > 1 else sub_oplist[0]]
        return ListOp(oplist) if len(oplist) > 1 else oplist[0]

def _z_exp(spmatrix):
    if False:
        return 10
    'Compute the sampling probabilities of the qubits after applying measurement on the\n    auxiliary qubit.'
    dok = spmatrix.todok()
    num_qubits = int(np.log2(dok.shape[1]))
    exp = scipy.sparse.dok_matrix((1, 2 ** (num_qubits - 1)))
    for (index, amplitude) in dok.items():
        binary = bin(index[1])[2:].zfill(num_qubits)
        sign = -1 if binary[0] == '1' else 1
        new_index = int(binary[1:], 2)
        exp[0, new_index] = exp[0, new_index] + 2 * sign * np.abs(amplitude) ** 2
    return exp