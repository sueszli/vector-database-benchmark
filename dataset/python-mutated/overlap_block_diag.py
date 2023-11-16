"""The module for the Quantum Fisher Information."""
from typing import List, Union
import numpy as np
from scipy.linalg import block_diag
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression
from qiskit.utils.arithmetic import triu_to_dense
from qiskit.utils.deprecation import deprecate_func
from ...list_ops.list_op import ListOp
from ...primitive_ops.circuit_op import CircuitOp
from ...expectations.pauli_expectation import PauliExpectation
from ...operator_globals import Zero
from ...state_fns.state_fn import StateFn
from ...state_fns.circuit_state_fn import CircuitStateFn
from ...exceptions import OpflowError
from .circuit_qfi import CircuitQFI
from ..derivative_base import _coeff_derivative
from .overlap_diag import _get_generators, _partition_circuit

class OverlapBlockDiag(CircuitQFI):
    """Deprecated: Compute the block-diagonal of the QFI given a pure, parameterized quantum state.

    The blocks are given by all parameterized gates in quantum circuit layer.
    See also :class:`~qiskit.opflow.QFI`.
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()

    def convert(self, operator: Union[CircuitOp, CircuitStateFn], params: Union[ParameterExpression, ParameterVector, List[ParameterExpression]]) -> ListOp:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            operator: The operator corresponding to the quantum state :math:`|\\psi(\\omega)\\rangle`\n                for which we compute the QFI.\n            params: The parameters :math:`\\omega` with respect to which we are computing the QFI.\n\n        Returns:\n            A ``ListOp[ListOp]`` where the operator at position ``[k][l]`` corresponds to the matrix\n            element :math:`k, l` of the QFI.\n\n        Raises:\n            NotImplementedError: If ``operator`` is neither ``CircuitOp`` nor ``CircuitStateFn``.\n        '
        if not isinstance(operator, (CircuitOp, CircuitStateFn)):
            raise NotImplementedError('operator must be a CircuitOp or CircuitStateFn')
        return self._block_diag_approx(operator=operator, params=params)

    def _block_diag_approx(self, operator: Union[CircuitOp, CircuitStateFn], params: Union[ParameterExpression, ParameterVector, List[ParameterExpression]]) -> ListOp:
        if False:
            return 10
        '\n        Args:\n            operator: The operator corresponding to the quantum state :math:`|\\psi(\\omega)\\rangle`\n                for which we compute the QFI.\n            params: The parameters :math:`\\omega` with respect to which we are computing the QFI.\n\n        Returns:\n            A ``ListOp[ListOp]`` where the operator at position ``[k][l]`` corresponds to the matrix\n            element :math:`k, l` of the QFI.\n\n        Raises:\n            NotImplementedError: If a circuit is found such that one parameter controls multiple\n                gates, or one gate contains multiple parameters.\n            OpflowError: If there are more than one parameter.\n\n        '
        if isinstance(params, ParameterExpression):
            params = [params]
        circuit = operator.primitive
        layers = _partition_circuit(circuit)
        if layers[-1].num_parameters == 0:
            layers.pop(-1)
        block_params = [list(layer.parameters) for layer in layers]
        block_params = [[param for param in block if param in params] for block in block_params]
        perm = [params.index(param) for block in block_params for param in block]
        psis = [CircuitOp(layer) for layer in layers]
        for (i, psi) in enumerate(psis):
            if i == 0:
                continue
            psis[i] = psi @ psis[i - 1]
        generators = _get_generators(params, circuit)
        blocks = []
        for (k, psi_i) in enumerate(psis):
            params = block_params[k]
            block = np.zeros((len(params), len(params))).tolist()
            single_terms = np.zeros(len(params)).tolist()
            for (i, p_i) in enumerate(params):
                generator = generators[p_i]
                psi_gen_i = ~StateFn(generator) @ psi_i @ Zero
                psi_gen_i = PauliExpectation().convert(psi_gen_i)
                single_terms[i] = psi_gen_i

            def get_parameter_expression(circuit, param):
                if False:
                    return 10
                if len(circuit._parameter_table[param]) > 1:
                    raise NotImplementedError('OverlapDiag does not yet support multiple gates parameterized by a single parameter. For such circuits use LinCombFull')
                gate = next(iter(circuit._parameter_table[param]))[0]
                if len(gate.params) > 1:
                    raise OpflowError('OverlapDiag cannot yet support gates with more than one parameter.')
                param_value = gate.params[0]
                return param_value
            for (i, p_i) in enumerate(params):
                generator_i = generators[p_i]
                param_expr_i = get_parameter_expression(circuit, p_i)
                for (j, p_j) in enumerate(params[i:], i):
                    if i == j:
                        block[i][i] = ListOp([single_terms[i]], combo_fn=lambda x: 1 - x[0] ** 2)
                        if isinstance(param_expr_i, ParameterExpression) and (not isinstance(param_expr_i, Parameter)):
                            expr_grad_i = _coeff_derivative(param_expr_i, p_i)
                            block[i][i] *= expr_grad_i * expr_grad_i
                        continue
                    generator_j = generators[p_j]
                    generator = ~generator_j @ generator_i
                    param_expr_j = get_parameter_expression(circuit, p_j)
                    psi_gen_ij = ~StateFn(generator) @ psi_i @ Zero
                    psi_gen_ij = PauliExpectation().convert(psi_gen_ij)
                    cross_term = ListOp([single_terms[i], single_terms[j]], combo_fn=np.prod)
                    block[i][j] = psi_gen_ij - cross_term
                    if type(param_expr_i) == ParameterExpression:
                        expr_grad_i = _coeff_derivative(param_expr_i, p_i)
                        block[i][j] *= expr_grad_i
                    if type(param_expr_j) == ParameterExpression:
                        expr_grad_j = _coeff_derivative(param_expr_j, p_j)
                        block[i][j] *= expr_grad_j
            wrapped_block = ListOp([ListOp([block[i][j] for j in range(i, len(params))]) for i in range(len(params))], combo_fn=triu_to_dense)
            blocks.append(wrapped_block)
        return ListOp(oplist=blocks, combo_fn=lambda x: np.real(block_diag(*x))[:, perm][perm, :])