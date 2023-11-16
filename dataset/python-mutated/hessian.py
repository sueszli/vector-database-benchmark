"""The module to compute Hessians."""
from typing import Union, List, Tuple, Optional
import numpy as np
from qiskit.circuit import ParameterVector, ParameterExpression
from qiskit.circuit._utils import sort_parameters
from qiskit.utils import optionals as _optionals
from qiskit.utils.deprecation import deprecate_func
from ..operator_globals import Zero, One
from ..state_fns.circuit_state_fn import CircuitStateFn
from ..state_fns.state_fn import StateFn
from ..expectations.pauli_expectation import PauliExpectation
from ..list_ops.list_op import ListOp
from ..list_ops.composed_op import ComposedOp
from ..list_ops.summed_op import SummedOp
from ..list_ops.tensored_op import TensoredOp
from ..operator_base import OperatorBase
from .gradient import Gradient
from .derivative_base import _coeff_derivative
from .hessian_base import HessianBase
from ..exceptions import OpflowError
from ...utils.arithmetic import triu_to_dense
from .circuit_gradients.circuit_gradient import CircuitGradient

class Hessian(HessianBase):
    """Deprecated: Compute the Hessian of an expected value."""

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, hess_method: Union[str, CircuitGradient]='param_shift', **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(hess_method=hess_method, **kwargs)

    def convert(self, operator: OperatorBase, params: Optional[Union[Tuple[ParameterExpression, ParameterExpression], List[Tuple[ParameterExpression, ParameterExpression]], List[ParameterExpression], ParameterVector]]=None) -> OperatorBase:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            operator: The operator for which we compute the Hessian\n            params: The parameters we are computing the Hessian with respect to\n                    Either give directly the tuples/list of tuples for which the second order\n                    derivative is to be computed or give a list of parameters to build the\n                    full Hessian for those parameters. If not explicitly passed, the full Hessian is\n                    constructed. The parameters are then inferred from the operator and sorted by\n                    name.\n\n        Returns:\n            OperatorBase: An operator whose evaluation yields the Hessian\n        '
        expec_op = PauliExpectation(group_paulis=False).convert(operator).reduce()
        cleaned_op = self._factor_coeffs_out_of_composed_op(expec_op)
        return self.get_hessian(cleaned_op, params)

    def get_hessian(self, operator: OperatorBase, params: Optional[Union[Tuple[ParameterExpression, ParameterExpression], List[Tuple[ParameterExpression, ParameterExpression]], List[ParameterExpression], ParameterVector]]=None) -> OperatorBase:
        if False:
            return 10
        'Get the Hessian for the given operator w.r.t. the given parameters\n\n        Args:\n            operator: Operator w.r.t. which we take the Hessian.\n            params: Parameters w.r.t. which we compute the Hessian. If not explicitly passed,\n                the full Hessian is constructed. The parameters are then inferred from the operator\n                and sorted by name.\n\n        Returns:\n            Operator which represents the gradient w.r.t. the given params.\n\n        Raises:\n            ValueError: If ``params`` contains a parameter not present in ``operator``.\n            ValueError: If ``operator`` is not parameterized.\n            OpflowError: If the coefficient of the operator could not be reduced to 1.\n            OpflowError: If the differentiation of a combo_fn\n                         requires JAX but the package is not installed.\n            TypeError: If the operator does not include a StateFn given by a quantum circuit\n            TypeError: If the parameters were given in an unsupported format.\n            Exception: Unintended code is reached\n            MissingOptionalLibraryError: jax not installed\n        '
        if len(operator.parameters) == 0:
            raise ValueError('The operator we are taking the gradient of is not parameterized!')
        if params is None:
            params = sort_parameters(operator.parameters)
        if isinstance(params, (ParameterVector, list)):
            if all((isinstance(param, ParameterExpression) for param in params)):
                return ListOp([ListOp([self.get_hessian(operator, (p_i, p_j)) for (i, p_i) in enumerate(params[j:], j)]) for (j, p_j) in enumerate(params)], combo_fn=triu_to_dense)
            elif all((isinstance(param, tuple) for param in params)):
                return ListOp([self.get_hessian(operator, param_pair) for param_pair in params])

        def is_coeff_c(coeff, c):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(coeff, ParameterExpression):
                expr = coeff._symbol_expr
                return expr == c
            return coeff == c
        if isinstance(params, ParameterExpression):
            return Gradient(grad_method=self._hess_method).get_gradient(operator, params)
        if not isinstance(params, tuple) or not len(params) == 2:
            raise TypeError('Parameters supplied in unsupported format.')
        p_0 = params[0]
        p_1 = params[1]
        if not is_coeff_c(operator._coeff, 1.0):
            coeff = operator._coeff
            op = operator / coeff
            d0_op = self.get_hessian(op, p_0)
            d1_op = self.get_hessian(op, p_1)
            d0_coeff = _coeff_derivative(coeff, p_0)
            d1_coeff = _coeff_derivative(coeff, p_1)
            dd_op = self.get_hessian(op, params)
            dd_coeff = _coeff_derivative(d0_coeff, p_1)
            grad_op = 0
            if dd_op != ~Zero @ One and (not is_coeff_c(coeff, 0)):
                grad_op += coeff * dd_op
            if d0_op != ~Zero @ One and (not is_coeff_c(d1_coeff, 0)):
                grad_op += d1_coeff * d0_op
            if d1_op != ~Zero @ One and (not is_coeff_c(d0_coeff, 0)):
                grad_op += d0_coeff * d1_op
            if not is_coeff_c(dd_coeff, 0):
                grad_op += dd_coeff * op
            if grad_op == 0:
                return ~Zero @ One
            return grad_op
        if isinstance(operator, ComposedOp):
            if not is_coeff_c(operator.coeff, 1.0):
                raise OpflowError('Operator pre-processing failed. Coefficients were not properly collected inside the ComposedOp.')
            if not isinstance(operator[-1], CircuitStateFn):
                raise TypeError('The gradient framework is compatible with states that are given as CircuitStateFn')
            return self.hess_method.convert(operator, params)
        elif isinstance(operator, ListOp):
            dd_ops = [self.get_hessian(op, params) for op in operator.oplist]
            if operator.combo_fn == ListOp([]).combo_fn:
                return ListOp(oplist=dd_ops)
            elif isinstance(operator, SummedOp):
                return SummedOp(oplist=dd_ops)
            elif isinstance(operator, TensoredOp):
                return TensoredOp(oplist=dd_ops)
            d1d0_ops = ListOp([ListOp([Gradient(grad_method=self._hess_method).convert(op, param) for param in params], combo_fn=np.prod) for op in operator.oplist])
            _optionals.HAS_JAX.require_now('automatic differentiation')
            from jax import grad, jit
            if operator.grad_combo_fn:
                first_partial_combo_fn = operator.grad_combo_fn
                second_partial_combo_fn = jit(grad(lambda x: first_partial_combo_fn(x)[0], holomorphic=True))
            else:
                first_partial_combo_fn = jit(grad(operator.combo_fn, holomorphic=True))
                second_partial_combo_fn = jit(grad(lambda x: first_partial_combo_fn(x)[0], holomorphic=True))
            term1 = ListOp([ListOp(operator.oplist, combo_fn=first_partial_combo_fn), ListOp(dd_ops)], combo_fn=lambda x: np.dot(x[1], x[0]))
            term2 = ListOp([ListOp(operator.oplist, combo_fn=second_partial_combo_fn), d1d0_ops], combo_fn=lambda x: np.dot(x[1], x[0]))
            return SummedOp([term1, term2])
        elif isinstance(operator, StateFn):
            if not operator.is_measurement:
                return self.hess_method.convert(operator, params)
            else:
                raise TypeError('The computation of Hessians is only supported for Operators which represent expectation values or quantum states.')
        else:
            raise TypeError('The computation of Hessians is only supported for Operators which represent expectation values.')