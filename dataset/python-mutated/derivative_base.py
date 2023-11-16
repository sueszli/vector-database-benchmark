"""DerivativeBase Class"""
from abc import abstractmethod
from typing import Callable, Iterable, List, Optional, Tuple, Union
import numpy as np
from qiskit.utils.deprecation import deprecate_func
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.providers import Backend
from ..converters.converter_base import ConverterBase
from ..expectations import ExpectationBase, PauliExpectation
from ..list_ops.composed_op import ComposedOp
from ..list_ops.list_op import ListOp
from ..list_ops.tensored_op import TensoredOp
from ..operator_base import OperatorBase
from ..primitive_ops.primitive_op import PrimitiveOp
from ..state_fns import StateFn, OperatorStateFn
OperatorType = Union[StateFn, PrimitiveOp, ListOp]

class DerivativeBase(ConverterBase):
    """Deprecated: Base class for differentiating opflow objects.

    Converter for differentiating opflow objects and handling
    things like properly differentiating combo_fn's and enforcing product rules
    when operator coefficients are parameterized.

    This is distinct from CircuitGradient converters which use quantum
    techniques such as parameter shifts and linear combination of unitaries
    to compute derivatives of circuits.

    CircuitGradient - uses quantum techniques to get derivatives of circuits
    DerivativeBase - uses classical techniques to differentiate opflow data structures
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()

    @abstractmethod
    def convert(self, operator: OperatorBase, params: Optional[Union[ParameterVector, ParameterExpression, List[ParameterExpression]]]=None) -> OperatorBase:
        if False:
            while True:
                i = 10
        '\n        Args:\n            operator: The operator we are taking the gradient, Hessian or QFI of\n            params: The parameters we are taking the gradient, Hessian or QFI with respect to.\n\n        Returns:\n            An operator whose evaluation yields the gradient, Hessian or QFI.\n\n        Raises:\n            ValueError: If ``params`` contains a parameter not present in ``operator``.\n        '
        raise NotImplementedError

    def gradient_wrapper(self, operator: OperatorBase, bind_params: Union[ParameterExpression, ParameterVector, List[ParameterExpression]], grad_params: Optional[Union[ParameterExpression, ParameterVector, List[ParameterExpression], Tuple[ParameterExpression, ParameterExpression], List[Tuple[ParameterExpression, ParameterExpression]]]]=None, backend: Optional[Union[Backend, QuantumInstance]]=None, expectation: Optional[ExpectationBase]=None) -> Callable[[Iterable], np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        'Get a callable function which provides the respective gradient, Hessian or QFI for given\n        parameter values. This callable can be used as gradient function for optimizers.\n\n        Args:\n            operator: The operator for which we want to get the gradient, Hessian or QFI.\n            bind_params: The operator parameters to which the parameter values are assigned.\n            grad_params: The parameters with respect to which we are taking the gradient, Hessian\n                or QFI. If grad_params = None, then grad_params = bind_params\n            backend: The quantum backend or QuantumInstance to use to evaluate the gradient,\n                Hessian or QFI.\n            expectation: The expectation converter to be used. If none is set then\n                `PauliExpectation()` is used.\n\n        Returns:\n            Function to compute a gradient, Hessian or QFI. The function\n            takes an iterable as argument which holds the parameter values.\n        '
        from ..converters import CircuitSampler
        if grad_params is None:
            grad_params = bind_params
        grad = self.convert(operator, grad_params)
        if expectation is None:
            expectation = PauliExpectation()
        grad = expectation.convert(grad)
        sampler = CircuitSampler(backend=backend) if backend is not None else None

        def gradient_fn(p_values):
            if False:
                for i in range(10):
                    print('nop')
            p_values_dict = dict(zip(bind_params, p_values))
            if not backend:
                converter = grad.assign_parameters(p_values_dict)
                return np.real(converter.eval())
            else:
                p_values_list = {k: [v] for (k, v) in p_values_dict.items()}
                sampled = sampler.convert(grad, p_values_list)
                fully_bound = sampled.bind_parameters(p_values_dict)
                return np.real(fully_bound.eval()[0])
        return gradient_fn

    @staticmethod
    @deprecate_func(since='0.18.0', package_name='qiskit-terra', additional_msg='Instead, use the ParameterExpression.gradient method.')
    def parameter_expression_grad(param_expr: ParameterExpression, param: ParameterExpression) -> Union[ParameterExpression, float]:
        if False:
            i = 10
            return i + 15
        'Get the derivative of a parameter expression w.r.t. the given parameter.\n\n        Args:\n            param_expr: The Parameter Expression for which we compute the derivative\n            param: Parameter w.r.t. which we want to take the derivative\n\n        Returns:\n            ParameterExpression representing the gradient of param_expr w.r.t. param\n        '
        return _coeff_derivative(param_expr, param)

    @classmethod
    def _erase_operator_coeffs(cls, operator: OperatorBase) -> OperatorBase:
        if False:
            return 10
        'This method traverses an input operator and deletes all of the coefficients\n\n        Args:\n            operator: An operator type object.\n\n        Returns:\n            An operator which is equal to the input operator but whose coefficients\n            have all been set to 1.0\n\n        Raises:\n            TypeError: If unknown operator type is reached.\n        '
        if isinstance(operator, PrimitiveOp):
            return operator / operator.coeff
        op_coeff = operator.coeff
        return (operator / op_coeff).traverse(cls._erase_operator_coeffs)

    @classmethod
    def _factor_coeffs_out_of_composed_op(cls, operator: OperatorBase) -> OperatorBase:
        if False:
            print('Hello World!')
        "Factor all coefficients of ComposedOp out into a single global coefficient.\n\n        Part of the automatic differentiation logic inside of Gradient and Hessian\n        counts on the fact that no product or chain rules need to be computed between\n        operators or coefficients within a ComposedOp. To ensure this condition is met,\n        this function traverses an operator and replaces each ComposedOp with an equivalent\n        ComposedOp, but where all coefficients have been factored out and placed onto the\n        ComposedOp. Note that this cannot be done properly if an OperatorMeasurement contains\n        a SummedOp as it's primitive.\n\n        Args:\n            operator: The operator whose coefficients are being re-organized\n\n        Returns:\n            An operator equivalent to the input operator, but whose coefficients have been\n            reorganized\n\n        Raises:\n            ValueError: If an element within a ComposedOp has a primitive of type ListOp,\n                        then it is not possible to factor all coefficients out of the ComposedOp.\n        "
        if isinstance(operator, ListOp) and (not isinstance(operator, ComposedOp)):
            return operator.traverse(cls._factor_coeffs_out_of_composed_op)
        if isinstance(operator, ComposedOp):
            total_coeff = operator.coeff
            take_norm_of_coeffs = False
            for (k, op) in enumerate(operator.oplist):
                if take_norm_of_coeffs:
                    total_coeff *= op.coeff * np.conj(op.coeff)
                else:
                    total_coeff *= op.coeff
                if hasattr(op, 'primitive'):
                    prim = op.primitive
                    if isinstance(op, StateFn) and isinstance(prim, TensoredOp):
                        for prim_op in prim.oplist:
                            if isinstance(prim_op.coeff, ParameterExpression):
                                prim_tensored = StateFn(prim.reduce(), is_measurement=op.is_measurement, coeff=op.coeff)
                                operator.oplist[k] = prim_tensored
                                return operator.traverse(cls._factor_coeffs_out_of_composed_op)
                    elif isinstance(prim, ListOp):
                        raise ValueError('This operator was not properly decomposed. By this point, all operator measurements should contain single operators, otherwise the coefficient gradients will not be handled properly.')
                    if hasattr(prim, 'coeff'):
                        if take_norm_of_coeffs:
                            total_coeff *= prim._coeff * np.conj(prim._coeff)
                        else:
                            total_coeff *= prim._coeff
                if isinstance(op, OperatorStateFn) and op.is_measurement:
                    take_norm_of_coeffs = True
            return cls._erase_operator_coeffs(operator).mul(total_coeff)
        else:
            return operator

def _coeff_derivative(coeff, param):
    if False:
        return 10
    if isinstance(coeff, ParameterExpression) and len(coeff.parameters) > 0:
        return coeff.gradient(param)
    return 0