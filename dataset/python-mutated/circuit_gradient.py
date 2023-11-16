"""CircuitGradient Class"""
from abc import abstractmethod
from typing import List, Union, Optional, Tuple, Set
from qiskit import QuantumCircuit, QiskitError, transpile
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.utils.deprecation import deprecate_func
from ...converters.converter_base import ConverterBase
from ...operator_base import OperatorBase

class CircuitGradient(ConverterBase):
    """Deprecated: Circuit to gradient operator converter.

    Converter for changing parameterized circuits into operators
    whose evaluation yields the gradient with respect to the circuit parameters.

    This is distinct from DerivativeBase converters which take gradients of composite
    operators and handle things like differentiating combo_fn's and enforcing product rules
    when operator coefficients are parameterized.

    CircuitGradient - uses quantum techniques to get derivatives of circuits
    DerivativeBase - uses classical techniques to differentiate operator flow data structures
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    @abstractmethod
    def convert(self, operator: OperatorBase, params: Optional[Union[ParameterExpression, ParameterVector, List[ParameterExpression], Tuple[ParameterExpression, ParameterExpression], List[Tuple[ParameterExpression, ParameterExpression]]]]=None) -> OperatorBase:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            operator: The operator we are taking the gradient of\n            params: The parameters we are taking the gradient wrt: ω\n                    If a ParameterExpression, ParameterVector or List[ParameterExpression] is given,\n                    then the 1st order derivative of the operator is calculated.\n                    If a Tuple[ParameterExpression, ParameterExpression] or\n                    List[Tuple[ParameterExpression, ParameterExpression]]\n                    is given, then the 2nd order derivative of the operator is calculated.\n\n        Returns:\n            An operator whose evaluation yields the Gradient.\n\n        Raises:\n            ValueError: If ``params`` contains a parameter not present in ``operator``.\n        '
        raise NotImplementedError

    @staticmethod
    def _transpile_to_supported_operations(circuit: QuantumCircuit, supported_gates: Set[str]) -> QuantumCircuit:
        if False:
            while True:
                i = 10
        'Transpile the given circuit into a gate set for which the gradients may be computed.\n\n        Args:\n            circuit: Quantum circuit to be transpiled into supported operations.\n            supported_gates: Set of quantum operations supported by a gradient method intended to\n                            be used on the quantum circuit.\n\n        Returns:\n            Quantum circuit which is transpiled into supported operations.\n\n        Raises:\n            QiskitError: when circuit transpiling fails.\n\n        '
        unique_ops = set(circuit.count_ops())
        if not unique_ops.issubset(supported_gates):
            try:
                circuit = transpile(circuit, basis_gates=list(supported_gates), optimization_level=0)
            except Exception as exc:
                raise QiskitError(f'Could not transpile the circuit provided {circuit} into supported gates {supported_gates}.') from exc
        return circuit