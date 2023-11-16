"""CircuitQFI Class"""
from abc import abstractmethod
from typing import List, Union
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.utils.deprecation import deprecate_func
from ...converters.converter_base import ConverterBase
from ...operator_base import OperatorBase

class CircuitQFI(ConverterBase):
    """Deprecated: Circuit to Quantum Fisher Information operator converter.

    Converter for changing parameterized circuits into operators
    whose evaluation yields Quantum Fisher Information metric tensor
    with respect to the given circuit parameters

    This is distinct from DerivativeBase converters which take gradients of composite
    operators and handle things like differentiating combo_fn's and enforcing product rules
    when operator coefficients are parameterized.

    CircuitQFI - uses quantum techniques to get the QFI of circuits
    DerivativeBase - uses classical techniques to differentiate opflow data structures
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    @abstractmethod
    def convert(self, operator: OperatorBase, params: Union[ParameterExpression, ParameterVector, List[ParameterExpression]]) -> OperatorBase:
        if False:
            return 10
        '\n        Args:\n            operator: The operator corresponding to the quantum state :math:`|\\psi(\\omega)\\rangle`\n                for which we compute the QFI.\n            params: The parameters :math:`\\omega` with respect to which we are computing the QFI.\n\n        Returns:\n            An operator whose evaluation yields the QFI metric tensor.\n\n        Raises:\n            ValueError: If ``params`` contains a parameter not present in ``operator``.\n        '
        raise NotImplementedError