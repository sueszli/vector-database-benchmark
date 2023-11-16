"""ConverterBase Class"""
from abc import ABC, abstractmethod
from qiskit.opflow.operator_base import OperatorBase
from qiskit.utils.deprecation import deprecate_func

class ConverterBase(ABC):
    """
    Deprecated: Converters take an Operator and return a new Operator, generally isomorphic
    in some way with the first, but with certain desired properties. For example,
    a converter may accept ``CircuitOp`` and return a ``SummedOp`` of
    ``PauliOps`` representing the circuit unitary. Converters may not
    have polynomial space or time scaling in their operations. On the contrary, many
    converters, such as a ``MatrixExpectation`` or ``MatrixEvolution``, which convert
    ``PauliOps`` to ``MatrixOps`` internally, will require time or space exponential
    in the number of qubits unless a clever trick is known (such as the use of sparse
    matrices)."""

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def convert(self, operator: OperatorBase) -> OperatorBase:
        if False:
            print('Hello World!')
        'Accept the Operator and return the converted Operator\n\n        Args:\n            operator: The Operator to convert.\n\n        Returns:\n            The converted Operator.\n\n        '
        raise NotImplementedError