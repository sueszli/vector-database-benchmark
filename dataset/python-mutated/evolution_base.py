"""EvolutionBase Class"""
from abc import ABC, abstractmethod
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.converters.converter_base import ConverterBase
from qiskit.utils.deprecation import deprecate_func

class EvolutionBase(ConverterBase, ABC):
    """
    Deprecated: A base for Evolution converters.
    Evolutions are converters which traverse an Operator tree, replacing any ``EvolvedOp`` `e`
    with a Schrodinger equation-style evolution ``CircuitOp`` equalling or approximating the
    matrix exponential of -i * the Operator contained inside (`e.primitive`). The Evolutions are
    essentially implementations of Hamiltonian Simulation algorithms, including various methods
    for Trotterization.

    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()

    @abstractmethod
    def convert(self, operator: OperatorBase) -> OperatorBase:
        if False:
            while True:
                i = 10
        'Traverse the operator, replacing any ``EvolutionOps`` with their equivalent evolution\n        ``CircuitOps``.\n\n         Args:\n             operator: The Operator to convert.\n\n        Returns:\n            The converted Operator, with ``EvolutionOps`` replaced by ``CircuitOps``.\n\n        '
        raise NotImplementedError