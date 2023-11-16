"""Trotterization Algorithm Base"""
from abc import abstractmethod
from qiskit.opflow.evolutions.evolution_base import EvolutionBase
from qiskit.opflow.operator_base import OperatorBase
from qiskit.utils.deprecation import deprecate_func

class TrotterizationBase(EvolutionBase):
    """Deprecated: A base for Trotterization methods, algorithms for approximating exponentiations of
    operator sums by compositions of exponentiations.
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, reps: int=1) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self._reps = reps

    @property
    def reps(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'The number of repetitions to use in the Trotterization, improving the approximation\n        accuracy.\n        '
        return self._reps

    @reps.setter
    def reps(self, reps: int) -> None:
        if False:
            i = 10
            return i + 15
        'Set the number of repetitions to use in the Trotterization.'
        self._reps = reps

    @abstractmethod
    def convert(self, operator: OperatorBase) -> OperatorBase:
        if False:
            for i in range(10):
                print('nop')
        "\n        Convert a ``SummedOp`` into a ``ComposedOp`` or ``CircuitOp`` representing an\n        approximation of e^-i*``op_sum``.\n\n        Args:\n            operator: The ``SummedOp`` to evolve.\n\n        Returns:\n            The Operator approximating op_sum's evolution.\n\n        Raises:\n            TypeError: A non-SummedOps Operator is passed into ``convert``.\n\n        "
        raise NotImplementedError