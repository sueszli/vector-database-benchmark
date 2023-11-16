"""Trotter Class"""
from qiskit.opflow.evolutions.trotterizations.suzuki import Suzuki
from qiskit.utils.deprecation import deprecate_func

class Trotter(Suzuki):
    """
    Deprecated: Simple Trotter expansion, composing the evolution circuits of each Operator in the sum
    together ``reps`` times and dividing the evolution time of each by ``reps``.
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, reps: int=1) -> None:
        if False:
            while True:
                i = 10
        '\n        Args:\n            reps: The number of times to repeat the Trotterization circuit.\n        '
        super().__init__(order=1, reps=reps)