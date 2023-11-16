"""TrotterizationFactory Class"""
from qiskit.opflow.evolutions.trotterizations.qdrift import QDrift
from qiskit.opflow.evolutions.trotterizations.suzuki import Suzuki
from qiskit.opflow.evolutions.trotterizations.trotter import Trotter
from qiskit.opflow.evolutions.trotterizations.trotterization_base import TrotterizationBase
from qiskit.utils.deprecation import deprecate_func

class TrotterizationFactory:
    """Deprecated: A factory for conveniently creating TrotterizationBase instances."""

    @staticmethod
    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def build(mode: str='trotter', reps: int=1) -> TrotterizationBase:
        if False:
            return 10
        "A factory for conveniently creating TrotterizationBase instances.\n\n        Args:\n            mode: One of 'trotter', 'suzuki', 'qdrift'\n            reps: The number of times to repeat the Trotterization circuit.\n\n        Returns:\n            The desired TrotterizationBase instance.\n\n        Raises:\n            ValueError: A string not in ['trotter', 'suzuki', 'qdrift'] is given for mode.\n        "
        if mode == 'trotter':
            return Trotter(reps=reps)
        elif mode == 'suzuki':
            return Suzuki(reps=reps)
        elif mode == 'qdrift':
            return QDrift(reps=reps)
        raise ValueError(f'Trotter mode {mode} not supported')