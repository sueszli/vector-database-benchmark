"""EvolutionFactory Class"""
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.evolutions.evolution_base import EvolutionBase
from qiskit.opflow.evolutions.pauli_trotter_evolution import PauliTrotterEvolution
from qiskit.opflow.evolutions.matrix_evolution import MatrixEvolution
from qiskit.utils.deprecation import deprecate_func

class EvolutionFactory:
    """Deprecated: A factory class for convenient automatic selection of an
    Evolution algorithm based on the Operator to be converted.
    """

    @staticmethod
    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def build(operator: OperatorBase=None) -> EvolutionBase:
        if False:
            i = 10
            return i + 15
        '\n        A factory method for convenient automatic selection of an Evolution algorithm based on the\n        Operator to be converted.\n\n        Args:\n            operator: the Operator being evolved\n\n        Returns:\n            EvolutionBase: the ``EvolutionBase`` best suited to evolve operator.\n\n        Raises:\n            ValueError: If operator is not of a composition for which we know the best Evolution\n                method.\n\n        '
        primitive_strings = operator.primitive_strings()
        if 'Matrix' in primitive_strings:
            return MatrixEvolution()
        elif 'Pauli' in primitive_strings or 'SparsePauliOp' in primitive_strings:
            return PauliTrotterEvolution()
        else:
            raise ValueError('Evolutions of mixed Operators not yet supported.')