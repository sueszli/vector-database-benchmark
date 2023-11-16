"""MatrixEvolution Class"""
import logging
from qiskit.opflow.evolutions.evolution_base import EvolutionBase
from qiskit.opflow.evolutions.evolved_op import EvolvedOp
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.matrix_op import MatrixOp
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.utils.deprecation import deprecate_func
logger = logging.getLogger(__name__)

class MatrixEvolution(EvolutionBase):
    """
    Deprecated: Performs Evolution by classical matrix exponentiation, constructing a circuit with
    ``UnitaryGates`` or ``HamiltonianGates`` containing the exponentiation of the Operator.
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__()

    def convert(self, operator: OperatorBase) -> OperatorBase:
        if False:
            print('Hello World!')
        '\n        Traverse the operator, replacing ``EvolvedOps`` with ``CircuitOps`` containing\n        ``UnitaryGates`` or ``HamiltonianGates`` (if self.coeff is a ``ParameterExpression``)\n        equalling the exponentiation of -i * operator. This is done by converting the\n        ``EvolvedOp.primitive`` to a ``MatrixOp`` and simply calling ``.exp_i()`` on that.\n\n        Args:\n            operator: The Operator to convert.\n\n        Returns:\n            The converted operator.\n        '
        if isinstance(operator, EvolvedOp):
            if not {'Matrix'} == operator.primitive_strings():
                logger.warning('Evolved Hamiltonian is not composed of only MatrixOps, converting to Matrix representation, which can be expensive.')
                matrix_ham = operator.primitive.to_matrix_op(massive=False)
                operator = EvolvedOp(matrix_ham, coeff=operator.coeff)
            if isinstance(operator.primitive, ListOp):
                return operator.primitive.exp_i() * operator.coeff
            elif isinstance(operator.primitive, (MatrixOp, PauliOp)):
                return operator.primitive.exp_i()
        elif isinstance(operator, ListOp):
            return operator.traverse(self.convert).reduce()
        return operator