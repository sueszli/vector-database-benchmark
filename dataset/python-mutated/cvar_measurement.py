"""CVaRMeasurement class."""
from typing import Callable, Optional, Tuple, Union, cast, Dict
import numpy as np
from qiskit.circuit import ParameterExpression
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops import ListOp, SummedOp, TensoredOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops import PauliOp, PauliSumOp
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.opflow.state_fns.dict_state_fn import DictStateFn
from qiskit.opflow.state_fns.operator_state_fn import OperatorStateFn
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.opflow.state_fns.vector_state_fn import VectorStateFn
from qiskit.quantum_info import Statevector
from qiskit.utils.deprecation import deprecate_func

class CVaRMeasurement(OperatorStateFn):
    """Deprecated: A specialized measurement class to compute CVaR expectation values.
        See https://arxiv.org/pdf/1907.04769.pdf for further details.

    Used in :class:`~qiskit.opflow.CVaRExpectation`, see there for more details.
    """
    primitive: OperatorBase

    @deprecate_func(since='0.24.0', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, primitive: OperatorBase=None, alpha: float=1.0, coeff: Union[complex, ParameterExpression]=1.0) -> None:
        if False:
            while True:
                i = 10
        '\n        Args:\n            primitive: The ``OperatorBase`` which defines the diagonal operator\n                       measurement.\n            coeff: A coefficient by which to multiply the state function\n            alpha: A real-valued parameter between 0 and 1 which specifies the\n                   fraction of observed samples to include when computing the\n                   objective value. alpha = 1 corresponds to a standard observable\n                   expectation value. alpha = 0 corresponds to only using the single\n                   sample with the lowest energy. alpha = 0.5 corresponds to ranking each\n                   observation by lowest energy and using the best\n\n        Raises:\n            ValueError: TODO remove that this raises an error\n            ValueError: If alpha is not in [0, 1].\n            OpflowError: If the primitive is not diagonal.\n        '
        if primitive is None:
            raise ValueError
        if not 0 <= alpha <= 1:
            raise ValueError('The parameter alpha must be in [0, 1].')
        self._alpha = alpha
        if not _check_is_diagonal(primitive):
            raise OpflowError('Input operator to CVaRMeasurement must be diagonal, but is not:', str(primitive))
        super().__init__(primitive, coeff=coeff, is_measurement=True)

    @property
    def alpha(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        'A real-valued parameter between 0 and 1 which specifies the\n           fraction of observed samples to include when computing the\n           objective value. alpha = 1 corresponds to a standard observable\n           expectation value. alpha = 0 corresponds to only using the single\n           sample with the lowest energy. alpha = 0.5 corresponds to ranking each\n           observation by lowest energy and using the best half.\n\n        Returns:\n            The parameter alpha which was given at initialization\n        '
        return self._alpha

    @property
    def settings(self) -> Dict:
        if False:
            print('Hello World!')
        'Return settings.'
        return {'primitive': self._primitive, 'coeff': self._coeff, 'alpha': self._alpha}

    def add(self, other: OperatorBase) -> SummedOp:
        if False:
            for i in range(10):
                print('nop')
        return SummedOp([self, other])

    def adjoint(self):
        if False:
            print('Hello World!')
        'The adjoint of a CVaRMeasurement is not defined.\n\n        Returns:\n            Does not return anything, raises an error.\n\n        Raises:\n            OpflowError: The adjoint of a CVaRMeasurement is not defined.\n        '
        raise OpflowError('Adjoint of a CVaR measurement not defined')

    def mul(self, scalar: Union[complex, ParameterExpression]) -> 'CVaRMeasurement':
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not {} of type {}.'.format(scalar, type(scalar)))
        return self.__class__(self.primitive, coeff=self.coeff * scalar, alpha=self._alpha)

    def tensor(self, other: OperatorBase) -> Union['OperatorStateFn', TensoredOp]:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, OperatorStateFn):
            return OperatorStateFn(self.primitive.tensor(other.primitive), coeff=self.coeff * other.coeff)
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool=False):
        if False:
            for i in range(10):
                print('nop')
        'Not defined.'
        raise NotImplementedError

    def to_matrix_op(self, massive: bool=False):
        if False:
            print('Hello World!')
        'Not defined.'
        raise NotImplementedError

    def to_matrix(self, massive: bool=False):
        if False:
            return 10
        'Not defined.'
        raise NotImplementedError

    def to_circuit_op(self):
        if False:
            print('Hello World!')
        'Not defined.'
        raise NotImplementedError

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'CVaRMeasurement({str(self.primitive)}) * {self.coeff}'

    def eval(self, front: Union[str, dict, np.ndarray, OperatorBase, Statevector]=None) -> complex:
        if False:
            while True:
                i = 10
        '\n        Given the energies of each sampled measurement outcome (H_i) as well as the\n        sampling probability of each measurement outcome (p_i, we can compute the\n        CVaR as H_j + 1/α*(sum_i<j p_i*(H_i - H_j)). Note that index j corresponds\n        to the measurement outcome such that only some of the samples with\n        measurement outcome j will be used in computing CVaR. Note also that the\n        sampling probabilities serve as an alternative to knowing the counts of each\n        observation.\n\n        This computation is broken up into two subroutines. One which evaluates each\n        measurement outcome and determines the sampling probabilities of each. And one\n        which carries out the above calculation. The computation is split up this way\n        to enable a straightforward calculation of the variance of this estimator.\n\n        Args:\n            front: A StateFn or primitive which specifies the results of evaluating\n                      a quantum state.\n\n        Returns:\n            The CVaR of the diagonal observable specified by self.primitive and\n                the sampled quantum state described by the inputs\n                (energies, probabilities). For index j (described above), the CVaR\n                is computed as H_j + 1/α*(sum_i<j p_i*(H_i - H_j))\n        '
        (energies, probabilities) = self.get_outcome_energies_probabilities(front)
        return self.compute_cvar(energies, probabilities)

    def eval_variance(self, front: Optional[Union[str, dict, np.ndarray, OperatorBase]]=None) -> complex:
        if False:
            i = 10
            return i + 15
        '\n        Given the energies of each sampled measurement outcome (H_i) as well as the\n        sampling probability of each measurement outcome (p_i, we can compute the\n        variance of the CVaR estimator as\n        H_j^2 + 1/α * (sum_i<j p_i*(H_i^2 - H_j^2)).\n        This follows from the definition that Var[X] = E[X^2] - E[X]^2.\n        In this case, X = E[<bi|H|bi>], where H is the diagonal observable and bi\n        corresponds to measurement outcome i. Given this, E[X^2] = E[<bi|H|bi>^2]\n\n        Args:\n            front: A StateFn or primitive which specifies the results of evaluating\n                      a quantum state.\n\n        Returns:\n            The Var[CVaR] of the diagonal observable specified by self.primitive\n                and the sampled quantum state described by the inputs\n                (energies, probabilities). For index j (described above), the CVaR\n                is computed as H_j^2 + 1/α*(sum_i<j p_i*(H_i^2 - H_j^2))\n        '
        (energies, probabilities) = self.get_outcome_energies_probabilities(front)
        sq_energies = [energy ** 2 for energy in energies]
        return self.compute_cvar(sq_energies, probabilities) - self.eval(front) ** 2

    def get_outcome_energies_probabilities(self, front: Optional[Union[str, dict, np.ndarray, OperatorBase, Statevector]]=None) -> Tuple[list, list]:
        if False:
            i = 10
            return i + 15
        "\n        In order to compute the  CVaR of an observable expectation, we require\n        the energies of each sampled measurement outcome as well as the sampling\n        probability of each measurement outcome. Note that the counts for each\n        measurement outcome will also suffice (and this is often how the CVaR\n        is presented).\n\n        Args:\n            front: A StateFn or a primitive which defines a StateFn.\n                   This input holds the results of a sampled/simulated circuit.\n\n        Returns:\n            Two lists of equal length. `energies` contains the energy of each\n                unique measurement outcome computed against the diagonal observable\n                stored in self.primitive. `probabilities` contains the corresponding\n                sampling probability for each measurement outcome in `energies`.\n\n        Raises:\n            ValueError: front isn't a DictStateFn or VectorStateFn\n        "
        if isinstance(front, CircuitStateFn):
            front = cast(StateFn, front.eval())
        if isinstance(front, DictStateFn):
            data = front.primitive
        elif isinstance(front, VectorStateFn):
            vec = front.primitive.data
            key_len = int(np.ceil(np.log2(len(vec))))
            data = {format(index, '0' + str(key_len) + 'b'): val for (index, val) in enumerate(vec)}
        else:
            raise ValueError('Unsupported input to CVaRMeasurement.eval:', type(front))
        obs = self.primitive
        outcomes = list(data.items())
        for (i, outcome) in enumerate(outcomes):
            key = outcome[0]
            outcomes[i] += (obs.eval(key).adjoint().eval(key),)
        outcomes = sorted(outcomes, key=lambda x: x[2])
        (_, root_probabilities, energies) = zip(*outcomes)
        probabilities = [p_i * np.conj(p_i) for p_i in root_probabilities]
        return (list(energies), probabilities)

    def compute_cvar(self, energies: list, probabilities: list) -> complex:
        if False:
            print('Hello World!')
        "\n        Given the energies of each sampled measurement outcome (H_i) as well as the\n        sampling probability of each measurement outcome (p_i, we can compute the\n        CVaR. Note that the sampling probabilities serve as an alternative to knowing\n        the counts of each observation and that the input energies are assumed to be\n        sorted in increasing order.\n\n        Consider the outcome with index j, such that only some of the samples with\n        measurement outcome j will be used in computing CVaR. The CVaR calculation\n        can then be separated into two parts. First we sum each of the energies for\n        outcomes i < j, weighted by the probability of observing that outcome (i.e\n        the normalized counts). Second, we add the energy for outcome j, weighted by\n        the difference (α  - \\sum_i<j p_i)\n\n        Args:\n            energies: A list containing the energies (H_i) of each sample measurement\n                      outcome, sorted in increasing order.\n            probabilities: The sampling probabilities (p_i) for each corresponding\n                           measurement outcome.\n\n        Returns:\n            The CVaR of the diagonal observable specified by self.primitive and\n                the sampled quantum state described by the inputs\n                (energies, probabilities). For index j (described above), the CVaR\n                is computed as H_j + 1/α * (sum_i<j p_i*(H_i - H_j))\n\n        Raises:\n            ValueError: front isn't a DictStateFn or VectorStateFn\n        "
        alpha = self._alpha
        j = 0
        running_total = 0
        for (i, p_i) in enumerate(probabilities):
            running_total += p_i
            j = i
            if running_total > alpha:
                break
        h_j = energies[j]
        cvar = alpha * h_j
        if alpha == 0 or j == 0:
            return self.coeff * h_j
        energies = energies[:j]
        probabilities = probabilities[:j]
        for (h_i, p_i) in zip(energies, probabilities):
            cvar += p_i * (h_i - h_j)
        return self.coeff * cvar / alpha

    def traverse(self, convert_fn: Callable, coeff: Optional[Union[complex, ParameterExpression]]=None) -> OperatorBase:
        if False:
            print('Hello World!')
        '\n        Apply the convert_fn to the internal primitive if the primitive is an Operator (as in\n        the case of ``OperatorStateFn``). Otherwise do nothing. Used by converters.\n\n        Args:\n            convert_fn: The function to apply to the internal OperatorBase.\n            coeff: A coefficient to multiply by after applying convert_fn.\n                If it is None, self.coeff is used instead.\n\n        Returns:\n            The converted StateFn.\n        '
        if coeff is None:
            coeff = self.coeff
        if isinstance(self.primitive, OperatorBase):
            return self.__class__(convert_fn(self.primitive), coeff=coeff, alpha=self._alpha)
        return self

    def sample(self, shots: int=1024, massive: bool=False, reverse_endianness: bool=False):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

def _check_is_diagonal(operator: OperatorBase) -> bool:
    if False:
        return 10
    'Check whether ``operator`` is diagonal.\n\n    Args:\n        operator: The operator to check for diagonality.\n\n    Returns:\n        True, if the operator is diagonal, False otherwise.\n\n    Raises:\n        OpflowError: If the operator is not diagonal.\n    '
    if isinstance(operator, PauliOp):
        return not np.any(operator.primitive.x)
    if isinstance(operator, PauliSumOp):
        if not np.any(operator.primitive.paulis.x):
            return True
    elif isinstance(operator, SummedOp):
        if all((isinstance(op, PauliOp) and (not np.any(op.primitive.x)) for op in operator.oplist)):
            return True
    elif isinstance(operator, ListOp):
        return all(operator.traverse(_check_is_diagonal))
    matrix = operator.to_matrix()
    return np.all(matrix == np.diag(np.diagonal(matrix)))