"""
Readout mitigator class based on the A-matrix inversion method
"""
from typing import Optional, List, Tuple, Iterable, Callable, Union, Dict
import numpy as np
from qiskit.exceptions import QiskitError
from ..distributions.quasi import QuasiDistribution
from ..counts import Counts
from .base_readout_mitigator import BaseReadoutMitigator
from .utils import counts_probability_vector, z_diagonal, str2diag

class CorrelatedReadoutMitigator(BaseReadoutMitigator):
    """N-qubit readout error mitigator.

    Mitigates :meth:`expectation_value` and :meth:`quasi_probabilities`.
    The mitigation_matrix should be calibrated using qiskit experiments.
    This mitigation method should be used in case the readout errors of the qubits
    are assumed to be correlated. The mitigation_matrix of *N* qubits is of size
    :math:`2^N x 2^N` so the mitigation complexity is :math:`O(4^N)`.
    """

    def __init__(self, assignment_matrix: np.ndarray, qubits: Optional[Iterable[int]]=None):
        if False:
            return 10
        'Initialize a CorrelatedReadoutMitigator\n\n        Args:\n            assignment_matrix: readout error assignment matrix.\n            qubits: Optional, the measured physical qubits for mitigation.\n\n        Raises:\n            QiskitError: matrix size does not agree with number of qubits\n        '
        if np.any(assignment_matrix < 0) or not np.allclose(np.sum(assignment_matrix, axis=0), 1):
            raise QiskitError('Assignment matrix columns must be valid probability distributions')
        assignment_matrix = np.asarray(assignment_matrix, dtype=float)
        matrix_qubits_num = int(np.log2(assignment_matrix.shape[0]))
        if qubits is None:
            self._num_qubits = matrix_qubits_num
            self._qubits = range(self._num_qubits)
        else:
            if len(qubits) != matrix_qubits_num:
                raise QiskitError('The number of given qubits ({}) is different than the number of qubits inferred from the matrices ({})'.format(len(qubits), matrix_qubits_num))
            self._qubits = qubits
            self._num_qubits = len(self._qubits)
        self._qubit_index = dict(zip(self._qubits, range(self._num_qubits)))
        self._assignment_mat = assignment_matrix
        self._mitigation_mats = {}

    @property
    def settings(self) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        'Return settings.'
        return {'assignment_matrix': self._assignment_mat, 'qubits': self._qubits}

    def expectation_value(self, data: Counts, diagonal: Union[Callable, dict, str, np.ndarray]=None, qubits: Iterable[int]=None, clbits: Optional[List[int]]=None, shots: Optional[int]=None) -> Tuple[float, float]:
        if False:
            while True:
                i = 10
        'Compute the mitigated expectation value of a diagonal observable.\n\n        This computes the mitigated estimator of\n        :math:`\\langle O \\rangle = \\mbox{Tr}[\\rho. O]` of a diagonal observable\n        :math:`O = \\sum_{x\\in\\{0, 1\\}^n} O(x)|x\\rangle\\!\\langle x|`.\n\n        Args:\n            data: Counts object\n            diagonal: Optional, the vector of diagonal values for summing the\n                      expectation value. If ``None`` the default value is\n                      :math:`[1, -1]^\\otimes n`.\n            qubits: Optional, the measured physical qubits the count\n                    bitstrings correspond to. If None qubits are assumed to be\n                    :math:`[0, ..., n-1]`.\n            clbits: Optional, if not None marginalize counts to the specified bits.\n            shots: the number of shots.\n\n        Returns:\n            (float, float): the expectation value and an upper bound of the standard deviation.\n\n        Additional Information:\n            The diagonal observable :math:`O` is input using the ``diagonal`` kwarg as\n            a list or Numpy array :math:`[O(0), ..., O(2^n -1)]`. If no diagonal is specified\n            the diagonal of the Pauli operator\n            :math`O = \\mbox{diag}(Z^{\\otimes n}) = [1, -1]^{\\otimes n}` is used.\n            The ``clbits`` kwarg is used to marginalize the input counts dictionary\n            over the specified bit-values, and the ``qubits`` kwarg is used to specify\n            which physical qubits these bit-values correspond to as\n            ``circuit.measure(qubits, clbits)``.\n        '
        if qubits is None:
            qubits = self._qubits
        (probs_vec, shots) = counts_probability_vector(data, qubit_index=self._qubit_index, clbits=clbits, qubits=qubits)
        mit_mat = self.mitigation_matrix(qubits)
        if diagonal is None:
            diagonal = z_diagonal(2 ** self._num_qubits)
        elif isinstance(diagonal, str):
            diagonal = str2diag(diagonal)
        coeffs = mit_mat.T.dot(diagonal)
        expval = coeffs.dot(probs_vec)
        stddev_upper_bound = self.stddev_upper_bound(shots)
        return (expval, stddev_upper_bound)

    def quasi_probabilities(self, data: Counts, qubits: Optional[List[int]]=None, clbits: Optional[List[int]]=None, shots: Optional[int]=None) -> QuasiDistribution:
        if False:
            while True:
                i = 10
        'Compute mitigated quasi probabilities value.\n\n        Args:\n            data: counts object\n            qubits: qubits the count bitstrings correspond to.\n            clbits: Optional, marginalize counts to just these bits.\n            shots: Optional, the total number of shots, if None shots will\n                be calculated as the sum of all counts.\n\n        Returns:\n            QuasiDistribution: A dictionary containing pairs of [output, mean] where "output"\n                is the key in the dictionaries,\n                which is the length-N bitstring of a measured standard basis state,\n                and "mean" is the mean of non-zero quasi-probability estimates.\n        '
        if qubits is None:
            qubits = self._qubits
        (probs_vec, calculated_shots) = counts_probability_vector(data, qubit_index=self._qubit_index, clbits=clbits, qubits=qubits)
        if shots is None:
            shots = calculated_shots
        mit_mat = self.mitigation_matrix(qubits)
        probs_vec = mit_mat.dot(probs_vec)
        probs_dict = {}
        for (index, _) in enumerate(probs_vec):
            probs_dict[index] = probs_vec[index]
        quasi_dist = QuasiDistribution(probs_dict, stddev_upper_bound=self.stddev_upper_bound(shots))
        return quasi_dist

    def mitigation_matrix(self, qubits: List[int]=None) -> np.ndarray:
        if False:
            print('Hello World!')
        'Return the readout mitigation matrix for the specified qubits.\n\n        The mitigation matrix :math:`A^{-1}` is defined as the inverse of the\n        :meth:`assignment_matrix` :math:`A`.\n\n        Args:\n            qubits: Optional, qubits being measured.\n\n        Returns:\n            np.ndarray: the measurement error mitigation matrix :math:`A^{-1}`.\n        '
        if qubits is None:
            qubits = self._qubits
        qubits = tuple(sorted(qubits))
        if qubits not in self._mitigation_mats:
            marginal_matrix = self.assignment_matrix(qubits)
            try:
                mit_mat = np.linalg.inv(marginal_matrix)
            except np.linalg.LinAlgError:
                mit_mat = np.linalg.pinv(marginal_matrix)
            self._mitigation_mats[qubits] = mit_mat
        return self._mitigation_mats[qubits]

    def assignment_matrix(self, qubits: List[int]=None) -> np.ndarray:
        if False:
            return 10
        'Return the readout assignment matrix for specified qubits.\n\n        The assignment matrix is the stochastic matrix :math:`A` which assigns\n        a noisy readout probability distribution to an ideal input\n        readout distribution: :math:`P(i|j) = \\langle i | A | j \\rangle`.\n\n        Args:\n            qubits: Optional, qubits being measured.\n\n        Returns:\n            np.ndarray: the assignment matrix A.\n        '
        if qubits is None:
            qubits = self._qubits
        if qubits == self._num_qubits:
            return self._assignment_mat
        if isinstance(qubits, int):
            qubits = [qubits]
        qubit_indices = [self._qubit_index[qubit] for qubit in qubits]
        axis = tuple((self._num_qubits - 1 - i for i in set(range(self._num_qubits)).difference(qubit_indices)))
        num_qubits = len(qubits)
        new_amat = np.zeros(2 * [2 ** num_qubits], dtype=float)
        for (i, col) in enumerate(self._assignment_mat.T[self._keep_indexes(qubit_indices)]):
            new_amat[i] = np.reshape(col, self._num_qubits * [2]).sum(axis=axis).reshape([2 ** num_qubits])
        new_amat = new_amat.T
        return new_amat

    @staticmethod
    def _keep_indexes(qubits):
        if False:
            return 10
        indexes = [0]
        for i in sorted(qubits):
            indexes += [idx + (1 << i) for idx in indexes]
        return indexes

    def _compute_gamma(self):
        if False:
            return 10
        'Compute gamma for N-qubit mitigation'
        mitmat = self.mitigation_matrix(qubits=self._qubits)
        return np.max(np.sum(np.abs(mitmat), axis=0))

    def stddev_upper_bound(self, shots: int):
        if False:
            for i in range(10):
                print('nop')
        'Return an upper bound on standard deviation of expval estimator.\n\n        Args:\n            shots: Number of shots used for expectation value measurement.\n\n        Returns:\n            float: the standard deviation upper bound.\n        '
        gamma = self._compute_gamma()
        return gamma / np.sqrt(shots)

    @property
    def qubits(self) -> Tuple[int]:
        if False:
            for i in range(10):
                print('nop')
        'The device qubits for this mitigator'
        return self._qubits