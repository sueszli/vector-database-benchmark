"""
Readout mitigator class based on the 1-qubit local tensored mitigation method
"""
from typing import Optional, List, Tuple, Iterable, Callable, Union, Dict
import numpy as np
from qiskit.exceptions import QiskitError
from ..distributions.quasi import QuasiDistribution
from ..counts import Counts
from .base_readout_mitigator import BaseReadoutMitigator
from .utils import counts_probability_vector, z_diagonal, str2diag

class LocalReadoutMitigator(BaseReadoutMitigator):
    """1-qubit tensor product readout error mitigator.

    Mitigates :meth:`expectation_value` and :meth:`quasi_probabilities`.
    The mitigator should either be calibrated using qiskit experiments,
    or calculated directly from the backend properties.
    This mitigation method should be used in case the readout errors of the qubits
    are assumed to be uncorrelated. For *N* qubits there are *N* mitigation matrices,
    each of size :math:`2 x 2` and the mitigation complexity is :math:`O(2^N)`,
    so it is more efficient than the :class:`CorrelatedReadoutMitigator` class.
    """

    def __init__(self, assignment_matrices: Optional[List[np.ndarray]]=None, qubits: Optional[Iterable[int]]=None, backend=None):
        if False:
            return 10
        'Initialize a LocalReadoutMitigator\n\n        Args:\n            assignment_matrices: Optional, list of single-qubit readout error assignment matrices.\n            qubits: Optional, the measured physical qubits for mitigation.\n            backend: Optional, backend name.\n\n        Raises:\n            QiskitError: matrices sizes do not agree with number of qubits\n        '
        if assignment_matrices is None:
            assignment_matrices = self._from_backend(backend, qubits)
        else:
            assignment_matrices = [np.asarray(amat, dtype=float) for amat in assignment_matrices]
        for amat in assignment_matrices:
            if np.any(amat < 0) or not np.allclose(np.sum(amat, axis=0), 1):
                raise QiskitError('Assignment matrix columns must be valid probability distributions')
        if qubits is None:
            self._num_qubits = len(assignment_matrices)
            self._qubits = range(self._num_qubits)
        else:
            if len(qubits) != len(assignment_matrices):
                raise QiskitError('The number of given qubits ({}) is different than the number of qubits inferred from the matrices ({})'.format(len(qubits), len(assignment_matrices)))
            self._qubits = qubits
            self._num_qubits = len(self._qubits)
        self._qubit_index = dict(zip(self._qubits, range(self._num_qubits)))
        self._assignment_mats = assignment_matrices
        self._mitigation_mats = np.zeros([self._num_qubits, 2, 2], dtype=float)
        self._gammas = np.zeros(self._num_qubits, dtype=float)
        for i in range(self._num_qubits):
            mat = self._assignment_mats[i]
            error0 = mat[1, 0]
            error1 = mat[0, 1]
            self._gammas[i] = (1 + abs(error0 - error1)) / (1 - error0 - error1)
            try:
                ainv = np.linalg.inv(mat)
            except np.linalg.LinAlgError:
                ainv = np.linalg.pinv(mat)
            self._mitigation_mats[i] = ainv

    @property
    def settings(self) -> Dict:
        if False:
            while True:
                i = 10
        'Return settings.'
        return {'assignment_matrices': self._assignment_mats, 'qubits': self._qubits}

    def expectation_value(self, data: Counts, diagonal: Union[Callable, dict, str, np.ndarray]=None, qubits: Iterable[int]=None, clbits: Optional[List[int]]=None, shots: Optional[int]=None) -> Tuple[float, float]:
        if False:
            i = 10
            return i + 15
        'Compute the mitigated expectation value of a diagonal observable.\n\n        This computes the mitigated estimator of\n        :math:`\\langle O \\rangle = \\mbox{Tr}[\\rho. O]` of a diagonal observable\n        :math:`O = \\sum_{x\\in\\{0, 1\\}^n} O(x)|x\\rangle\\!\\langle x|`.\n\n        Args:\n            data: Counts object\n            diagonal: Optional, the vector of diagonal values for summing the\n                      expectation value. If ``None`` the default value is\n                      :math:`[1, -1]^\\otimes n`.\n            qubits: Optional, the measured physical qubits the count\n                    bitstrings correspond to. If None qubits are assumed to be\n                    :math:`[0, ..., n-1]`.\n            clbits: Optional, if not None marginalize counts to the specified bits.\n            shots: the number of shots.\n\n        Returns:\n            (float, float): the expectation value and an upper bound of the standard deviation.\n\n        Additional Information:\n            The diagonal observable :math:`O` is input using the ``diagonal`` kwarg as\n            a list or Numpy array :math:`[O(0), ..., O(2^n -1)]`. If no diagonal is specified\n            the diagonal of the Pauli operator\n            :math`O = \\mbox{diag}(Z^{\\otimes n}) = [1, -1]^{\\otimes n}` is used.\n            The ``clbits`` kwarg is used to marginalize the input counts dictionary\n            over the specified bit-values, and the ``qubits`` kwarg is used to specify\n            which physical qubits these bit-values correspond to as\n            ``circuit.measure(qubits, clbits)``.\n        '
        if qubits is None:
            qubits = self._qubits
        num_qubits = len(qubits)
        (probs_vec, shots) = counts_probability_vector(data, qubit_index=self._qubit_index, clbits=clbits, qubits=qubits)
        qubit_indices = [self._qubit_index[qubit] for qubit in qubits]
        ainvs = self._mitigation_mats[qubit_indices]
        if diagonal is None:
            diagonal = z_diagonal(2 ** num_qubits)
        elif isinstance(diagonal, str):
            diagonal = str2diag(diagonal)
        coeffs = np.reshape(diagonal, num_qubits * [2])
        einsum_args = [coeffs, list(range(num_qubits))]
        for (i, ainv) in enumerate(reversed(ainvs)):
            einsum_args += [ainv.T, [num_qubits + i, i]]
        einsum_args += [list(range(num_qubits, 2 * num_qubits))]
        coeffs = np.einsum(*einsum_args).ravel()
        expval = coeffs.dot(probs_vec)
        stddev_upper_bound = self.stddev_upper_bound(shots, qubits)
        return (expval, stddev_upper_bound)

    def quasi_probabilities(self, data: Counts, qubits: Optional[List[int]]=None, clbits: Optional[List[int]]=None, shots: Optional[int]=None) -> QuasiDistribution:
        if False:
            print('Hello World!')
        'Compute mitigated quasi probabilities value.\n\n        Args:\n            data: counts object\n            qubits: qubits the count bitstrings correspond to.\n            clbits: Optional, marginalize counts to just these bits.\n            shots: Optional, the total number of shots, if None shots will\n                be calculated as the sum of all counts.\n\n        Returns:\n            QuasiDistribution: A dictionary containing pairs of [output, mean] where "output"\n                is the key in the dictionaries,\n                which is the length-N bitstring of a measured standard basis state,\n                and "mean" is the mean of non-zero quasi-probability estimates.\n\n        Raises:\n            QiskitError: if qubit and clbit kwargs are not valid.\n        '
        if qubits is None:
            qubits = self._qubits
        num_qubits = len(qubits)
        (probs_vec, calculated_shots) = counts_probability_vector(data, qubit_index=self._qubit_index, clbits=clbits, qubits=qubits)
        if shots is None:
            shots = calculated_shots
        qubit_indices = [self._qubit_index[qubit] for qubit in qubits]
        ainvs = self._mitigation_mats[qubit_indices]
        prob_tens = np.reshape(probs_vec, num_qubits * [2])
        einsum_args = [prob_tens, list(range(num_qubits))]
        for (i, ainv) in enumerate(reversed(ainvs)):
            einsum_args += [ainv, [num_qubits + i, i]]
        einsum_args += [list(range(num_qubits, 2 * num_qubits))]
        probs_vec = np.einsum(*einsum_args).ravel()
        probs_dict = {}
        for (index, _) in enumerate(probs_vec):
            probs_dict[index] = probs_vec[index]
        quasi_dist = QuasiDistribution(probs_dict, shots=shots, stddev_upper_bound=self.stddev_upper_bound(shots, qubits))
        return quasi_dist

    def mitigation_matrix(self, qubits: Optional[Union[List[int], int]]=None) -> np.ndarray:
        if False:
            print('Hello World!')
        'Return the measurement mitigation matrix for the specified qubits.\n\n        The mitigation matrix :math:`A^{-1}` is defined as the inverse of the\n        :meth:`assignment_matrix` :math:`A`.\n\n        Args:\n            qubits: Optional, qubits being measured for operator expval.\n                    if a single int is given, it is assumed to be the index\n                    of the qubit in self._qubits\n\n        Returns:\n            np.ndarray: the measurement error mitigation matrix :math:`A^{-1}`.\n        '
        if qubits is None:
            qubits = self._qubits
        if isinstance(qubits, int):
            qubits = [self._qubits[qubits]]
        qubit_indices = [self._qubit_index[qubit] for qubit in qubits]
        mat = self._mitigation_mats[qubit_indices[0]]
        for i in qubit_indices[1:]:
            mat = np.kron(self._mitigation_mats[i], mat)
        return mat

    def assignment_matrix(self, qubits: List[int]=None) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        'Return the measurement assignment matrix for specified qubits.\n\n        The assignment matrix is the stochastic matrix :math:`A` which assigns\n        a noisy measurement probability distribution to an ideal input\n        measurement distribution: :math:`P(i|j) = \\langle i | A | j \\rangle`.\n\n        Args:\n            qubits: Optional, qubits being measured for operator expval.\n\n        Returns:\n            np.ndarray: the assignment matrix A.\n        '
        if qubits is None:
            qubits = self._qubits
        if isinstance(qubits, int):
            qubits = [qubits]
        qubit_indices = [self._qubit_index[qubit] for qubit in qubits]
        mat = self._assignment_mats[qubit_indices[0]]
        for i in qubit_indices[1:]:
            mat = np.kron(self._assignment_mats[i], mat)
        return mat

    def _compute_gamma(self, qubits=None):
        if False:
            i = 10
            return i + 15
        'Compute gamma for N-qubit mitigation'
        if qubits is None:
            gammas = self._gammas
        else:
            qubit_indices = [self._qubit_index[qubit] for qubit in qubits]
            gammas = self._gammas[qubit_indices]
        return np.prod(gammas)

    def stddev_upper_bound(self, shots: int, qubits: List[int]=None):
        if False:
            return 10
        'Return an upper bound on standard deviation of expval estimator.\n\n        Args:\n            shots: Number of shots used for expectation value measurement.\n            qubits: qubits being measured for operator expval.\n\n        Returns:\n            float: the standard deviation upper bound.\n        '
        gamma = self._compute_gamma(qubits=qubits)
        return gamma / np.sqrt(shots)

    def _from_backend(self, backend, qubits):
        if False:
            i = 10
            return i + 15
        'Calculates amats from backend properties readout_error'
        backend_qubits = backend.properties().qubits
        if qubits is not None:
            if any((qubit >= len(backend_qubits) for qubit in qubits)):
                raise QiskitError('The chosen backend does not contain the specified qubits.')
            reduced_backend_qubits = [backend_qubits[i] for i in qubits]
            backend_qubits = reduced_backend_qubits
        num_qubits = len(backend_qubits)
        amats = np.zeros([num_qubits, 2, 2], dtype=float)
        for (qubit_idx, qubit_prop) in enumerate(backend_qubits):
            for prop in qubit_prop:
                if prop.name == 'prob_meas0_prep1':
                    amats[qubit_idx][0, 1] = prop.value
                    amats[qubit_idx][1, 1] = 1 - prop.value
                if prop.name == 'prob_meas1_prep0':
                    amats[qubit_idx][1, 0] = prop.value
                    amats[qubit_idx][0, 0] = 1 - prop.value
        return amats

    @property
    def qubits(self) -> Tuple[int]:
        if False:
            print('Hello World!')
        'The device qubits for this mitigator'
        return self._qubits