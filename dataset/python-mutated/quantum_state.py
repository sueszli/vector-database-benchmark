"""
Abstract QuantumState class.
"""
from __future__ import annotations
import copy
from abc import abstractmethod
import numpy as np
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.result.counts import Counts

class QuantumState:
    """Abstract quantum state base class"""

    def __init__(self, op_shape: OpShape | None=None):
        if False:
            return 10
        'Initialize a QuantumState object.\n\n        Args:\n            op_shape (OpShape): Optional, an OpShape object for state dimensions.\n\n        .. note::\n\n            If `op_shape`` is specified it will take precedence over other\n            kwargs.\n        '
        self._op_shape = op_shape
        self._rng_generator = None
    __array_priority__ = 20

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, self.__class__) and self.dims() == other.dims()

    @property
    def dim(self):
        if False:
            return 10
        'Return total state dimension.'
        return self._op_shape.shape[0]

    @property
    def num_qubits(self):
        if False:
            while True:
                i = 10
        'Return the number of qubits if a N-qubit state or None otherwise.'
        return self._op_shape.num_qubits

    @property
    def _rng(self):
        if False:
            i = 10
            return i + 15
        if self._rng_generator is None:
            return np.random.default_rng()
        return self._rng_generator

    def dims(self, qargs=None):
        if False:
            i = 10
            return i + 15
        'Return tuple of input dimension for specified subsystems.'
        return self._op_shape.dims_l(qargs)

    def copy(self):
        if False:
            i = 10
            return i + 15
        'Make a copy of current operator.'
        return copy.deepcopy(self)

    def seed(self, value=None):
        if False:
            while True:
                i = 10
        'Set the seed for the quantum state RNG.'
        if value is None:
            self._rng_generator = None
        elif isinstance(value, np.random.Generator):
            self._rng_generator = value
        else:
            self._rng_generator = np.random.default_rng(value)

    @abstractmethod
    def is_valid(self, atol=None, rtol=None):
        if False:
            print('Hello World!')
        'Return True if a valid quantum state.'
        pass

    @abstractmethod
    def to_operator(self):
        if False:
            print('Hello World!')
        'Convert state to matrix operator class'
        pass

    @abstractmethod
    def conjugate(self):
        if False:
            i = 10
            return i + 15
        'Return the conjugate of the operator.'
        pass

    @abstractmethod
    def trace(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the trace of the quantum state as a density matrix.'
        pass

    @abstractmethod
    def purity(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the purity of the quantum state.'
        pass

    @abstractmethod
    def tensor(self, other: QuantumState) -> QuantumState:
        if False:
            while True:
                i = 10
        'Return the tensor product state self ⊗ other.\n\n        Args:\n            other (QuantumState): a quantum state object.\n\n        Returns:\n            QuantumState: the tensor product operator self ⊗ other.\n\n        Raises:\n            QiskitError: if other is not a quantum state.\n        '
        pass

    @abstractmethod
    def expand(self, other: QuantumState) -> QuantumState:
        if False:
            print('Hello World!')
        'Return the tensor product state other ⊗ self.\n\n        Args:\n            other (QuantumState): a quantum state object.\n\n        Returns:\n            QuantumState: the tensor product state other ⊗ self.\n\n        Raises:\n            QiskitError: if other is not a quantum state.\n        '
        pass

    def _add(self, other):
        if False:
            while True:
                i = 10
        'Return the linear combination self + other.\n\n        Args:\n            other (QuantumState): a state object.\n\n        Returns:\n            QuantumState: the linear combination self + other.\n\n        Raises:\n            NotImplementedError: if subclass does not support addition.\n        '
        raise NotImplementedError(f'{type(self)} does not support addition')

    def _multiply(self, other):
        if False:
            print('Hello World!')
        'Return the scalar multipled state other * self.\n\n        Args:\n            other (complex): a complex number.\n\n        Returns:\n            QuantumState: the scalar multipled state other * self.\n\n        Raises:\n            NotImplementedError: if subclass does not support scala\n                                 multiplication.\n        '
        raise NotImplementedError(f'{type(self)} does not support scalar multiplication')

    @abstractmethod
    def evolve(self, other: Operator | QuantumChannel, qargs: list | None=None) -> QuantumState:
        if False:
            return 10
        'Evolve a quantum state by the operator.\n\n        Args:\n            other (Operator or QuantumChannel): The operator to evolve by.\n            qargs (list): a list of QuantumState subsystem positions to apply\n                           the operator on.\n\n        Returns:\n            QuantumState: the output quantum state.\n\n        Raises:\n            QiskitError: if the operator dimension does not match the\n                         specified QuantumState subsystem dimensions.\n        '
        pass

    @abstractmethod
    def expectation_value(self, oper: BaseOperator, qargs: None | list=None) -> complex:
        if False:
            return 10
        'Compute the expectation value of an operator.\n\n        Args:\n            oper (BaseOperator): an operator to evaluate expval.\n            qargs (None or list): subsystems to apply the operator on.\n\n        Returns:\n            complex: the expectation value.\n        '
        pass

    @abstractmethod
    def probabilities(self, qargs: None | list=None, decimals: None | int=None) -> np.ndarray:
        if False:
            print('Hello World!')
        'Return the subsystem measurement probability vector.\n\n        Measurement probabilities are with respect to measurement in the\n        computation (diagonal) basis.\n\n        Args:\n            qargs (None or list): subsystems to return probabilities for,\n                if None return for all subsystems (Default: None).\n            decimals (None or int): the number of decimal places to round\n                values. If None no rounding is done (Default: None).\n\n        Returns:\n            np.array: The Numpy vector array of probabilities.\n        '
        pass

    def probabilities_dict(self, qargs: None | list=None, decimals: None | int=None) -> dict:
        if False:
            return 10
        'Return the subsystem measurement probability dictionary.\n\n        Measurement probabilities are with respect to measurement in the\n        computation (diagonal) basis.\n\n        This dictionary representation uses a Ket-like notation where the\n        dictionary keys are qudit strings for the subsystem basis vectors.\n        If any subsystem has a dimension greater than 10 comma delimiters are\n        inserted between integers so that subsystems can be distinguished.\n\n        Args:\n            qargs (None or list): subsystems to return probabilities for,\n                if None return for all subsystems (Default: None).\n            decimals (None or int): the number of decimal places to round\n                values. If None no rounding is done (Default: None).\n\n        Returns:\n            dict: The measurement probabilities in dict (ket) form.\n        '
        return self._vector_to_dict(self.probabilities(qargs=qargs, decimals=decimals), self.dims(qargs), string_labels=True)

    def sample_memory(self, shots: int, qargs: None | list=None) -> np.ndarray:
        if False:
            return 10
        'Sample a list of qubit measurement outcomes in the computational basis.\n\n        Args:\n            shots (int): number of samples to generate.\n            qargs (None or list): subsystems to sample measurements for,\n                                if None sample measurement of all\n                                subsystems (Default: None).\n\n        Returns:\n            np.array: list of sampled counts if the order sampled.\n\n        Additional Information:\n\n            This function *samples* measurement outcomes using the measure\n            :meth:`probabilities` for the current state and `qargs`. It does\n            not actually implement the measurement so the current state is\n            not modified.\n\n            The seed for random number generator used for sampling can be\n            set to a fixed value by using the stats :meth:`seed` method.\n        '
        probs = self.probabilities(qargs)
        labels = self._index_to_ket_array(np.arange(len(probs)), self.dims(qargs), string_labels=True)
        return self._rng.choice(labels, p=probs, size=shots)

    def sample_counts(self, shots: int, qargs: None | list=None) -> Counts:
        if False:
            print('Hello World!')
        'Sample a dict of qubit measurement outcomes in the computational basis.\n\n        Args:\n            shots (int): number of samples to generate.\n            qargs (None or list): subsystems to sample measurements for,\n                                if None sample measurement of all\n                                subsystems (Default: None).\n\n        Returns:\n            Counts: sampled counts dictionary.\n\n        Additional Information:\n\n            This function *samples* measurement outcomes using the measure\n            :meth:`probabilities` for the current state and `qargs`. It does\n            not actually implement the measurement so the current state is\n            not modified.\n\n            The seed for random number generator used for sampling can be\n            set to a fixed value by using the stats :meth:`seed` method.\n        '
        samples = self.sample_memory(shots, qargs=qargs)
        (inds, counts) = np.unique(samples, return_counts=True)
        return Counts(zip(inds, counts))

    def measure(self, qargs: list | None=None) -> tuple:
        if False:
            return 10
        'Measure subsystems and return outcome and post-measure state.\n\n        Note that this function uses the QuantumStates internal random\n        number generator for sampling the measurement outcome. The RNG\n        seed can be set using the :meth:`seed` method.\n\n        Args:\n            qargs (list or None): subsystems to sample measurements for,\n                                  if None sample measurement of all\n                                  subsystems (Default: None).\n\n        Returns:\n            tuple: the pair ``(outcome, state)`` where ``outcome`` is the\n                   measurement outcome string label, and ``state`` is the\n                   collapsed post-measurement state for the corresponding\n                   outcome.\n        '
        dims = self.dims(qargs)
        probs = self.probabilities(qargs)
        sample = self._rng.choice(len(probs), p=probs, size=1)
        outcome = self._index_to_ket_array(sample, self.dims(qargs), string_labels=True)[0]
        proj = np.zeros(len(probs), dtype=complex)
        proj[sample] = 1 / np.sqrt(probs[sample])
        ret = self.evolve(Operator(np.diag(proj), input_dims=dims, output_dims=dims), qargs=qargs)
        return (outcome, ret)

    @staticmethod
    def _index_to_ket_array(inds: np.ndarray, dims: tuple, string_labels: bool=False) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        'Convert an index array into a ket array.\n\n        Args:\n            inds (np.array): an integer index array.\n            dims (tuple): a list of subsystem dimensions.\n            string_labels (bool): return ket as string if True, otherwise\n                                  return as index array (Default: False).\n\n        Returns:\n            np.array: an array of ket strings if string_label=True, otherwise\n                      an array of ket lists.\n        '
        shifts = [1]
        for dim in dims[:-1]:
            shifts.append(shifts[-1] * dim)
        kets = np.array([inds // shift % dim for (dim, shift) in zip(dims, shifts)])
        if string_labels:
            max_dim = max(dims)
            char_kets = np.asarray(kets, dtype=np.str_)
            str_kets = char_kets[0]
            for row in char_kets[1:]:
                if max_dim > 10:
                    str_kets = np.char.add(',', str_kets)
                str_kets = np.char.add(row, str_kets)
            return str_kets.T
        return kets.T

    @staticmethod
    def _vector_to_dict(vec, dims, decimals=None, string_labels=False):
        if False:
            return 10
        'Convert a vector to a ket dictionary.\n\n        This representation will not show zero values in the output dict.\n\n        Args:\n            vec (array): a Numpy vector array.\n            dims (tuple): subsystem dimensions.\n            decimals (None or int): number of decimal places to round to.\n                                    (See Numpy.round), if None no rounding\n                                    is done (Default: None).\n            string_labels (bool): return ket as string if True, otherwise\n                                  return as index array (Default: False).\n\n        Returns:\n            dict: the vector in dictionary `ket` form.\n        '
        vals = vec if decimals is None else vec.round(decimals=decimals)
        (inds,) = vals.nonzero()
        kets = QuantumState._index_to_ket_array(inds, dims, string_labels=string_labels)
        if string_labels:
            return dict(zip(kets, vec[inds]))
        return {tuple(ket): val for (ket, val) in zip(kets, vals[inds])}

    @staticmethod
    def _matrix_to_dict(mat, dims, decimals=None, string_labels=False):
        if False:
            for i in range(10):
                print('nop')
        'Convert a matrix to a ket dictionary.\n\n        This representation will not show zero values in the output dict.\n\n        Args:\n            mat (array): a Numpy matrix array.\n            dims (tuple): subsystem dimensions.\n            decimals (None or int): number of decimal places to round to.\n                                    (See Numpy.round), if None no rounding\n                                    is done (Default: None).\n            string_labels (bool): return ket as string if True, otherwise\n                                  return as index array (Default: False).\n\n        Returns:\n            dict: the matrix in dictionary `ket` form.\n        '
        vals = mat if decimals is None else mat.round(decimals=decimals)
        (inds_row, inds_col) = vals.nonzero()
        bras = QuantumState._index_to_ket_array(inds_row, dims, string_labels=string_labels)
        kets = QuantumState._index_to_ket_array(inds_col, dims, string_labels=string_labels)
        if string_labels:
            return {f'{ket}|{bra}': val for (ket, bra, val) in zip(kets, bras, vals[inds_row, inds_col])}
        return {(tuple(ket), tuple(bra)): val for (ket, bra, val) in zip(kets, bras, vals[inds_row, inds_col])}

    @staticmethod
    def _subsystem_probabilities(probs: np.ndarray, dims: tuple, qargs: None | list=None) -> np.ndarray:
        if False:
            while True:
                i = 10
        'Marginalize a probability vector according to subsystems.\n\n        Args:\n            probs (np.array): a probability vector Numpy array.\n            dims (tuple): subsystem dimensions.\n            qargs (None or list): a list of subsystems to return\n                marginalized probabilities for. If None return all\n                probabilities (Default: None).\n\n        Returns:\n            np.array: the marginalized probability vector flattened\n                      for the specified qargs.\n        '
        if qargs is None:
            return probs
        probs_tens = np.reshape(probs, list(reversed(dims)))
        ndim = probs_tens.ndim
        qargs_axes = [ndim - 1 - i for i in reversed(qargs)]
        sum_axis = tuple((i for i in range(ndim) if i not in qargs_axes))
        if sum_axis:
            probs_tens = np.sum(probs_tens, axis=sum_axis)
            qargs_axes = np.argsort(np.argsort(qargs_axes))
        probs_tens = np.transpose(probs_tens, axes=qargs_axes)
        new_probs = np.reshape(probs_tens, (probs_tens.size,))
        return new_probs

    def __and__(self, other):
        if False:
            while True:
                i = 10
        return self.evolve(other)

    def __xor__(self, other):
        if False:
            i = 10
            return i + 15
        return self.tensor(other)

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        return self._multiply(other)

    def __truediv__(self, other):
        if False:
            return 10
        return self._multiply(1 / other)

    def __rmul__(self, other):
        if False:
            return 10
        return self.__mul__(other)

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._add(other)

    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._add(-other)

    def __neg__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._multiply(-1)