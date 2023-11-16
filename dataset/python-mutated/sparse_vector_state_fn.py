"""SparseVectorStateFn class."""
from typing import Dict, Optional, Set, Union
import numpy as np
import scipy
from qiskit.circuit import ParameterExpression
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.opflow.state_fns.vector_state_fn import VectorStateFn
from qiskit.quantum_info import Statevector
from qiskit.utils import algorithm_globals
from qiskit.utils.deprecation import deprecate_func

class SparseVectorStateFn(StateFn):
    """Deprecated: A class for sparse state functions and measurements in vector representation.

    This class uses ``scipy.sparse.spmatrix`` for the internal representation.
    """
    primitive: scipy.sparse.spmatrix

    @deprecate_func(since='0.24.0', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, primitive: scipy.sparse.spmatrix, coeff: Union[complex, ParameterExpression]=1.0, is_measurement: bool=False) -> None:
        if False:
            return 10
        '\n        Args:\n            primitive: The underlying sparse vector.\n            coeff: A coefficient multiplying the state function.\n            is_measurement: Whether the StateFn is a measurement operator\n\n        Raises:\n            ValueError: If the primitive is not a column vector.\n            ValueError: If the number of elements in the primitive is not a power of 2.\n\n        '
        if primitive.shape[0] != 1:
            raise ValueError('The primitive must be a row vector of shape (x, 1).')
        self._num_qubits = int(np.log2(primitive.shape[1]))
        if np.log2(primitive.shape[1]) != self._num_qubits:
            raise ValueError('The number of vector elements must be a power of 2.')
        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)

    def primitive_strings(self) -> Set[str]:
        if False:
            while True:
                i = 10
        return {'SparseVector'}

    @property
    def num_qubits(self) -> int:
        if False:
            print('Hello World!')
        return self._num_qubits

    def add(self, other: OperatorBase) -> OperatorBase:
        if False:
            print('Hello World!')
        if not self.num_qubits == other.num_qubits:
            raise ValueError('Sum over statefns with different numbers of qubits, {} and {}, is not well defined'.format(self.num_qubits, other.num_qubits))
        if isinstance(other, SparseVectorStateFn) and self.is_measurement == other.is_measurement:
            added = self.coeff * self.primitive + other.coeff * other.primitive
            return SparseVectorStateFn(added, is_measurement=self._is_measurement)
        return SummedOp([self, other])

    def adjoint(self) -> 'SparseVectorStateFn':
        if False:
            while True:
                i = 10
        return SparseVectorStateFn(self.primitive.conjugate(), coeff=self.coeff.conjugate(), is_measurement=not self.is_measurement)

    def equals(self, other: OperatorBase) -> bool:
        if False:
            return 10
        if not isinstance(other, SparseVectorStateFn) or not self.coeff == other.coeff:
            return False
        if self.primitive.shape != other.primitive.shape:
            return False
        if self.primitive.count_nonzero() != other.primitive.count_nonzero():
            return False
        return (self.primitive != other.primitive).nnz == 0

    def to_dict_fn(self) -> StateFn:
        if False:
            for i in range(10):
                print('nop')
        'Convert this state function to a ``DictStateFn``.\n\n        Returns:\n            A new DictStateFn equivalent to ``self``.\n        '
        from .dict_state_fn import DictStateFn
        num_qubits = self.num_qubits
        dok = self.primitive.todok()
        new_dict = {format(i[1], 'b').zfill(num_qubits): v for (i, v) in dok.items()}
        return DictStateFn(new_dict, coeff=self.coeff, is_measurement=self.is_measurement)

    def to_matrix(self, massive: bool=False) -> np.ndarray:
        if False:
            print('Hello World!')
        OperatorBase._check_massive('to_matrix', False, self.num_qubits, massive)
        vec = self.primitive.toarray() * self.coeff
        return vec if not self.is_measurement else vec.reshape(1, -1)

    def to_matrix_op(self, massive: bool=False) -> OperatorBase:
        if False:
            i = 10
            return i + 15
        return VectorStateFn(self.to_matrix())

    def to_spmatrix(self) -> OperatorBase:
        if False:
            while True:
                i = 10
        return self

    def to_circuit_op(self) -> OperatorBase:
        if False:
            for i in range(10):
                print('nop')
        'Convert this state function to a ``CircuitStateFn``.'
        from .circuit_state_fn import CircuitStateFn
        csfn = CircuitStateFn.from_vector(self.primitive) * self.coeff
        return csfn.adjoint() if self.is_measurement else csfn

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return '{}({})'.format('SparseVectorStateFn' if not self.is_measurement else 'MeasurementSparseVector', prim_str)
        else:
            return '{}({}) * {}'.format('SparseVectorStateFn' if not self.is_measurement else 'SparseMeasurementVector', prim_str, self.coeff)

    def eval(self, front: Optional[Union[str, Dict[str, complex], np.ndarray, Statevector, OperatorBase]]=None) -> Union[OperatorBase, complex]:
        if False:
            i = 10
            return i + 15
        if front is None:
            return self
        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError('Cannot compute overlap with StateFn or Operator if not Measurement. Try taking sf.adjoint() first to convert to measurement.')
        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn([self.eval(front.coeff * front_elem) for front_elem in front.oplist])
        if not isinstance(front, OperatorBase):
            front = StateFn(front)
        from ..operator_globals import EVAL_SIG_DIGITS
        from .operator_state_fn import OperatorStateFn
        from .circuit_state_fn import CircuitStateFn
        from .dict_state_fn import DictStateFn
        if isinstance(front, DictStateFn):
            return np.round(sum((v * self.primitive.data[int(b, 2)] * front.coeff for (b, v) in front.primitive.items())) * self.coeff, decimals=EVAL_SIG_DIGITS)
        if isinstance(front, VectorStateFn):
            return np.round(np.dot(self.to_matrix(), front.to_matrix())[0], decimals=EVAL_SIG_DIGITS)
        if isinstance(front, CircuitStateFn):
            return np.conj(front.adjoint().eval(self.adjoint().primitive)) * self.coeff
        if isinstance(front, OperatorStateFn):
            return front.adjoint().eval(self.primitive) * self.coeff
        return front.adjoint().eval(self.adjoint().primitive).adjoint() * self.coeff

    def sample(self, shots: int=1024, massive: bool=False, reverse_endianness: bool=False) -> dict:
        if False:
            while True:
                i = 10
        as_dict = self.to_dict_fn().primitive
        all_states = sum(as_dict.keys())
        deterministic_counts = {key: value / all_states for (key, value) in as_dict.items()}
        probs = np.array(list(deterministic_counts.values()))
        (unique, counts) = np.unique(algorithm_globals.random.choice(list(deterministic_counts.keys()), size=shots, p=probs / sum(probs)), return_counts=True)
        counts = dict(zip(unique, counts))
        if reverse_endianness:
            scaled_dict = {bstr[::-1]: prob / shots for (bstr, prob) in counts.items()}
        else:
            scaled_dict = {bstr: prob / shots for (bstr, prob) in counts.items()}
        return dict(sorted(scaled_dict.items(), key=lambda x: x[1], reverse=True))