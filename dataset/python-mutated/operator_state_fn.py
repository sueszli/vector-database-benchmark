"""OperatorStateFn Class"""
from typing import List, Optional, Set, Union, cast
import numpy as np
from qiskit.circuit import ParameterExpression
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.list_ops.tensored_op import TensoredOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.matrix_op import MatrixOp
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.quantum_info import Statevector
from qiskit.utils.deprecation import deprecate_func

class OperatorStateFn(StateFn):
    """
    Deprecated: A class for state functions and measurements which are defined by a density Operator,
    stored using an ``OperatorBase``.
    """
    primitive: OperatorBase

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, primitive: OperatorBase, coeff: Union[complex, ParameterExpression]=1.0, is_measurement: bool=False) -> None:
        if False:
            while True:
                i = 10
        '\n        Args:\n            primitive: The ``OperatorBase`` which defines the behavior of the underlying State\n                function.\n            coeff: A coefficient by which to multiply the state function\n            is_measurement: Whether the StateFn is a measurement operator\n        '
        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)

    def primitive_strings(self) -> Set[str]:
        if False:
            i = 10
            return i + 15
        return self.primitive.primitive_strings()

    @property
    def num_qubits(self) -> int:
        if False:
            print('Hello World!')
        return self.primitive.num_qubits

    def add(self, other: OperatorBase) -> Union['OperatorStateFn', SummedOp]:
        if False:
            print('Hello World!')
        if not self.num_qubits == other.num_qubits:
            raise ValueError('Sum over statefns with different numbers of qubits, {} and {}, is not well defined'.format(self.num_qubits, other.num_qubits))
        if isinstance(other, OperatorStateFn) and self.is_measurement == other.is_measurement:
            if isinstance(other.primitive, OperatorBase) and self.primitive == other.primitive:
                return OperatorStateFn(self.primitive, coeff=self.coeff + other.coeff, is_measurement=self.is_measurement)
            elif isinstance(other, OperatorStateFn):
                return OperatorStateFn((self.coeff * self.primitive).add(other.primitive * other.coeff), is_measurement=self._is_measurement)
        return SummedOp([self, other])

    def adjoint(self) -> 'OperatorStateFn':
        if False:
            while True:
                i = 10
        return OperatorStateFn(self.primitive.adjoint(), coeff=self.coeff.conjugate(), is_measurement=not self.is_measurement)

    def _expand_dim(self, num_qubits: int) -> 'OperatorStateFn':
        if False:
            for i in range(10):
                print('nop')
        return OperatorStateFn(self.primitive._expand_dim(num_qubits), coeff=self.coeff, is_measurement=self.is_measurement)

    def permute(self, permutation: List[int]) -> 'OperatorStateFn':
        if False:
            i = 10
            return i + 15
        return OperatorStateFn(self.primitive.permute(permutation), coeff=self.coeff, is_measurement=self.is_measurement)

    def tensor(self, other: OperatorBase) -> Union['OperatorStateFn', TensoredOp]:
        if False:
            print('Hello World!')
        if isinstance(other, OperatorStateFn):
            return OperatorStateFn(self.primitive.tensor(other.primitive), coeff=self.coeff * other.coeff, is_measurement=self.is_measurement)
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool=False) -> np.ndarray:
        if False:
            return 10
        'Return numpy matrix of density operator, warn if more than 16 qubits\n        to force the user to set\n        massive=True if they want such a large matrix. Generally big methods like\n        this should require the use of a\n        converter, but in this case a convenience method for quick hacking and\n        access to classical tools is\n        appropriate.'
        OperatorBase._check_massive('to_density_matrix', True, self.num_qubits, massive)
        return self.primitive.to_matrix() * self.coeff

    def to_matrix_op(self, massive: bool=False) -> 'OperatorStateFn':
        if False:
            return 10
        'Return a MatrixOp for this operator.'
        return OperatorStateFn(self.primitive.to_matrix_op(massive=massive) * self.coeff, is_measurement=self.is_measurement)

    def to_matrix(self, massive: bool=False) -> np.ndarray:
        if False:
            return 10
        '\n        Note: this does not return a density matrix, it returns a classical matrix\n        containing the quantum or classical vector representing the evaluation of the state\n        function on each binary basis state. Do not assume this is is a normalized quantum or\n        classical probability vector. If we allowed this to return a density matrix,\n        then we would need to change the definition of composition to be ~Op @ StateFn @ Op for\n        those cases, whereas by this methodology we can ensure that composition always means Op\n        @ StateFn.\n\n        Return numpy vector of state vector, warn if more than 16 qubits to force the user to set\n        massive=True if they want such a large vector.\n\n        Args:\n            massive: Whether to allow large conversions, e.g. creating a matrix representing\n                over 16 qubits.\n\n        Returns:\n            np.ndarray: Vector of state vector\n\n        Raises:\n            ValueError: Invalid parameters.\n        '
        OperatorBase._check_massive('to_matrix', False, self.num_qubits, massive)
        mat = self.primitive.to_matrix(massive=massive)

        def diag_over_tree(op):
            if False:
                while True:
                    i = 10
            if isinstance(op, list):
                return [diag_over_tree(o) for o in op]
            else:
                vec = np.diag(op) * self.coeff
                return vec if not self.is_measurement else vec.reshape(1, -1)
        return diag_over_tree(mat)

    def to_circuit_op(self):
        if False:
            return 10
        'Return ``StateFnCircuit`` corresponding to this StateFn. Ignore for now because this is\n        undefined. TODO maybe call to_pauli_op and diagonalize here, but that could be very\n        inefficient, e.g. splitting one Stabilizer measurement into hundreds of 1 qubit Paulis.'
        raise NotImplementedError

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return '{}({})'.format('OperatorStateFn' if not self.is_measurement else 'OperatorMeasurement', prim_str)
        else:
            return '{}({}) * {}'.format('OperatorStateFn' if not self.is_measurement else 'OperatorMeasurement', prim_str, self.coeff)

    def eval(self, front: Optional[Union[str, dict, np.ndarray, OperatorBase, Statevector]]=None) -> Union[OperatorBase, complex]:
        if False:
            while True:
                i = 10
        if front is None:
            matrix = cast(MatrixOp, self.primitive.to_matrix_op()).primitive.data
            from .vector_state_fn import VectorStateFn
            return VectorStateFn(matrix[0, :])
        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError('Cannot compute overlap with StateFn or Operator if not Measurement. Try taking sf.adjoint() first to convert to measurement.')
        if not isinstance(front, OperatorBase):
            front = StateFn(front)
        if isinstance(self.primitive, ListOp) and self.primitive.distributive:
            evals = [OperatorStateFn(op, is_measurement=self.is_measurement).eval(front) for op in self.primitive.oplist]
            result = self.primitive.combo_fn(evals)
            if isinstance(result, list):
                multiplied = self.primitive.coeff * self.coeff * np.array(result)
                return multiplied.tolist()
            return result * self.coeff * self.primitive.coeff
        from .vector_state_fn import VectorStateFn
        if isinstance(self.primitive, PauliSumOp) and isinstance(front, VectorStateFn):
            return front.primitive.expectation_value(self.primitive.primitive) * self.coeff * front.coeff
        if isinstance(front, ListOp) and type(front) == ListOp:
            return front.combo_fn([self.eval(front.coeff * front_elem) for front_elem in front.oplist])
        if isinstance(front, CircuitStateFn):
            front = front.eval()
        return front.adjoint().eval(cast(OperatorBase, self.primitive.eval(front))) * self.coeff

    def sample(self, shots: int=1024, massive: bool=False, reverse_endianness: bool=False):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError