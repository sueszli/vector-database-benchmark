"""TensoredOp Class"""
from functools import partial, reduce
from typing import List, Union, cast, Dict
import numpy as np
from qiskit.circuit import ParameterExpression, QuantumCircuit
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.quantum_info import Statevector
from qiskit.utils.deprecation import deprecate_func

class TensoredOp(ListOp):
    """Deprecated: A class for lazily representing tensor products of Operators. Often Operators
    cannot be efficiently tensored to one another, but may be manipulated further so that they can be
    later. This class holds logic to indicate that the Operators in ``oplist`` are meant to
    be tensored together, and therefore if they reach a point in which they can be, such as after
    conversion to QuantumCircuits, they can be reduced by tensor product."""

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, oplist: List[OperatorBase], coeff: Union[complex, ParameterExpression]=1.0, abelian: bool=False) -> None:
        if False:
            return 10
        '\n        Args:\n            oplist: The Operators being tensored.\n            coeff: A coefficient multiplying the operator\n            abelian: Indicates whether the Operators in ``oplist`` are known to mutually commute.\n        '
        super().__init__(oplist, combo_fn=partial(reduce, np.kron), coeff=coeff, abelian=abelian)

    @property
    def num_qubits(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return sum((op.num_qubits for op in self.oplist))

    @property
    def distributive(self) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    @property
    def settings(self) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        'Return settings.'
        return {'oplist': self._oplist, 'coeff': self._coeff, 'abelian': self._abelian}

    def _expand_dim(self, num_qubits: int) -> 'TensoredOp':
        if False:
            i = 10
            return i + 15
        'Appends I ^ num_qubits to ``oplist``. Choice of PauliOp as\n        identity is arbitrary and can be substituted for other PrimitiveOp identity.\n\n        Returns:\n            TensoredOp expanded with identity operator.\n        '
        from ..operator_globals import I
        return TensoredOp(self.oplist + [I ^ num_qubits], coeff=self.coeff)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if False:
            print('Hello World!')
        if isinstance(other, TensoredOp):
            return TensoredOp(self.oplist + other.oplist, coeff=self.coeff * other.coeff)
        return TensoredOp(self.oplist + [other], coeff=self.coeff)

    def eval(self, front: Union[str, dict, np.ndarray, OperatorBase, Statevector]=None) -> Union[OperatorBase, complex]:
        if False:
            print('Hello World!')
        if self._is_empty():
            return 0.0
        return cast(Union[OperatorBase, complex], self.to_matrix_op().eval(front=front))

    def reduce(self) -> OperatorBase:
        if False:
            for i in range(10):
                print('nop')
        reduced_ops = [op.reduce() for op in self.oplist]
        if self._is_empty():
            return self.__class__([], coeff=self.coeff, abelian=self.abelian)
        reduced_ops = reduce(lambda x, y: x.tensor(y), reduced_ops) * self.coeff
        if isinstance(reduced_ops, ListOp) and len(reduced_ops.oplist) == 1:
            return reduced_ops.oplist[0]
        else:
            return cast(OperatorBase, reduced_ops)

    def to_circuit(self) -> QuantumCircuit:
        if False:
            while True:
                i = 10
        'Returns the quantum circuit, representing the tensored operator.\n\n        Returns:\n            The circuit representation of the tensored operator.\n\n        Raises:\n            OpflowError: for operators where a single underlying circuit can not be produced.\n        '
        circuit_op = self.to_circuit_op()
        from ..state_fns.circuit_state_fn import CircuitStateFn
        from ..primitive_ops.primitive_op import PrimitiveOp
        if isinstance(circuit_op, (PrimitiveOp, CircuitStateFn)):
            return circuit_op.to_circuit()
        raise OpflowError('Conversion to_circuit supported only for operators, where a single underlying circuit can be produced.')

    def to_matrix(self, massive: bool=False) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        OperatorBase._check_massive('to_matrix', True, self.num_qubits, massive)
        mat = self.coeff * reduce(np.kron, [np.asarray(op.to_matrix()) for op in self.oplist])
        return np.asarray(mat, dtype=complex)