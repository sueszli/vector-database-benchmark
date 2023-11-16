"""EvolutionOp Class"""
from typing import List, Optional, Set, Union, cast
import numpy as np
import scipy
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops.composed_op import ComposedOp
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.list_ops.tensored_op import TensoredOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.matrix_op import MatrixOp
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.quantum_info import Statevector
from qiskit.utils.deprecation import deprecate_func

class EvolvedOp(PrimitiveOp):
    """
    Deprecated: Class for wrapping Operator Evolutions for compilation (``convert``) by an EvolutionBase
    method later, essentially acting as a placeholder. Note that EvolvedOp is a weird case of
    PrimitiveOp. It happens to be that it fits into the PrimitiveOp interface nearly perfectly,
    and it essentially represents a placeholder for a PrimitiveOp later, even though it doesn't
    actually hold a primitive object. We could have chosen for it to be an OperatorBase,
    but would have ended up copying and pasting a lot of code from PrimitiveOp."""
    primitive: PrimitiveOp

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, primitive: OperatorBase, coeff: Union[complex, ParameterExpression]=1.0) -> None:
        if False:
            while True:
                i = 10
        '\n        Args:\n            primitive: The operator being wrapped to signify evolution later.\n            coeff: A coefficient multiplying the operator\n        '
        super().__init__(primitive, coeff=coeff)

    def primitive_strings(self) -> Set[str]:
        if False:
            print('Hello World!')
        return self.primitive.primitive_strings()

    @property
    def num_qubits(self) -> int:
        if False:
            while True:
                i = 10
        return self.primitive.num_qubits

    def add(self, other: OperatorBase) -> Union['EvolvedOp', SummedOp]:
        if False:
            for i in range(10):
                print('nop')
        if not self.num_qubits == other.num_qubits:
            raise ValueError('Sum over operators with different numbers of qubits, {} and {}, is not well defined'.format(self.num_qubits, other.num_qubits))
        if isinstance(other, EvolvedOp) and self.primitive == other.primitive:
            return EvolvedOp(self.primitive, coeff=self.coeff + other.coeff)
        if isinstance(other, SummedOp):
            op_list = [cast(OperatorBase, self)] + other.oplist
            return SummedOp(op_list)
        return SummedOp([self, other])

    def adjoint(self) -> 'EvolvedOp':
        if False:
            while True:
                i = 10
        return EvolvedOp(self.primitive.adjoint() * -1, coeff=self.coeff.conjugate())

    def equals(self, other: OperatorBase) -> bool:
        if False:
            while True:
                i = 10
        if not isinstance(other, EvolvedOp) or not self.coeff == other.coeff:
            return False
        return self.primitive == other.primitive

    def tensor(self, other: OperatorBase) -> TensoredOp:
        if False:
            print('Hello World!')
        if isinstance(other, TensoredOp):
            return TensoredOp([cast(OperatorBase, self)] + other.oplist)
        return TensoredOp([self, other])

    def _expand_dim(self, num_qubits: int) -> TensoredOp:
        if False:
            return 10
        from ..operator_globals import I
        return self.tensor(I ^ num_qubits)

    def permute(self, permutation: List[int]) -> 'EvolvedOp':
        if False:
            return 10
        return EvolvedOp(self.primitive.permute(permutation), coeff=self.coeff)

    def compose(self, other: OperatorBase, permutation: Optional[List[int]]=None, front: bool=False) -> OperatorBase:
        if False:
            return 10
        (new_self, other) = self._expand_shorter_operator_and_permute(other, permutation)
        if front:
            return other.compose(new_self)
        if isinstance(other, ComposedOp):
            return ComposedOp([new_self] + other.oplist)
        return ComposedOp([new_self, other])

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return f'e^(-i*{prim_str})'
        else:
            return f'{self.coeff} * e^(-i*{prim_str})'

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'EvolvedOp({repr(self.primitive)}, coeff={self.coeff})'

    def reduce(self) -> 'EvolvedOp':
        if False:
            while True:
                i = 10
        return EvolvedOp(self.primitive.reduce(), coeff=self.coeff)

    def assign_parameters(self, param_dict: dict) -> Union['EvolvedOp', ListOp]:
        if False:
            for i in range(10):
                print('nop')
        param_value = self.coeff
        if isinstance(self.coeff, ParameterExpression):
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                return ListOp([self.assign_parameters(param_dict) for param_dict in unrolled_dict])
            if self.coeff.parameters <= set(unrolled_dict.keys()):
                binds = {param: unrolled_dict[param] for param in self.coeff.parameters}
                param_value = float(self.coeff.bind(binds))
        return EvolvedOp(self.primitive.bind_parameters(param_dict), coeff=param_value)

    def eval(self, front: Optional[Union[str, dict, np.ndarray, OperatorBase, Statevector]]=None) -> Union[OperatorBase, complex]:
        if False:
            return 10
        return cast(Union[OperatorBase, complex], self.to_matrix_op().eval(front=front))

    def to_matrix(self, massive: bool=False) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.primitive, ListOp) and self.primitive.__class__.__name__ == ListOp.__name__:
            return np.array([op.exp_i().to_matrix(massive=massive) * self.primitive.coeff * self.coeff for op in self.primitive.oplist], dtype=complex)
        prim_mat = -1j * self.primitive.to_matrix()
        return scipy.linalg.expm(prim_mat) * self.coeff

    def to_matrix_op(self, massive: bool=False) -> Union[ListOp, MatrixOp]:
        if False:
            while True:
                i = 10
        'Returns a ``MatrixOp`` equivalent to this Operator.'
        primitive = self.primitive
        if isinstance(primitive, ListOp) and primitive.__class__.__name__ == ListOp.__name__:
            return ListOp([op.exp_i().to_matrix_op() for op in primitive.oplist], coeff=primitive.coeff * self.coeff)
        prim_mat = EvolvedOp(primitive).to_matrix(massive=massive)
        return MatrixOp(prim_mat, coeff=self.coeff)

    def log_i(self, massive: bool=False) -> OperatorBase:
        if False:
            print('Hello World!')
        return self.primitive * self.coeff

    def to_instruction(self, massive: bool=False) -> Instruction:
        if False:
            while True:
                i = 10
        mat_op = self.to_matrix_op(massive=massive)
        if not isinstance(mat_op, MatrixOp):
            raise OpflowError('to_instruction is not allowed for ListOp.')
        return mat_op.to_instruction()