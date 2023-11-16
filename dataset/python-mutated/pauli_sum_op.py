"""PauliSumOp Class"""
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union, cast
import numpy as np
from scipy.sparse import spmatrix
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.list_ops.tensored_op import TensoredOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit.quantum_info.operators.custom_iterator import CustomIterator
from qiskit.utils.deprecation import deprecate_func

class PauliSumOp(PrimitiveOp):
    """Deprecated: Class for Operators backed by Terra's ``SparsePauliOp`` class."""
    primitive: SparsePauliOp

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, primitive: SparsePauliOp, coeff: Union[complex, ParameterExpression]=1.0, grouping_type: str='None') -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            primitive: The SparsePauliOp which defines the behavior of the underlying function.\n            coeff: A coefficient multiplying the primitive.\n            grouping_type: The type of grouping. If None, the operator is not grouped.\n\n        Raises:\n            TypeError: invalid parameters.\n        '
        if not isinstance(primitive, SparsePauliOp):
            raise TypeError(f'PauliSumOp can only be instantiated with SparsePauliOp, not {type(primitive)}')
        super().__init__(primitive, coeff=coeff)
        self._grouping_type = grouping_type

    def primitive_strings(self) -> Set[str]:
        if False:
            i = 10
            return i + 15
        return {'SparsePauliOp'}

    @property
    def grouping_type(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Returns: Type of Grouping\n        '
        return self._grouping_type

    @property
    def num_qubits(self) -> int:
        if False:
            i = 10
            return i + 15
        return self.primitive.num_qubits

    @property
    def coeffs(self):
        if False:
            while True:
                i = 10
        'Return the Pauli coefficients.'
        return self.coeff * self.primitive.coeffs

    @property
    def settings(self) -> Dict:
        if False:
            return 10
        'Return operator settings.'
        data = super().settings
        data.update({'grouping_type': self._grouping_type})
        return data

    def matrix_iter(self, sparse=False):
        if False:
            for i in range(10):
                print('nop')
        'Return a matrix representation iterator.\n\n        This is a lazy iterator that converts each term in the PauliSumOp\n        into a matrix as it is used. To convert to a single matrix use the\n        :meth:`to_matrix` method.\n\n        Args:\n            sparse (bool): optionally return sparse CSR matrices if True,\n                           otherwise return Numpy array matrices\n                           (Default: False)\n\n        Returns:\n            MatrixIterator: matrix iterator object for the PauliSumOp.\n        '

        class MatrixIterator(CustomIterator):
            """Matrix representation iteration and item access."""

            def __repr__(self):
                if False:
                    while True:
                        i = 10
                return f'<PauliSumOp_matrix_iterator at {hex(id(self))}>'

            def __getitem__(self, key):
                if False:
                    i = 10
                    return i + 15
                sumopcoeff = self.obj.coeff * self.obj.primitive.coeffs[key]
                return sumopcoeff * self.obj.primitive.paulis[key].to_matrix(sparse=sparse)
        return MatrixIterator(self)

    def add(self, other: OperatorBase) -> OperatorBase:
        if False:
            print('Hello World!')
        if not self.num_qubits == other.num_qubits:
            raise ValueError(f'Sum of operators with different numbers of qubits, {self.num_qubits} and {other.num_qubits}, is not well defined')
        if isinstance(other, PauliSumOp) and (not isinstance(self.coeff, ParameterExpression)) and (not isinstance(other.coeff, ParameterExpression)):
            return PauliSumOp(self.coeff * self.primitive + other.coeff * other.primitive, coeff=1)
        if isinstance(other, PauliOp) and (not isinstance(self.coeff, ParameterExpression)) and (not isinstance(other.coeff, ParameterExpression)):
            return PauliSumOp(self.coeff * self.primitive + other.coeff * SparsePauliOp(other.primitive))
        return SummedOp([self, other])

    def mul(self, scalar: Union[complex, ParameterExpression]) -> OperatorBase:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(scalar, (int, float, complex)) and scalar != 0:
            return PauliSumOp(scalar * self.primitive, coeff=self.coeff)
        return PauliSumOp(self.primitive, coeff=self.coeff * scalar)

    def adjoint(self) -> 'PauliSumOp':
        if False:
            while True:
                i = 10
        return PauliSumOp(self.primitive.adjoint(), coeff=self.coeff.conjugate())

    def equals(self, other: OperatorBase) -> bool:
        if False:
            while True:
                i = 10
        (self_reduced, other_reduced) = (self.reduce(), other.reduce())
        if isinstance(other_reduced, PauliOp):
            other_reduced = PauliSumOp(SparsePauliOp(other_reduced.primitive, coeffs=[other_reduced.coeff]))
        if not isinstance(other_reduced, PauliSumOp):
            return False
        if isinstance(self_reduced.coeff, ParameterExpression) or isinstance(other_reduced.coeff, ParameterExpression):
            return self_reduced.coeff == other_reduced.coeff and self_reduced.primitive.equiv(other_reduced.primitive)
        return len(self_reduced) == len(other_reduced) and self_reduced.primitive.equiv(other_reduced.primitive)

    def _expand_dim(self, num_qubits: int) -> 'PauliSumOp':
        if False:
            print('Hello World!')
        return PauliSumOp(self.primitive.tensor(SparsePauliOp(Pauli('I' * num_qubits))), coeff=self.coeff)

    def tensor(self, other: OperatorBase) -> Union['PauliSumOp', TensoredOp]:
        if False:
            print('Hello World!')
        if isinstance(other, PauliSumOp):
            return PauliSumOp(self.primitive.tensor(other.primitive), coeff=self.coeff * other.coeff)
        if isinstance(other, PauliOp):
            return PauliSumOp(self.primitive.tensor(other.primitive), coeff=self.coeff * other.coeff)
        return TensoredOp([self, other])

    def permute(self, permutation: List[int]) -> 'PauliSumOp':
        if False:
            return 10
        'Permutes the sequence of ``PauliSumOp``.\n\n        Args:\n            permutation: A list defining where each Pauli should be permuted. The Pauli at index\n                j of the primitive should be permuted to position permutation[j].\n\n        Returns:\n              A new PauliSumOp representing the permuted operator. For operator (X ^ Y ^ Z) and\n              indices=[1,2,4], it returns (X ^ I ^ Y ^ Z ^ I).\n\n        Raises:\n            OpflowError: if indices do not define a new index for each qubit.\n        '
        set_perm = set(permutation)
        if len(set_perm) != len(permutation) or any((index < 0 for index in set_perm)):
            raise OpflowError(f'List {permutation} is not a permutation.')
        if len(permutation) != self.num_qubits:
            raise OpflowError('List of indices to permute must have the same size as Pauli Operator')
        length = max(permutation) + 1
        if length > self.num_qubits:
            spop = self.primitive.tensor(SparsePauliOp(Pauli('I' * (length - self.num_qubits))))
        else:
            spop = self.primitive.copy()
        permutation = [i for i in range(length) if i not in permutation] + permutation
        permu_arr = np.arange(length)[np.argsort(permutation)]
        spop.paulis.x = spop.paulis.x[:, permu_arr]
        spop.paulis.z = spop.paulis.z[:, permu_arr]
        return PauliSumOp(spop, self.coeff)

    def compose(self, other: OperatorBase, permutation: Optional[List[int]]=None, front: bool=False) -> OperatorBase:
        if False:
            i = 10
            return i + 15
        (new_self, other) = self._expand_shorter_operator_and_permute(other, permutation)
        new_self = cast(PauliSumOp, new_self)
        if front:
            return other.compose(new_self)
        if not np.any(np.logical_or(new_self.primitive.paulis.x, new_self.primitive.paulis.z)):
            return other * new_self.coeff * sum(new_self.primitive.coeffs)
        if isinstance(other, PauliSumOp):
            return PauliSumOp(new_self.primitive.dot(other.primitive), coeff=new_self.coeff * other.coeff)
        if isinstance(other, PauliOp):
            other_primitive = SparsePauliOp(other.primitive)
            return PauliSumOp(new_self.primitive.dot(other_primitive), coeff=new_self.coeff * other.coeff)
        from ..state_fns.circuit_state_fn import CircuitStateFn
        from .circuit_op import CircuitOp
        if isinstance(other, (CircuitOp, CircuitStateFn)):
            pauli_op = cast(Union[PauliOp, SummedOp], new_self.to_pauli_op())
            return pauli_op.to_circuit_op().compose(other)
        return super(PauliSumOp, new_self).compose(other)

    def to_matrix(self, massive: bool=False) -> np.ndarray:
        if False:
            return 10
        OperatorBase._check_massive('to_matrix', True, self.num_qubits, massive)
        if isinstance(self.coeff, ParameterExpression):
            return self.primitive.to_matrix(sparse=True).toarray() * self.coeff
        return (self.primitive.to_matrix(sparse=True) * self.coeff).toarray()

    def __str__(self) -> str:
        if False:
            return 10

        def format_sign(x):
            if False:
                print('Hello World!')
            return x.real if np.isreal(x) else x

        def format_number(x):
            if False:
                return 10
            x = format_sign(x)
            if isinstance(x, (int, float)) and x < 0:
                return f'- {-x}'
            return f'+ {x}'
        indent = '' if self.coeff == 1 else '  '
        prim_list = self.primitive.to_list()
        if prim_list:
            first = prim_list[0]
            if isinstance(first[1], (int, float)) and first[1] < 0:
                main_string = indent + f'- {-first[1].real} * {first[0]}'
            else:
                main_string = indent + f'{format_sign(first[1])} * {first[0]}'
        main_string += ''.join([f'\n{indent}{format_number(c)} * {p}' for (p, c) in prim_list[1:]])
        return f'{main_string}' if self.coeff == 1 else f'{self.coeff} * (\n{main_string}\n)'

    def eval(self, front: Optional[Union[str, Dict[str, complex], np.ndarray, OperatorBase, Statevector]]=None) -> Union[OperatorBase, complex]:
        if False:
            print('Hello World!')
        if front is None:
            return self.to_matrix_op()
        from ..list_ops.list_op import ListOp
        from ..state_fns.circuit_state_fn import CircuitStateFn
        from ..state_fns.dict_state_fn import DictStateFn
        from ..state_fns.state_fn import StateFn
        from .circuit_op import CircuitOp
        if not isinstance(front, OperatorBase):
            front = StateFn(front, is_measurement=False)
        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn([self.eval(front.coeff * front_elem) for front_elem in front.oplist])
        else:
            if self.num_qubits != front.num_qubits:
                raise ValueError('eval does not support operands with differing numbers of qubits, {} and {}, respectively.'.format(self.num_qubits, front.num_qubits))
            if isinstance(front, DictStateFn):
                new_dict: Dict[str, int] = defaultdict(int)
                corrected_x_bits = self.primitive.paulis.x[:, ::-1]
                corrected_z_bits = self.primitive.paulis.z[:, ::-1]
                coeffs = self.primitive.coeffs
                for (bstr, v) in front.primitive.items():
                    bitstr = np.fromiter(bstr, dtype=int).astype(bool)
                    new_b_str = np.logical_xor(bitstr, corrected_x_bits)
                    new_str = [''.join([str(b) for b in bs]) for bs in new_b_str.astype(int)]
                    z_factor = np.prod(1 - 2 * np.logical_and(bitstr, corrected_z_bits), axis=1)
                    y_factor = np.prod(np.sqrt(1 - 2 * np.logical_and(corrected_x_bits, corrected_z_bits) + 0j), axis=1)
                    for (i, n_str) in enumerate(new_str):
                        new_dict[n_str] += v * z_factor[i] * y_factor[i] * coeffs[i]
                return DictStateFn(new_dict, coeff=self.coeff * front.coeff)
            elif isinstance(front, StateFn) and front.is_measurement:
                raise ValueError('Operator composed with a measurement is undefined.')
            elif isinstance(front, (PauliSumOp, PauliOp, CircuitOp, CircuitStateFn)):
                return self.compose(front).eval()
        front = cast(StateFn, front)
        return self.to_matrix_op().eval(front.to_matrix_op())

    def exp_i(self) -> OperatorBase:
        if False:
            print('Hello World!')
        'Return a ``CircuitOp`` equivalent to e^-iH for this operator H.'
        from ..evolutions.evolved_op import EvolvedOp
        return EvolvedOp(self)

    def to_instruction(self) -> Instruction:
        if False:
            print('Hello World!')
        return self.to_matrix_op().to_circuit().to_instruction()

    def to_pauli_op(self, massive: bool=False) -> Union[PauliOp, SummedOp]:
        if False:
            for i in range(10):
                print('nop')

        def to_native(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.item() if isinstance(x, np.generic) else x
        if len(self.primitive) == 1:
            return PauliOp(Pauli((self.primitive.paulis.z[0], self.primitive.paulis.x[0])), to_native(np.real_if_close(self.primitive.coeffs[0])) * self.coeff)
        coeffs = np.real_if_close(self.primitive.coeffs)
        return SummedOp([PauliOp(pauli, to_native(coeff)) for (pauli, coeff) in zip(self.primitive.paulis, coeffs)], coeff=self.coeff)

    def __getitem__(self, offset: Union[int, slice]) -> 'PauliSumOp':
        if False:
            while True:
                i = 10
        'Allows array-indexing style access to the ``PauliSumOp``.\n\n        Args:\n            offset: The index of ``PauliSumOp``.\n\n        Returns:\n            The ``PauliSumOp`` at index ``offset``,\n        '
        return PauliSumOp(self.primitive[offset], self.coeff)

    def __iter__(self):
        if False:
            return 10
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Length of ``SparsePauliOp``.\n\n        Returns:\n            An int equal to the length of SparsePauliOp.\n        '
        return len(self.primitive)

    def reduce(self, atol: Optional[float]=None, rtol: Optional[float]=None) -> 'PauliSumOp':
        if False:
            i = 10
            return i + 15
        'Simplify the primitive ``SparsePauliOp``.\n\n        Args:\n            atol: Absolute tolerance for checking if coefficients are zero (Default: 1e-8).\n            rtol: Relative tolerance for checking if coefficients are zero (Default: 1e-5).\n\n        Returns:\n            The simplified ``PauliSumOp``.\n        '
        if isinstance(self.coeff, (int, float, complex)):
            primitive = self.coeff * self.primitive
            return PauliSumOp(primitive.simplify(atol=atol, rtol=rtol))
        return PauliSumOp(self.primitive.simplify(atol=atol, rtol=rtol), self.coeff)

    def to_spmatrix(self) -> spmatrix:
        if False:
            return 10
        'Returns SciPy sparse matrix representation of the ``PauliSumOp``.\n\n        Returns:\n            CSR sparse matrix representation of the ``PauliSumOp``.\n\n        Raises:\n            ValueError: invalid parameters.\n        '
        return self.primitive.to_matrix(sparse=True) * self.coeff

    @classmethod
    def from_list(cls, pauli_list: List[Tuple[str, Union[complex, ParameterExpression]]], coeff: Union[complex, ParameterExpression]=1.0, dtype: type=complex) -> 'PauliSumOp':
        if False:
            i = 10
            return i + 15
        'Construct from a pauli_list with the form [(pauli_str, coeffs)]\n\n        Args:\n            pauli_list: A list of Tuple of pauli_str and coefficient.\n            coeff: A coefficient multiplying the primitive.\n            dtype: The dtype to use to construct the internal SparsePauliOp.\n                Defaults to ``complex``.\n\n        Returns:\n            The PauliSumOp constructed from the pauli_list.\n        '
        return cls(SparsePauliOp.from_list(pauli_list, dtype=dtype), coeff=coeff)

    def is_zero(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Return this operator is zero operator or not.\n        '
        op = self.reduce()
        primitive: SparsePauliOp = op.primitive
        return op.coeff == 1 and len(op) == 1 and (primitive.coeffs[0] == 0)

    def is_hermitian(self):
        if False:
            print('Hello World!')
        return np.isreal(self.coeffs).all() and np.all(self.primitive.paulis.phase == 0)