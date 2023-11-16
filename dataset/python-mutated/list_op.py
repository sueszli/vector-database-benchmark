"""ListOp Operator Class"""
from functools import reduce
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Sequence, Union, cast
import numpy as np
from scipy.sparse import spmatrix
from qiskit.circuit import ParameterExpression, QuantumCircuit
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.operator_base import OperatorBase
from qiskit.quantum_info import Statevector
from qiskit.utils import arithmetic
from qiskit.utils.deprecation import deprecate_func

class ListOp(OperatorBase):
    """
    Deprecated: A Class for manipulating List Operators, and parent class to ``SummedOp``,
    ``ComposedOp`` and ``TensoredOp``.

    List Operators are classes for storing and manipulating lists of Operators, State functions,
    or Measurements, and include some rule or ``combo_fn`` defining how the Operator functions
    of the list constituents should be combined to form to cumulative Operator function of the
    ``ListOp``. For example, a ``SummedOp`` has an addition-based ``combo_fn``, so once the
    Operators in its list are evaluated against some bitstring to produce a list of results,
    we know to add up those results to produce the final result of the ``SummedOp``'s
    evaluation. In theory, this ``combo_fn`` can be any function over classical complex values,
    but for convenience we've chosen for them to be defined over NumPy arrays and values. This way,
    large numbers of evaluations, such as after calling ``to_matrix`` on the list constituents,
    can be efficiently combined. While the combination function is defined over classical
    values, it should be understood as the operation by which each Operators' underlying
    function is combined to form the underlying Operator function of the ``ListOp``. In this
    way, the ``ListOps`` are the basis for constructing large and sophisticated Operators,
    State Functions, and Measurements.

    The base ``ListOp`` class is particularly interesting, as its ``combo_fn`` is "the identity
    list Operation". Meaning, if we understand the ``combo_fn`` as a function from a list of
    complex values to some output, one such function is returning the list as-is. This is
    powerful for constructing compact hierarchical Operators which return many measurements in
    multiple dimensional lists.
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, oplist: Sequence[OperatorBase], combo_fn: Optional[Callable]=None, coeff: Union[complex, ParameterExpression]=1.0, abelian: bool=False, grad_combo_fn: Optional[Callable]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            oplist: The list of ``OperatorBases`` defining this Operator\'s underlying function.\n            combo_fn: The recombination function to combine classical results of the\n                ``oplist`` Operators\' eval functions (e.g. sum). Default is lambda x: x.\n            coeff: A coefficient multiplying the operator\n            abelian: Indicates whether the Operators in ``oplist`` are known to mutually commute.\n            grad_combo_fn: The gradient of recombination function. If None, the gradient will\n                be computed automatically.\n            Note that the default "recombination function" lambda above is essentially the\n            identity - it accepts the list of values, and returns them in a list.\n        '
        super().__init__()
        self._oplist = self._check_input_types(oplist)
        self._combo_fn = combo_fn
        self._coeff = coeff
        self._abelian = abelian
        self._grad_combo_fn = grad_combo_fn

    def _check_input_types(self, oplist):
        if False:
            return 10
        if all((isinstance(x, OperatorBase) for x in oplist)):
            return list(oplist)
        else:
            badval = next((x for x in oplist if not isinstance(x, OperatorBase)))
            raise TypeError(f'ListOp expecting objects of type OperatorBase, got {badval}')

    def _state(self, coeff: Optional[Union[complex, ParameterExpression]]=None, combo_fn: Optional[Callable]=None, abelian: Optional[bool]=None, grad_combo_fn: Optional[Callable]=None) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        return {'coeff': coeff if coeff is not None else self.coeff, 'combo_fn': combo_fn if combo_fn is not None else self.combo_fn, 'abelian': abelian if abelian is not None else self.abelian, 'grad_combo_fn': grad_combo_fn if grad_combo_fn is not None else self.grad_combo_fn}

    @property
    def settings(self) -> Dict:
        if False:
            i = 10
            return i + 15
        'Return settings.'
        return {'oplist': self._oplist, 'combo_fn': self._combo_fn, 'coeff': self._coeff, 'abelian': self._abelian, 'grad_combo_fn': self._grad_combo_fn}

    @property
    def oplist(self) -> List[OperatorBase]:
        if False:
            i = 10
            return i + 15
        'The list of ``OperatorBases`` defining the underlying function of this\n        Operator.\n\n        Returns:\n            The Operators defining the ListOp\n        '
        return self._oplist

    @staticmethod
    def default_combo_fn(x: Any) -> Any:
        if False:
            print('Hello World!')
        'ListOp default combo function i.e. lambda x: x'
        return x

    @property
    def combo_fn(self) -> Callable:
        if False:
            for i in range(10):
                print('nop')
        "The function defining how to combine ``oplist`` (or Numbers, or NumPy arrays) to\n        produce the Operator's underlying function. For example, SummedOp's combination function\n        is to add all of the Operators in ``oplist``.\n\n        Returns:\n            The combination function.\n        "
        if self._combo_fn is None:
            return ListOp.default_combo_fn
        return self._combo_fn

    @property
    def grad_combo_fn(self) -> Optional[Callable]:
        if False:
            while True:
                i = 10
        'The gradient of ``combo_fn``.'
        return self._grad_combo_fn

    @property
    def abelian(self) -> bool:
        if False:
            return 10
        'Whether the Operators in ``oplist`` are known to commute with one another.\n\n        Returns:\n            A bool indicating whether the ``oplist`` is Abelian.\n        '
        return self._abelian

    @property
    def distributive(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Indicates whether the ListOp or subclass is distributive under composition.\n        ListOp and SummedOp are, meaning that (opv @ op) = (opv[0] @ op + opv[1] @ op)\n        (using plus for SummedOp, list for ListOp, etc.), while ComposedOp and TensoredOp\n        do not behave this way.\n\n        Returns:\n            A bool indicating whether the ListOp is distributive under composition.\n        '
        return True

    @property
    def coeff(self) -> Union[complex, ParameterExpression]:
        if False:
            return 10
        'The scalar coefficient multiplying the Operator.\n\n        Returns:\n            The coefficient.\n        '
        return self._coeff

    @property
    def coeffs(self) -> List[Union[complex, ParameterExpression]]:
        if False:
            while True:
                i = 10
        'Return a list of the coefficients of the operators listed.\n        Raises exception for nested Listops.\n        '
        if any((isinstance(op, ListOp) for op in self.oplist)):
            raise TypeError('Coefficients are not returned for nested ListOps.')
        return [self.coeff * op.coeff for op in self.oplist]

    def primitive_strings(self) -> Set[str]:
        if False:
            i = 10
            return i + 15
        return reduce(set.union, [op.primitive_strings() for op in self.oplist])

    @property
    def num_qubits(self) -> int:
        if False:
            i = 10
            return i + 15
        num_qubits0 = self.oplist[0].num_qubits
        if not all((num_qubits0 == op.num_qubits for op in self.oplist)):
            raise ValueError('Operators in ListOp have differing numbers of qubits.')
        return num_qubits0

    def add(self, other: OperatorBase) -> 'ListOp':
        if False:
            while True:
                i = 10
        if self == other:
            return self.mul(2.0)
        from .summed_op import SummedOp
        return SummedOp([self, other])

    def adjoint(self) -> 'ListOp':
        if False:
            while True:
                i = 10
        if self.__class__ == ListOp:
            return ListOp([op.adjoint() for op in self.oplist], **self._state(coeff=self.coeff.conjugate()))
        return self.__class__([op.adjoint() for op in self.oplist], coeff=self.coeff.conjugate(), abelian=self.abelian)

    def traverse(self, convert_fn: Callable, coeff: Optional[Union[complex, ParameterExpression]]=None) -> 'ListOp':
        if False:
            return 10
        'Apply the convert_fn to each node in the oplist.\n\n        Args:\n            convert_fn: The function to apply to the internal OperatorBase.\n            coeff: A coefficient to multiply by after applying convert_fn.\n                If it is None, self.coeff is used instead.\n\n        Returns:\n            The converted ListOp.\n        '
        if coeff is None:
            coeff = self.coeff
        if self.__class__ == ListOp:
            return ListOp([convert_fn(op) for op in self.oplist], **self._state(coeff=coeff))
        return self.__class__([convert_fn(op) for op in self.oplist], coeff=coeff, abelian=self.abelian)

    def equals(self, other: OperatorBase) -> bool:
        if False:
            i = 10
            return i + 15
        if not isinstance(other, type(self)) or not len(self.oplist) == len(other.oplist):
            return False
        return self.coeff == other.coeff and all((op1 == op2 for (op1, op2) in zip(self.oplist, other.oplist)))
    __array_priority__ = 10000

    def mul(self, scalar: Union[complex, ParameterExpression]) -> 'ListOp':
        if False:
            print('Hello World!')
        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not {} of type {}.'.format(scalar, type(scalar)))
        if self.__class__ == ListOp:
            return ListOp(self.oplist, **self._state(coeff=scalar * self.coeff))
        return self.__class__(self.oplist, coeff=scalar * self.coeff, abelian=self.abelian)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if False:
            while True:
                i = 10
        from .tensored_op import TensoredOp
        return TensoredOp([self, other])

    def tensorpower(self, other: int) -> Union[OperatorBase, int]:
        if False:
            while True:
                i = 10
        if other == 0:
            return 1
        if not isinstance(other, int) or other <= 0:
            raise TypeError('Tensorpower can only take positive int arguments')
        from .tensored_op import TensoredOp
        return TensoredOp([self] * other)

    def _expand_dim(self, num_qubits: int) -> 'ListOp':
        if False:
            i = 10
            return i + 15
        oplist = [op._expand_dim(num_qubits + self.num_qubits - op.num_qubits) for op in self.oplist]
        return ListOp(oplist, **self._state())

    def permute(self, permutation: List[int]) -> 'OperatorBase':
        if False:
            print('Hello World!')
        'Permute the qubits of the operator.\n\n        Args:\n            permutation: A list defining where each qubit should be permuted. The qubit at index\n                j should be permuted to position permutation[j].\n\n        Returns:\n            A new ListOp representing the permuted operator.\n\n        Raises:\n            OpflowError: if indices do not define a new index for each qubit.\n        '
        new_self = self
        circuit_size = max(permutation) + 1
        try:
            if self.num_qubits != len(permutation):
                raise OpflowError('New index must be defined for each qubit of the operator.')
        except ValueError:
            raise OpflowError('Permute is only possible if all operators in the ListOp have the same number of qubits.') from ValueError
        if self.num_qubits < circuit_size:
            new_self = self._expand_dim(circuit_size - self.num_qubits)
        qc = QuantumCircuit(circuit_size)
        permutation = list(filter(lambda x: x not in permutation, range(circuit_size))) + permutation
        transpositions = arithmetic.transpositions(permutation)
        for trans in transpositions:
            qc.swap(trans[0], trans[1])
        from ..primitive_ops.circuit_op import CircuitOp
        return CircuitOp(qc.reverse_ops()) @ new_self @ CircuitOp(qc)

    def compose(self, other: OperatorBase, permutation: Optional[List[int]]=None, front: bool=False) -> OperatorBase:
        if False:
            print('Hello World!')
        (new_self, other) = self._expand_shorter_operator_and_permute(other, permutation)
        new_self = cast(ListOp, new_self)
        if front:
            return other.compose(new_self)
        from .composed_op import ComposedOp
        return ComposedOp([new_self, other])

    def power(self, exponent: int) -> OperatorBase:
        if False:
            return 10
        if not isinstance(exponent, int) or exponent <= 0:
            raise TypeError('power can only take positive int arguments')
        from .composed_op import ComposedOp
        return ComposedOp([self] * exponent)

    def to_matrix(self, massive: bool=False) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        OperatorBase._check_massive('to_matrix', True, self.num_qubits, massive)
        mat = self.combo_fn(np.asarray([op.to_matrix(massive=massive) * self.coeff for op in self.oplist], dtype=object))
        return np.asarray(mat, dtype=complex)

    def to_spmatrix(self) -> Union[spmatrix, List[spmatrix]]:
        if False:
            while True:
                i = 10
        'Returns SciPy sparse matrix representation of the Operator.\n\n        Returns:\n            CSR sparse matrix representation of the Operator, or List thereof.\n        '
        return self.combo_fn([op.to_spmatrix() for op in self.oplist]) * self.coeff

    def eval(self, front: Optional[Union[str, Dict[str, complex], np.ndarray, OperatorBase, Statevector]]=None) -> Union[OperatorBase, complex]:
        if False:
            i = 10
            return i + 15
        '\n        Evaluate the Operator\'s underlying function, either on a binary string or another Operator.\n        A square binary Operator can be defined as a function taking a binary function to another\n        binary function. This method returns the value of that function for a given StateFn or\n        binary string. For example, ``op.eval(\'0110\').eval(\'1110\')`` can be seen as querying the\n        Operator\'s matrix representation by row 6 and column 14, and will return the complex\n        value at those "indices." Similarly for a StateFn, ``op.eval(\'1011\')`` will return the\n        complex value at row 11 of the vector representation of the StateFn, as all StateFns are\n        defined to be evaluated from Zero implicitly (i.e. it is as if ``.eval(\'0000\')`` is already\n        called implicitly to always "indexing" from column 0).\n\n        ListOp\'s eval recursively evaluates each Operator in ``oplist``,\n        and combines the results using the recombination function ``combo_fn``.\n\n        Args:\n            front: The bitstring, dict of bitstrings (with values being coefficients), or\n                StateFn to evaluated by the Operator\'s underlying function.\n\n        Returns:\n            The output of the ``oplist`` Operators\' evaluation function, combined with the\n            ``combo_fn``. If either self or front contain proper ``ListOps`` (not ListOp\n            subclasses), the result is an n-dimensional list of complex or StateFn results,\n            resulting from the recursive evaluation by each OperatorBase in the ListOps.\n\n        Raises:\n            NotImplementedError: Raised if called for a subclass which is not distributive.\n            TypeError: Operators with mixed hierarchies, such as a ListOp containing both\n                PrimitiveOps and ListOps, are not supported.\n            NotImplementedError: Attempting to call ListOp\'s eval from a non-distributive subclass.\n\n        '
        from ..state_fns.dict_state_fn import DictStateFn
        from ..state_fns.vector_state_fn import VectorStateFn
        from ..state_fns.sparse_vector_state_fn import SparseVectorStateFn
        if not self.distributive:
            raise NotImplementedError("ListOp's eval function is only defined for distributive ListOps.")
        evals = [op.eval(front) for op in self.oplist]
        if self._combo_fn is not None:
            if all((isinstance(op, DictStateFn) for op in evals)) or all((isinstance(op, VectorStateFn) for op in evals)) or all((isinstance(op, SparseVectorStateFn) for op in evals)):
                if not all((op.is_measurement == evals[0].is_measurement for op in evals)):
                    raise NotImplementedError('Combo_fn not yet supported for mixed measurement and non-measurement StateFns')
                result = self.combo_fn(evals)
                if isinstance(result, list):
                    multiplied = self.coeff * np.array(result)
                    return multiplied.tolist()
                return self.coeff * result
        if all((isinstance(op, OperatorBase) for op in evals)):
            return self.__class__(evals)
        elif any((isinstance(op, OperatorBase) for op in evals)):
            raise TypeError('Cannot handle mixed scalar and Operator eval results.')
        else:
            result = self.combo_fn(evals)
            if isinstance(result, list):
                multiplied = self.coeff * np.array(result)
                return multiplied.tolist()
            return self.coeff * result

    def exp_i(self) -> OperatorBase:
        if False:
            print('Hello World!')
        'Return an ``OperatorBase`` equivalent to an exponentiation of self * -i, e^(-i*op).'
        if type(self) == ListOp:
            return ListOp([op.exp_i() for op in self.oplist], **self._state(abelian=False))
        from ..evolutions.evolved_op import EvolvedOp
        return EvolvedOp(self)

    def log_i(self, massive: bool=False) -> OperatorBase:
        if False:
            for i in range(10):
                print('nop')
        'Return a ``MatrixOp`` equivalent to log(H)/-i for this operator H. This\n        function is the effective inverse of exp_i, equivalent to finding the Hermitian\n        Operator which produces self when exponentiated. For proper ListOps, applies ``log_i``\n        to all ops in oplist.\n        '
        if self.__class__.__name__ == ListOp.__name__:
            return ListOp([op.log_i(massive=massive) for op in self.oplist], **self._state(abelian=False))
        return self.to_matrix_op(massive=massive).log_i(massive=massive)

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        content_string = ',\n'.join([str(op) for op in self.oplist])
        main_string = '{}([\n{}\n])'.format(self.__class__.__name__, self._indent(content_string, indentation=self.INDENTATION))
        if self.abelian:
            main_string = 'Abelian' + main_string
        if self.coeff != 1.0:
            main_string = f'{self.coeff} * ' + main_string
        return main_string

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return '{}({}, coeff={}, abelian={})'.format(self.__class__.__name__, repr(self.oplist), self.coeff, self.abelian)

    @property
    def parameters(self):
        if False:
            return 10
        params = set()
        for op in self.oplist:
            params.update(op.parameters)
        if isinstance(self.coeff, ParameterExpression):
            params.update(self.coeff.parameters)
        return params

    def assign_parameters(self, param_dict: dict) -> OperatorBase:
        if False:
            i = 10
            return i + 15
        param_value = self.coeff
        if isinstance(self.coeff, ParameterExpression):
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                return ListOp([self.assign_parameters(param_dict) for param_dict in unrolled_dict])
            if self.coeff.parameters <= set(unrolled_dict.keys()):
                binds = {param: unrolled_dict[param] for param in self.coeff.parameters}
                param_value = float(self.coeff.bind(binds))
        return self.traverse(lambda x: x.assign_parameters(param_dict), coeff=param_value)

    def reduce(self) -> OperatorBase:
        if False:
            return 10
        reduced_ops = [op.reduce() for op in self.oplist]
        if self.__class__ == ListOp:
            return ListOp(reduced_ops, **self._state())
        return self.__class__(reduced_ops, coeff=self.coeff, abelian=self.abelian)

    def to_matrix_op(self, massive: bool=False) -> 'ListOp':
        if False:
            for i in range(10):
                print('nop')
        'Returns an equivalent Operator composed of only NumPy-based primitives, such as\n        ``MatrixOp`` and ``VectorStateFn``.'
        if self.__class__ == ListOp:
            return cast(ListOp, ListOp([op.to_matrix_op(massive=massive) for op in self.oplist], **self._state()).reduce())
        return cast(ListOp, self.__class__([op.to_matrix_op(massive=massive) for op in self.oplist], coeff=self.coeff, abelian=self.abelian).reduce())

    def to_circuit_op(self) -> OperatorBase:
        if False:
            print('Hello World!')
        'Returns an equivalent Operator composed of only QuantumCircuit-based primitives,\n        such as ``CircuitOp`` and ``CircuitStateFn``.'
        from ..state_fns.operator_state_fn import OperatorStateFn
        if self.__class__ == ListOp:
            return ListOp([op.to_circuit_op() if not isinstance(op, OperatorStateFn) else op for op in self.oplist], **self._state()).reduce()
        return self.__class__([op.to_circuit_op() if not isinstance(op, OperatorStateFn) else op for op in self.oplist], coeff=self.coeff, abelian=self.abelian).reduce()

    def to_pauli_op(self, massive: bool=False) -> 'ListOp':
        if False:
            print('Hello World!')
        'Returns an equivalent Operator composed of only Pauli-based primitives,\n        such as ``PauliOp``.'
        from ..state_fns.state_fn import StateFn
        if self.__class__ == ListOp:
            return ListOp([op.to_pauli_op(massive=massive) if not isinstance(op, StateFn) else op for op in self.oplist], **self._state()).reduce()
        return self.__class__([op.to_pauli_op(massive=massive) if not isinstance(op, StateFn) else op for op in self.oplist], coeff=self.coeff, abelian=self.abelian).reduce()

    def _is_empty(self):
        if False:
            return 10
        return len(self.oplist) == 0

    def __getitem__(self, offset: Union[int, slice]) -> OperatorBase:
        if False:
            print('Hello World!')
        'Allows array-indexing style access to the Operators in ``oplist``.\n\n        Args:\n            offset: The index of ``oplist`` desired.\n\n        Returns:\n            The ``OperatorBase`` at index ``offset`` of ``oplist``,\n            or another ListOp with the same properties as this one if offset is a slice.\n        '
        if isinstance(offset, int):
            return self.oplist[offset]
        if self.__class__ == ListOp:
            return ListOp(oplist=self._oplist[offset], **self._state())
        return self.__class__(oplist=self._oplist[offset], coeff=self._coeff, abelian=self._abelian)

    def __iter__(self) -> Iterator:
        if False:
            while True:
                i = 10
        'Returns an iterator over the operators in ``oplist``.\n\n        Returns:\n            An iterator over the operators in ``oplist``\n        '
        return iter(self.oplist)

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        'Length of ``oplist``.\n\n        Returns:\n            An int equal to the length of ``oplist``.\n        '
        return len(self.oplist)