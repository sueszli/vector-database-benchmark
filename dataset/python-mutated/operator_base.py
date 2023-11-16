"""OperatorBase Class"""
import itertools
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union, cast
import numpy as np
from scipy.sparse import csr_matrix, spmatrix
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.mixins import StarAlgebraMixin, TensorMixin
from qiskit.quantum_info import Statevector
from qiskit.utils import algorithm_globals
from qiskit.utils.deprecation import deprecate_func

class OperatorBase(StarAlgebraMixin, TensorMixin, ABC):
    """Deprecated: A base class for all Operators: PrimitiveOps, StateFns, ListOps, etc. Operators are
    defined as functions which take one complex binary function to another. These complex binary
    functions are represented by StateFns, which are themselves a special class of Operators
    taking only the ``Zero`` StateFn to the complex binary function they represent.

    Operators can be used to construct complicated functions and computation, and serve as the
    building blocks for algorithms.

    """
    INDENTATION = '  '
    _count = itertools.count()

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._instance_id = next(self._count)

    @property
    @abstractmethod
    def settings(self) -> Dict:
        if False:
            return 10
        'Return settings of this object in a dictionary.\n\n        You can, for example, use this ``settings`` dictionary to serialize the\n        object in JSON format, if the JSON encoder you use supports all types in\n        the dictionary.\n\n        Returns:\n            Object settings in a dictionary.\n        '
        raise NotImplementedError

    @property
    def instance_id(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Return the unique instance id.'
        return self._instance_id

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        "The number of qubits over which the Operator is defined. If\n        ``op.num_qubits == 5``, then ``op.eval('1' * 5)`` will be valid, but\n        ``op.eval('11')`` will not.\n\n        Returns:\n            The number of qubits accepted by the Operator's underlying function.\n        "
        raise NotImplementedError

    @abstractmethod
    def primitive_strings(self) -> Set[str]:
        if False:
            while True:
                i = 10
        "Return a set of strings describing the primitives contained in the Operator. For\n        example, ``{'QuantumCircuit', 'Pauli'}``. For hierarchical Operators, such as ``ListOps``,\n        this can help illuminate the primitives represented in the various recursive levels,\n        and therefore which conversions can be applied.\n\n        Returns:\n            A set of strings describing the primitives contained within the Operator.\n        "
        raise NotImplementedError

    @abstractmethod
    def eval(self, front: Optional[Union[str, Dict[str, complex], np.ndarray, 'OperatorBase', Statevector]]=None) -> Union['OperatorBase', complex]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Evaluate the Operator\'s underlying function, either on a binary string or another Operator.\n        A square binary Operator can be defined as a function taking a binary function to another\n        binary function. This method returns the value of that function for a given StateFn or\n        binary string. For example, ``op.eval(\'0110\').eval(\'1110\')`` can be seen as querying the\n        Operator\'s matrix representation by row 6 and column 14, and will return the complex\n        value at those "indices." Similarly for a StateFn, ``op.eval(\'1011\')`` will return the\n        complex value at row 11 of the vector representation of the StateFn, as all StateFns are\n        defined to be evaluated from Zero implicitly (i.e. it is as if ``.eval(\'0000\')`` is already\n        called implicitly to always "indexing" from column 0).\n\n        If ``front`` is None, the matrix-representation of the operator is returned.\n\n        Args:\n            front: The bitstring, dict of bitstrings (with values being coefficients), or\n                StateFn to evaluated by the Operator\'s underlying function, or None.\n\n        Returns:\n            The output of the Operator\'s evaluation function. If self is a ``StateFn``, the result\n            is a float or complex. If self is an Operator (``PrimitiveOp, ComposedOp, SummedOp,\n            EvolvedOp,`` etc.), the result is a StateFn.\n            If ``front`` is None, the matrix-representation of the operator is returned, which\n            is a ``MatrixOp`` for the operators and a ``VectorStateFn`` for state-functions.\n            If either self or front contain proper\n            ``ListOps`` (not ListOp subclasses), the result is an n-dimensional list of complex\n            or StateFn results, resulting from the recursive evaluation by each OperatorBase\n            in the ListOps.\n\n        '
        raise NotImplementedError

    @abstractmethod
    def reduce(self):
        if False:
            return 10
        'Try collapsing the Operator structure, usually after some type of conversion,\n        e.g. trying to add Operators in a SummedOp or delete needless IGates in a CircuitOp.\n        If no reduction is available, just returns self.\n\n        Returns:\n            The reduced ``OperatorBase``.\n        '
        raise NotImplementedError

    @abstractmethod
    def to_matrix(self, massive: bool=False) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        "Return NumPy representation of the Operator. Represents the evaluation of\n        the Operator's underlying function on every combination of basis binary strings.\n        Warn if more than 16 qubits to force having to set ``massive=True`` if such a\n        large vector is desired.\n\n        Returns:\n              The NumPy ``ndarray`` equivalent to this Operator.\n        "
        raise NotImplementedError

    @abstractmethod
    def to_matrix_op(self, massive: bool=False) -> 'OperatorBase':
        if False:
            return 10
        'Returns a ``MatrixOp`` equivalent to this Operator.'
        raise NotImplementedError

    @abstractmethod
    def to_circuit_op(self) -> 'OperatorBase':
        if False:
            print('Hello World!')
        'Returns a ``CircuitOp`` equivalent to this Operator.'
        raise NotImplementedError

    def to_spmatrix(self) -> spmatrix:
        if False:
            print('Hello World!')
        "Return SciPy sparse matrix representation of the Operator. Represents the evaluation of\n        the Operator's underlying function on every combination of basis binary strings.\n\n        Returns:\n              The SciPy ``spmatrix`` equivalent to this Operator.\n        "
        return csr_matrix(self.to_matrix())

    def is_hermitian(self) -> bool:
        if False:
            return 10
        'Return True if the operator is hermitian.\n\n        Returns: Boolean value\n        '
        return (self.to_spmatrix() != self.to_spmatrix().getH()).nnz == 0

    @staticmethod
    def _indent(lines: str, indentation: str=INDENTATION) -> str:
        if False:
            return 10
        'Indented representation to allow pretty representation of nested operators.'
        indented_str = indentation + lines.replace('\n', f'\n{indentation}')
        if indented_str.endswith(f'\n{indentation}'):
            indented_str = indented_str[:-len(indentation)]
        return indented_str

    @abstractmethod
    def add(self, other: 'OperatorBase') -> 'OperatorBase':
        if False:
            print('Hello World!')
        "Return Operator addition of self and other, overloaded by ``+``.\n\n        Args:\n            other: An ``OperatorBase`` with the same number of qubits as self, and in the same\n                'Operator', 'State function', or 'Measurement' category as self (i.e. the same type\n                of underlying function).\n\n        Returns:\n            An ``OperatorBase`` equivalent to the sum of self and other.\n        "
        raise NotImplementedError

    def neg(self) -> 'OperatorBase':
        if False:
            return 10
        "Return the Operator's negation, effectively just multiplying by -1.0,\n        overloaded by ``-``.\n\n        Returns:\n            An ``OperatorBase`` equivalent to the negation of self.\n        "
        return self.mul(-1.0)

    @abstractmethod
    def adjoint(self) -> 'OperatorBase':
        if False:
            i = 10
            return i + 15
        "Return a new Operator equal to the Operator's adjoint (conjugate transpose),\n        overloaded by ``~``. For StateFns, this also turns the StateFn into a measurement.\n\n        Returns:\n            An ``OperatorBase`` equivalent to the adjoint of self.\n        "
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        'Overload ``==`` operation to evaluate equality between Operators.\n\n        Args:\n            other: The ``OperatorBase`` to compare to self.\n\n        Returns:\n            A bool equal to the equality of self and other.\n        '
        if not isinstance(other, OperatorBase):
            return NotImplemented
        return self.equals(cast(OperatorBase, other))

    @abstractmethod
    def equals(self, other: 'OperatorBase') -> bool:
        if False:
            while True:
                i = 10
        '\n        Evaluate Equality between Operators, overloaded by ``==``. Only returns True if self and\n        other are of the same representation (e.g. a DictStateFn and CircuitStateFn will never be\n        equal, even if their vector representations are equal), their underlying primitives are\n        equal (this means for ListOps, OperatorStateFns, or EvolvedOps the equality is evaluated\n        recursively downwards), and their coefficients are equal.\n\n        Args:\n            other: The ``OperatorBase`` to compare to self.\n\n        Returns:\n            A bool equal to the equality of self and other.\n\n        '
        raise NotImplementedError

    @abstractmethod
    def mul(self, scalar: Union[complex, ParameterExpression]) -> 'OperatorBase':
        if False:
            return 10
        "\n        Returns the scalar multiplication of the Operator, overloaded by ``*``, including\n        support for Terra's ``Parameters``, which can be bound to values later (via\n        ``bind_parameters``).\n\n        Args:\n            scalar: The real or complex scalar by which to multiply the Operator,\n                or the ``ParameterExpression`` to serve as a placeholder for a scalar factor.\n\n        Returns:\n            An ``OperatorBase`` equivalent to product of self and scalar.\n        "
        raise NotImplementedError

    @abstractmethod
    def tensor(self, other: 'OperatorBase') -> 'OperatorBase':
        if False:
            return 10
        "Return tensor product between self and other, overloaded by ``^``.\n        Note: You must be conscious of Qiskit's big-endian bit printing convention.\n        Meaning, X.tensor(Y) produces an X on qubit 0 and an Y on qubit 1, or X⨂Y,\n        but would produce a QuantumCircuit which looks like\n\n            -[Y]-\n            -[X]-\n\n        Because Terra prints circuits and results with qubit 0 at the end of the string\n        or circuit.\n\n        Args:\n            other: The ``OperatorBase`` to tensor product with self.\n\n        Returns:\n            An ``OperatorBase`` equivalent to the tensor product of self and other.\n        "
        raise NotImplementedError

    @abstractmethod
    def tensorpower(self, other: int) -> Union['OperatorBase', int]:
        if False:
            i = 10
            return i + 15
        'Return tensor product with self multiple times, overloaded by ``^``.\n\n        Args:\n            other: The int number of times to tensor product self with itself via ``tensorpower``.\n\n        Returns:\n            An ``OperatorBase`` equivalent to the tensorpower of self by other.\n        '
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self):
        if False:
            return 10
        'Return a set of Parameter objects contained in the Operator.'
        raise NotImplementedError

    @abstractmethod
    def assign_parameters(self, param_dict: Dict[ParameterExpression, Union[complex, ParameterExpression, List[Union[complex, ParameterExpression]]]]) -> 'OperatorBase':
        if False:
            return 10
        "Binds scalar values to any Terra ``Parameters`` in the coefficients or primitives of\n        the Operator, or substitutes one ``Parameter`` for another. This method differs from\n        Terra's ``assign_parameters`` in that it also supports lists of values to assign for a\n        give ``Parameter``, in which case self will be copied for each parameterization in the\n        binding list(s), and all the copies will be returned in an ``OpList``. If lists of\n        parameterizations are used, every ``Parameter`` in the param_dict must have the same\n        length list of parameterizations.\n\n        Args:\n            param_dict: The dictionary of ``Parameters`` to replace, and values or lists of\n                values by which to replace them.\n\n        Returns:\n            The ``OperatorBase`` with the ``Parameters`` in self replaced by the\n            values or ``Parameters`` in param_dict. If param_dict contains parameterization lists,\n            this ``OperatorBase`` is an ``OpList``.\n        "
        raise NotImplementedError

    @abstractmethod
    def _expand_dim(self, num_qubits: int) -> 'OperatorBase':
        if False:
            i = 10
            return i + 15
        'Expands the operator with identity operator of dimension 2**num_qubits.\n\n        Returns:\n            Operator corresponding to self.tensor(identity_operator), where dimension of identity\n            operator is 2 ** num_qubits.\n        '
        raise NotImplementedError

    @abstractmethod
    def permute(self, permutation: List[int]) -> 'OperatorBase':
        if False:
            for i in range(10):
                print('nop')
        'Permutes the qubits of the operator.\n\n        Args:\n            permutation: A list defining where each qubit should be permuted. The qubit at index\n                j should be permuted to position permutation[j].\n\n        Returns:\n            A new OperatorBase containing the permuted operator.\n\n        Raises:\n            OpflowError: if indices do not define a new index for each qubit.\n        '
        raise NotImplementedError

    def bind_parameters(self, param_dict: Dict[ParameterExpression, Union[complex, ParameterExpression, List[Union[complex, ParameterExpression]]]]) -> 'OperatorBase':
        if False:
            print('Hello World!')
        '\n        Same as assign_parameters, but maintained for consistency with QuantumCircuit in\n        Terra (which has both assign_parameters and bind_parameters).\n        '
        return self.assign_parameters(param_dict)

    @staticmethod
    def _unroll_param_dict(value_dict: Dict[Union[ParameterExpression, ParameterVector], Union[complex, List[complex]]]) -> Union[Dict[ParameterExpression, complex], List[Dict[ParameterExpression, complex]]]:
        if False:
            return 10
        'Unrolls the ParameterVectors in a param_dict into separate Parameters, and unrolls\n        parameterization value lists into separate param_dicts without list nesting.'
        unrolled_value_dict = {}
        for (param, value) in value_dict.items():
            if isinstance(param, ParameterExpression):
                unrolled_value_dict[param] = value
            if isinstance(param, ParameterVector) and isinstance(value, (list, np.ndarray)):
                if not len(param) == len(value):
                    raise ValueError('ParameterVector {} has length {}, which differs from value list {} of len {}'.format(param, len(param), value, len(value)))
                unrolled_value_dict.update(zip(param, value))
        if isinstance(list(unrolled_value_dict.values())[0], list):
            unrolled_value_dict_list = []
            try:
                for i in range(len(list(unrolled_value_dict.values())[0])):
                    unrolled_value_dict_list.append(OperatorBase._get_param_dict_for_index(unrolled_value_dict, i))
                return unrolled_value_dict_list
            except IndexError as ex:
                raise OpflowError('Parameter binding lists must all be the same length.') from ex
        return unrolled_value_dict

    @staticmethod
    def _get_param_dict_for_index(unrolled_dict: Dict[ParameterExpression, List[complex]], i: int):
        if False:
            for i in range(10):
                print('nop')
        'Gets a single non-list-nested param_dict for a given list index from a nested one.'
        return {k: v[i] for (k, v) in unrolled_dict.items()}

    def _expand_shorter_operator_and_permute(self, other: 'OperatorBase', permutation: Optional[List[int]]=None) -> Tuple['OperatorBase', 'OperatorBase']:
        if False:
            return 10
        if permutation is not None:
            other = other.permute(permutation)
        new_self = self
        if not self.num_qubits == other.num_qubits:
            from .operator_globals import Zero
            if other == Zero:
                other = Zero.__class__('0' * self.num_qubits)
            elif other.num_qubits < self.num_qubits:
                other = other._expand_dim(self.num_qubits - other.num_qubits)
            elif other.num_qubits > self.num_qubits:
                new_self = self._expand_dim(other.num_qubits - self.num_qubits)
        return (new_self, other)

    def copy(self) -> 'OperatorBase':
        if False:
            return 10
        'Return a deep copy of the Operator.'
        return deepcopy(self)

    @abstractmethod
    def compose(self, other: 'OperatorBase', permutation: Optional[List[int]]=None, front: bool=False) -> 'OperatorBase':
        if False:
            while True:
                i = 10
        'Return Operator Composition between self and other (linear algebra-style:\n        A@B(x) = A(B(x))), overloaded by ``@``.\n\n        Note: You must be conscious of Quantum Circuit vs. Linear Algebra ordering\n        conventions. Meaning, X.compose(Y)\n        produces an X∘Y on qubit 0, but would produce a QuantumCircuit which looks like\n\n            -[Y]-[X]-\n\n        Because Terra prints circuits with the initial state at the left side of the circuit.\n\n        Args:\n            other: The ``OperatorBase`` with which to compose self.\n            permutation: ``List[int]`` which defines permutation on other operator.\n            front: If front==True, return ``other.compose(self)``.\n\n        Returns:\n            An ``OperatorBase`` equivalent to the function composition of self and other.\n        '
        raise NotImplementedError

    @staticmethod
    def _check_massive(method: str, matrix: bool, num_qubits: int, massive: bool) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Checks if matrix or vector generated will be too large.\n\n        Args:\n            method: Name of the calling method\n            matrix: True if object is matrix, otherwise vector\n            num_qubits: number of qubits\n            massive: True if it is ok to proceed with large matrix\n\n        Raises:\n            ValueError: Massive is False and number of qubits is greater than 16\n        '
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            if num_qubits > 16 and (not massive) and (not algorithm_globals.massive):
                dim = 2 ** num_qubits
                if matrix:
                    obj_type = 'matrix'
                    dimensions = f'{dim}x{dim}'
                else:
                    obj_type = 'vector'
                    dimensions = f'{dim}'
                raise ValueError(f"'{method}' will return an exponentially large {obj_type}, in this case '{dimensions}' elements. Set algorithm_globals.massive=True or the method argument massive=True if you want to proceed.")

    @abstractmethod
    def __str__(self) -> str:
        if False:
            print('Hello World!')
        raise NotImplementedError