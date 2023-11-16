"""StateFn Class"""
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.opflow.operator_base import OperatorBase
from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit.utils.deprecation import deprecate_func

class StateFn(OperatorBase):
    """
    Deprecated: A class for representing state functions and measurements.

    State functions are defined to be complex functions over a single binary string (as
    compared to an operator, which is defined as a function over two binary strings, or a
    function taking a binary function to another binary function). This function may be
    called by the eval() method.

    Measurements are defined to be functionals over StateFns, taking them to real values.
    Generally, this real value is interpreted to represent the probability of some classical
    state (binary string) being observed from a probabilistic or quantum system represented
    by a StateFn. This leads to the equivalent definition, which is that a measurement m is
    a function over binary strings producing StateFns, such that the probability of measuring
    a given binary string b from a system with StateFn f is equal to the inner
    product between f and m(b).

    NOTE: State functions here are not restricted to wave functions, as there is
    no requirement of normalization.
    """

    def __init_subclass__(cls):
        if False:
            print('Hello World!')
        cls.__new__ = lambda cls, *args, **kwargs: super().__new__(cls)

    @staticmethod
    def __new__(cls, primitive: Union[str, dict, Result, list, np.ndarray, Statevector, QuantumCircuit, Instruction, OperatorBase]=None, coeff: Union[complex, ParameterExpression]=1.0, is_measurement: bool=False) -> 'StateFn':
        if False:
            return 10
        "A factory method to produce the correct type of StateFn subclass\n        based on the primitive passed in. Primitive, coeff, and is_measurement arguments\n        are passed into subclass's init() as-is automatically by new().\n\n        Args:\n            primitive: The primitive which defines the behavior of the underlying State function.\n            coeff: A coefficient by which the state function is multiplied.\n            is_measurement: Whether the StateFn is a measurement operator\n\n        Returns:\n            The appropriate StateFn subclass for ``primitive``.\n\n        Raises:\n            TypeError: Unsupported primitive type passed.\n        "
        if cls.__name__ != StateFn.__name__:
            return super().__new__(cls)
        if isinstance(primitive, (str, dict, Result)):
            from .dict_state_fn import DictStateFn
            return DictStateFn.__new__(DictStateFn)
        if isinstance(primitive, (list, np.ndarray, Statevector)):
            from .vector_state_fn import VectorStateFn
            return VectorStateFn.__new__(VectorStateFn)
        if isinstance(primitive, (QuantumCircuit, Instruction)):
            from .circuit_state_fn import CircuitStateFn
            return CircuitStateFn.__new__(CircuitStateFn)
        if isinstance(primitive, OperatorBase):
            from .operator_state_fn import OperatorStateFn
            return OperatorStateFn.__new__(OperatorStateFn)
        raise TypeError('Unsupported primitive type {} passed into StateFn factory constructor'.format(type(primitive)))

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, primitive: Union[str, dict, Result, list, np.ndarray, Statevector, QuantumCircuit, Instruction, OperatorBase]=None, coeff: Union[complex, ParameterExpression]=1.0, is_measurement: bool=False) -> None:
        if False:
            return 10
        '\n        Args:\n            primitive: The primitive which defines the behavior of the underlying State function.\n            coeff: A coefficient by which the state function is multiplied.\n            is_measurement: Whether the StateFn is a measurement operator\n        '
        super().__init__()
        self._primitive = primitive
        self._is_measurement = is_measurement
        self._coeff = coeff

    @property
    def primitive(self):
        if False:
            return 10
        'The primitive which defines the behavior of the underlying State function.'
        return self._primitive

    @property
    def coeff(self) -> Union[complex, ParameterExpression]:
        if False:
            i = 10
            return i + 15
        'A coefficient by which the state function is multiplied.'
        return self._coeff

    @property
    def is_measurement(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Whether the StateFn object is a measurement Operator.'
        return self._is_measurement

    @property
    def settings(self) -> Dict:
        if False:
            i = 10
            return i + 15
        'Return settings.'
        return {'primitive': self._primitive, 'coeff': self._coeff, 'is_measurement': self._is_measurement}

    def primitive_strings(self) -> Set[str]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @property
    def num_qubits(self) -> int:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def add(self, other: OperatorBase) -> OperatorBase:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def adjoint(self) -> OperatorBase:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def _expand_dim(self, num_qubits: int) -> 'StateFn':
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def permute(self, permutation: List[int]) -> OperatorBase:
        if False:
            for i in range(10):
                print('nop')
        'Permute the qubits of the state function.\n\n        Args:\n            permutation: A list defining where each qubit should be permuted. The qubit at index\n                j of the circuit should be permuted to position permutation[j].\n\n        Returns:\n            A new StateFn containing the permuted primitive.\n        '
        raise NotImplementedError

    def equals(self, other: OperatorBase) -> bool:
        if False:
            print('Hello World!')
        if not isinstance(other, type(self)) or not self.coeff == other.coeff:
            return False
        return self.primitive == other.primitive

    def mul(self, scalar: Union[complex, ParameterExpression]) -> OperatorBase:
        if False:
            i = 10
            return i + 15
        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not {} of type {}.'.format(scalar, type(scalar)))
        if hasattr(self, 'from_operator'):
            return self.__class__(self.primitive, coeff=self.coeff * scalar, is_measurement=self.is_measurement, from_operator=self.from_operator)
        else:
            return self.__class__(self.primitive, coeff=self.coeff * scalar, is_measurement=self.is_measurement)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if False:
            print('Hello World!')
        "\n        Return tensor product between self and other, overloaded by ``^``.\n        Note: You must be conscious of Qiskit's big-endian bit printing\n        convention. Meaning, Plus.tensor(Zero)\n        produces a \\|+⟩ on qubit 0 and a \\|0⟩ on qubit 1, or \\|+⟩⨂\\|0⟩, but\n        would produce a QuantumCircuit like\n\n            \\|0⟩--\n            \\|+⟩--\n\n        Because Terra prints circuits and results with qubit 0\n        at the end of the string or circuit.\n\n        Args:\n            other: The ``OperatorBase`` to tensor product with self.\n\n        Returns:\n            An ``OperatorBase`` equivalent to the tensor product of self and other.\n        "
        raise NotImplementedError

    def tensorpower(self, other: int) -> Union[OperatorBase, int]:
        if False:
            return 10
        if not isinstance(other, int) or other <= 0:
            raise TypeError('Tensorpower can only take positive int arguments')
        temp = StateFn(self.primitive, coeff=self.coeff, is_measurement=self.is_measurement)
        for _ in range(other - 1):
            temp = temp.tensor(self)
        return temp

    def _expand_shorter_operator_and_permute(self, other: OperatorBase, permutation: Optional[List[int]]=None) -> Tuple[OperatorBase, OperatorBase]:
        if False:
            for i in range(10):
                print('nop')
        from ..operator_globals import Zero
        if self == StateFn({'0': 1}, is_measurement=True):
            return (StateFn('0' * other.num_qubits, is_measurement=True), other)
        elif other == Zero:
            return (self, StateFn('0' * self.num_qubits))
        return super()._expand_shorter_operator_and_permute(other, permutation)

    def to_matrix(self, massive: bool=False) -> np.ndarray:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def to_density_matrix(self, massive: bool=False) -> np.ndarray:
        if False:
            print('Hello World!')
        'Return matrix representing product of StateFn evaluated on pairs of basis states.\n        Overridden by child classes.\n\n        Args:\n            massive: Whether to allow large conversions, e.g. creating a matrix representing\n                over 16 qubits.\n\n        Returns:\n            The NumPy array representing the density matrix of the State function.\n\n        Raises:\n            ValueError: If massive is set to False, and exponentially large computation is needed.\n        '
        raise NotImplementedError

    def compose(self, other: OperatorBase, permutation: Optional[List[int]]=None, front: bool=False) -> OperatorBase:
        if False:
            return 10
        '\n        Composition (Linear algebra-style: A@B(x) = A(B(x))) is not well defined for states\n        in the binary function model, but is well defined for measurements.\n\n        Args:\n            other: The Operator to compose with self.\n            permutation: ``List[int]`` which defines permutation on other operator.\n            front: If front==True, return ``other.compose(self)``.\n\n        Returns:\n            An Operator equivalent to the function composition of self and other.\n\n        Raises:\n            ValueError: If self is not a measurement, it cannot be composed from the right.\n        '
        if not self.is_measurement and (not front):
            raise ValueError('Composition with a Statefunction in the first operand is not defined.')
        (new_self, other) = self._expand_shorter_operator_and_permute(other, permutation)
        if front:
            return other.compose(self)
        from ..primitive_ops.circuit_op import CircuitOp
        if self.primitive == {'0' * self.num_qubits: 1.0} and isinstance(other, CircuitOp):
            return StateFn(other.primitive, is_measurement=self.is_measurement, coeff=self.coeff * other.coeff)
        from ..list_ops.composed_op import ComposedOp
        if isinstance(other, ComposedOp):
            return ComposedOp([new_self] + other.oplist, coeff=new_self.coeff * other.coeff)
        return ComposedOp([new_self, other])

    def power(self, exponent: int) -> OperatorBase:
        if False:
            while True:
                i = 10
        'Compose with Self Multiple Times, undefined for StateFns.\n\n        Args:\n            exponent: The number of times to compose self with self.\n\n        Raises:\n            ValueError: This function is not defined for StateFns.\n        '
        raise ValueError('Composition power over Statefunctions or Measurements is not defined.')

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return '{}({})'.format('StateFunction' if not self.is_measurement else 'Measurement', self.coeff)
        else:
            return '{}({}) * {}'.format('StateFunction' if not self.is_measurement else 'Measurement', self.coeff, prim_str)

    def __repr__(self) -> str:
        if False:
            return 10
        return '{}({}, coeff={}, is_measurement={})'.format(self.__class__.__name__, repr(self.primitive), self.coeff, self.is_measurement)

    def eval(self, front: Optional[Union[str, Dict[str, complex], np.ndarray, OperatorBase, Statevector]]=None) -> Union[OperatorBase, complex]:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @property
    def parameters(self):
        if False:
            print('Hello World!')
        params = set()
        if isinstance(self.primitive, (OperatorBase, QuantumCircuit)):
            params.update(self.primitive.parameters)
        if isinstance(self.coeff, ParameterExpression):
            params.update(self.coeff.parameters)
        return params

    def assign_parameters(self, param_dict: dict) -> OperatorBase:
        if False:
            print('Hello World!')
        param_value = self.coeff
        if isinstance(self.coeff, ParameterExpression):
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                from ..list_ops.list_op import ListOp
                return ListOp([self.assign_parameters(param_dict) for param_dict in unrolled_dict])
            if self.coeff.parameters <= set(unrolled_dict.keys()):
                binds = {param: unrolled_dict[param] for param in self.coeff.parameters}
                param_value = float(self.coeff.bind(binds))
        return self.traverse(lambda x: x.assign_parameters(param_dict), coeff=param_value)

    def reduce(self) -> OperatorBase:
        if False:
            for i in range(10):
                print('nop')
        return self

    def traverse(self, convert_fn: Callable, coeff: Optional[Union[complex, ParameterExpression]]=None) -> OperatorBase:
        if False:
            while True:
                i = 10
        '\n        Apply the convert_fn to the internal primitive if the primitive is an Operator (as in\n        the case of ``OperatorStateFn``). Otherwise do nothing. Used by converters.\n\n        Args:\n            convert_fn: The function to apply to the internal OperatorBase.\n            coeff: A coefficient to multiply by after applying convert_fn.\n                If it is None, self.coeff is used instead.\n\n        Returns:\n            The converted StateFn.\n        '
        if coeff is None:
            coeff = self.coeff
        if isinstance(self.primitive, OperatorBase):
            return StateFn(convert_fn(self.primitive), coeff=coeff, is_measurement=self.is_measurement)
        else:
            return self

    def to_matrix_op(self, massive: bool=False) -> OperatorBase:
        if False:
            while True:
                i = 10
        'Return a ``VectorStateFn`` for this ``StateFn``.\n\n        Args:\n            massive: Whether to allow large conversions, e.g. creating a matrix representing\n                over 16 qubits.\n\n        Returns:\n            A VectorStateFn equivalent to self.\n        '
        from .vector_state_fn import VectorStateFn
        return VectorStateFn(self.to_matrix(massive=massive), is_measurement=self.is_measurement)

    def to_circuit_op(self) -> OperatorBase:
        if False:
            for i in range(10):
                print('nop')
        'Returns a ``CircuitOp`` equivalent to this Operator.'
        raise NotImplementedError

    def sample(self, shots: int=1024, massive: bool=False, reverse_endianness: bool=False) -> Dict[str, float]:
        if False:
            for i in range(10):
                print('nop')
        "Sample the state function as a normalized probability distribution. Returns dict of\n        bitstrings in order of probability, with values being probability.\n\n        Args:\n            shots: The number of samples to take to approximate the State function.\n            massive: Whether to allow large conversions, e.g. creating a matrix representing\n                over 16 qubits.\n            reverse_endianness: Whether to reverse the endianness of the bitstrings in the return\n                dict to match Terra's big-endianness.\n\n        Returns:\n            A dict containing pairs sampled strings from the State function and sampling\n            frequency divided by shots.\n        "
        raise NotImplementedError