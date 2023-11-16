"""A wrapper class for the purposes of validating modifications to
QuantumCircuit.data while maintaining the interface of a python list."""
from collections.abc import MutableSequence
from typing import Tuple, Iterable, Optional
from .exceptions import CircuitError
from .instruction import Instruction
from .operation import Operation
from .quantumregister import Qubit
from .classicalregister import Clbit

class CircuitInstruction:
    """A single instruction in a :class:`.QuantumCircuit`, comprised of the :attr:`operation` and
    various operands.

    .. note::

        There is some possible confusion in the names of this class, :class:`~.circuit.Instruction`,
        and :class:`~.circuit.Operation`, and this class's attribute :attr:`operation`.  Our
        preferred terminology is by analogy to assembly languages, where an "instruction" is made up
        of an "operation" and its "operands".

        Historically, :class:`~.circuit.Instruction` came first, and originally contained the qubits
        it operated on and any parameters, so it was a true "instruction".  Over time,
        :class:`.QuantumCircuit` became responsible for tracking qubits and clbits, and the class
        became better described as an "operation".  Changing the name of such a core object would be
        a very unpleasant API break for users, and so we have stuck with it.

        This class was created to provide a formal "instruction" context object in
        :class:`.QuantumCircuit.data`, which had long been made of ad-hoc tuples.  With this, and
        the advent of the :class:`~.circuit.Operation` interface for adding more complex objects to
        circuits, we took the opportunity to correct the historical naming.  For the time being,
        this leads to an awkward case where :attr:`.CircuitInstruction.operation` is often an
        :class:`~.circuit.Instruction` instance (:class:`~.circuit.Instruction` implements the
        :class:`.Operation` interface), but as the :class:`.Operation` interface gains more use,
        this confusion will hopefully abate.

    .. warning::

        This is a lightweight internal class and there is minimal error checking; you must respect
        the type hints when using it.  It is the user's responsibility to ensure that direct
        mutations of the object do not invalidate the types, nor the restrictions placed on it by
        its context.  Typically this will mean, for example, that :attr:`qubits` must be a sequence
        of distinct items, with no duplicates.
    """
    __slots__ = ('operation', 'qubits', 'clbits')
    operation: Operation
    'The logical operation that this instruction represents an execution of.'
    qubits: Tuple[Qubit, ...]
    'A sequence of the qubits that the operation is applied to.'
    clbits: Tuple[Clbit, ...]
    'A sequence of the classical bits that this operation reads from or writes to.'

    def __init__(self, operation: Operation, qubits: Iterable[Qubit]=(), clbits: Iterable[Clbit]=()):
        if False:
            return 10
        self.operation = operation
        self.qubits = tuple(qubits)
        self.clbits = tuple(clbits)

    def copy(self) -> 'CircuitInstruction':
        if False:
            for i in range(10):
                print('nop')
        'Return a shallow copy of the :class:`CircuitInstruction`.'
        return self.__class__(operation=self.operation, qubits=self.qubits, clbits=self.clbits)

    def replace(self, operation: Optional[Operation]=None, qubits: Optional[Iterable[Qubit]]=None, clbits: Optional[Iterable[Clbit]]=None) -> 'CircuitInstruction':
        if False:
            print('Hello World!')
        'Return a new :class:`CircuitInstruction` with the given fields replaced.'
        return self.__class__(operation=self.operation if operation is None else operation, qubits=self.qubits if qubits is None else qubits, clbits=self.clbits if clbits is None else clbits)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'{type(self).__name__}(operation={self.operation!r}, qubits={self.qubits!r}, clbits={self.clbits!r})'

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, type(self)):
            return self.clbits == other.clbits and self.qubits == other.qubits and (self.operation == other.operation)
        if isinstance(other, tuple):
            return self._legacy_format() == other
        return NotImplemented

    def _legacy_format(self):
        if False:
            i = 10
            return i + 15
        return (self.operation, list(self.qubits), list(self.clbits))

    def __getitem__(self, key):
        if False:
            return 10
        return self._legacy_format()[key]

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self._legacy_format())

    def __len__(self):
        if False:
            print('Hello World!')
        return 3

class QuantumCircuitData(MutableSequence):
    """A wrapper class for the purposes of validating modifications to
    QuantumCircuit.data while maintaining the interface of a python list."""

    def __init__(self, circuit):
        if False:
            print('Hello World!')
        self._circuit = circuit

    def __getitem__(self, i):
        if False:
            print('Hello World!')
        return self._circuit._data[i]

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        if isinstance(value, CircuitInstruction):
            (operation, qargs, cargs) = (value.operation, value.qubits, value.clbits)
        else:
            (operation, qargs, cargs) = value
        value = self._resolve_legacy_value(operation, qargs, cargs)
        self._circuit._data[key] = value
        if isinstance(value.operation, Instruction):
            self._circuit._update_parameter_table(value)

    def _resolve_legacy_value(self, operation, qargs, cargs) -> CircuitInstruction:
        if False:
            while True:
                i = 10
        'Resolve the old-style 3-tuple into the new :class:`CircuitInstruction` type.'
        if not isinstance(operation, Operation) and hasattr(operation, 'to_instruction'):
            operation = operation.to_instruction()
        if not isinstance(operation, Operation):
            raise CircuitError('object is not an Operation.')
        expanded_qargs = [self._circuit.qbit_argument_conversion(qarg) for qarg in qargs or []]
        expanded_cargs = [self._circuit.cbit_argument_conversion(carg) for carg in cargs or []]
        if isinstance(operation, Instruction):
            broadcast_args = list(operation.broadcast_arguments(expanded_qargs, expanded_cargs))
        else:
            broadcast_args = list(Instruction.broadcast_arguments(operation, expanded_qargs, expanded_cargs))
        if len(broadcast_args) > 1:
            raise CircuitError('QuantumCircuit.data modification does not support argument broadcasting.')
        (qargs, cargs) = broadcast_args[0]
        self._circuit._check_dups(qargs)
        return CircuitInstruction(operation, tuple(qargs), tuple(cargs))

    def insert(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self._circuit._data.insert(index, None)
        try:
            self[index] = value
        except CircuitError:
            del self._circuit._data[index]
            raise

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self._circuit._data)

    def __delitem__(self, i):
        if False:
            print('Hello World!')
        del self._circuit._data[i]

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._circuit._data)

    def __cast(self, other):
        if False:
            while True:
                i = 10
        return other._circuit._data if isinstance(other, QuantumCircuitData) else other

    def __repr__(self):
        if False:
            return 10
        return repr(self._circuit._data)

    def __lt__(self, other):
        if False:
            print('Hello World!')
        return self._circuit._data < self.__cast(other)

    def __le__(self, other):
        if False:
            print('Hello World!')
        return self._circuit._data <= self.__cast(other)

    def __eq__(self, other):
        if False:
            return 10
        return self._circuit._data == self.__cast(other)

    def __gt__(self, other):
        if False:
            while True:
                i = 10
        return self._circuit._data > self.__cast(other)

    def __ge__(self, other):
        if False:
            return 10
        return self._circuit._data >= self.__cast(other)

    def __add__(self, other):
        if False:
            print('Hello World!')
        return self._circuit._data + self.__cast(other)

    def __radd__(self, other):
        if False:
            print('Hello World!')
        return self.__cast(other) + self._circuit._data

    def __mul__(self, n):
        if False:
            print('Hello World!')
        return self._circuit._data * n

    def __rmul__(self, n):
        if False:
            i = 10
            return i + 15
        return n * self._circuit._data

    def sort(self, *args, **kwargs):
        if False:
            return 10
        'In-place stable sort. Accepts arguments of list.sort.'
        self._circuit._data.sort(*args, **kwargs)

    def copy(self):
        if False:
            while True:
                i = 10
        'Returns a shallow copy of instruction list.'
        return self._circuit._data.copy()