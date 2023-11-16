"""
Quantum bit and Classical bit objects.
"""
import copy
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.deprecation import deprecate_func

class Bit:
    """Implement a generic bit.

    .. note::
        This class should not be instantiated directly. This is just a superclass
        for :class:`~.Clbit` and :class:`~.circuit.Qubit`.

    """
    __slots__ = {'_register', '_index', '_hash', '_repr'}

    def __init__(self, register=None, index=None):
        if False:
            i = 10
            return i + 15
        'Create a new generic bit.'
        if (register, index) == (None, None):
            self._register = None
            self._index = None
            self._hash = object.__hash__(self)
        else:
            try:
                index = int(index)
            except Exception as ex:
                raise CircuitError(f'index needs to be castable to an int: type {type(index)} was provided') from ex
            if index < 0:
                index += register.size
            if index >= register.size:
                raise CircuitError(f'index must be under the size of the register: {index} was provided')
            self._register = register
            self._index = index
            self._hash = hash((self._register, self._index))
            self._repr = f'{self.__class__.__name__}({self._register}, {self._index})'

    @property
    @deprecate_func(is_property=True, since='0.17', package_name='qiskit-terra', additional_msg='Instead, use :meth:`~qiskit.circuit.quantumcircuit.QuantumCircuit.find_bit` to find all the containing registers within a circuit and the index of the bit within the circuit.')
    def register(self):
        if False:
            print('Hello World!')
        'Get the register of an old-style bit.\n\n        In modern Qiskit Terra (version 0.17+), bits are the fundamental object and registers are\n        aliases to collections of bits.  A bit can be in many registers depending on the circuit, so\n        a single containing register is no longer a property of a bit.  It is an error to access\n        this attribute on bits that were not constructed as "owned" by a register.'
        if (self._register, self._index) == (None, None):
            raise CircuitError('Attempt to query register of a new-style Bit.')
        return self._register

    @property
    @deprecate_func(is_property=True, since='0.17', package_name='qiskit-terra', additional_msg='Instead, use :meth:`~qiskit.circuit.quantumcircuit.QuantumCircuit.find_bit` to find all the containing registers within a circuit and the index of the bit within the circuit.')
    def index(self):
        if False:
            while True:
                i = 10
        'Get the index of an old-style bit in the register that owns it.\n\n        In modern Qiskit Terra (version 0.17+), bits are the fundamental object and registers are\n        aliases to collections of bits.  A bit can be in many registers depending on the circuit, so\n        a single containing register is no longer a property of a bit.  It is an error to access\n        this attribute on bits that were not constructed as "owned" by a register.'
        if (self._register, self._index) == (None, None):
            raise CircuitError('Attempt to query index of a new-style Bit.')
        return self._index

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Return the official string representing the bit.'
        if (self._register, self._index) == (None, None):
            return object.__repr__(self)
        return self._repr

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._hash

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if (self._register, self._index) == (None, None):
            return other is self
        try:
            return self._repr == other._repr
        except AttributeError:
            return False

    def __copy__(self):
        if False:
            while True:
                i = 10
        return self

    def __deepcopy__(self, memo=None):
        if False:
            while True:
                i = 10
        if (self._register, self._index) == (None, None):
            return self
        bit = type(self).__new__(type(self))
        bit._register = copy.deepcopy(self._register, memo)
        bit._index = self._index
        bit._hash = self._hash
        bit._repr = self._repr
        return bit