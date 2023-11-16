"""
Classical register reference object.
"""
import itertools
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.deprecation import deprecate_func
from .register import Register
from .bit import Bit

class Clbit(Bit):
    """Implement a classical bit."""
    __slots__ = ()

    def __init__(self, register=None, index=None):
        if False:
            for i in range(10):
                print('nop')
        'Creates a classical bit.\n\n        Args:\n            register (ClassicalRegister): Optional. A classical register containing the bit.\n            index (int): Optional. The index of the bit in its containing register.\n\n        Raises:\n            CircuitError: if the provided register is not a valid :class:`ClassicalRegister`\n        '
        if register is None or isinstance(register, ClassicalRegister):
            super().__init__(register, index)
        else:
            raise CircuitError('Clbit needs a ClassicalRegister and %s was provided' % type(register).__name__)

class ClassicalRegister(Register):
    """Implement a classical register."""
    instances_counter = itertools.count()
    prefix = 'c'
    bit_type = Clbit

    @deprecate_func(additional_msg='Correct exporting to OpenQASM 2 is the responsibility of a larger exporter; it cannot safely be done on an object-by-object basis without context. No replacement will be provided, because the premise is wrong.', since='0.23.0', package_name='qiskit-terra')
    def qasm(self):
        if False:
            while True:
                i = 10
        'Return OPENQASM string for this register.'
        return 'creg %s[%d];' % (self.name, self.size)