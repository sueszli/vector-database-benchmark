"""
Quantum register reference object.
"""
import itertools
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.deprecation import deprecate_func
from .register import Register
from .bit import Bit

class Qubit(Bit):
    """Implement a quantum bit."""
    __slots__ = ()

    def __init__(self, register=None, index=None):
        if False:
            i = 10
            return i + 15
        'Creates a qubit.\n\n        Args:\n            register (QuantumRegister): Optional. A quantum register containing the bit.\n            index (int): Optional. The index of the bit in its containing register.\n\n        Raises:\n            CircuitError: if the provided register is not a valid :class:`QuantumRegister`\n        '
        if register is None or isinstance(register, QuantumRegister):
            super().__init__(register, index)
        else:
            raise CircuitError('Qubit needs a QuantumRegister and %s was provided' % type(register).__name__)

class QuantumRegister(Register):
    """Implement a quantum register."""
    instances_counter = itertools.count()
    prefix = 'q'
    bit_type = Qubit

    @deprecate_func(additional_msg='Correct exporting to OpenQASM 2 is the responsibility of a larger exporter; it cannot safely be done on an object-by-object basis without context. No replacement will be provided, because the premise is wrong.', since='0.23.0', package_name='qiskit-terra')
    def qasm(self):
        if False:
            print('Hello World!')
        'Return OPENQASM string for this register.'
        return 'qreg %s[%d];' % (self.name, self.size)

class AncillaQubit(Qubit):
    """A qubit used as ancillary qubit."""
    __slots__ = ()
    pass

class AncillaRegister(QuantumRegister):
    """Implement an ancilla register."""
    instances_counter = itertools.count()
    prefix = 'a'
    bit_type = AncillaQubit