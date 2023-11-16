"""
PulseTarget
"""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from qiskit.pulse.exceptions import PulseError

class PulseTarget(ABC):
    """Base class of pulse target.

    A :class:`PulseTarget` object identifies a hardware component the user can control, the typical
    example being playing pulses on. Other examples include measurement related instruments.

    When playing a pulse on a quantum hardware, one typically has to define on what hardware component
    the pulse will be played, and the frame (frequency and phase) of the carrier wave.
    :class:`PulseTarget` addresses only the first of the two, and identifies the component which is the
    target of the pulse. Every played pulse and most other instructions are associated with a
    :class:`PulseTarget` on which they are performed.

    A subclass of :class:`PulseTarget` has to be hashable.
    """

    @abstractmethod
    def __hash__(self) -> int:
        if False:
            print('Hello World!')
        pass

class Port(PulseTarget):
    """A ``Port`` type ``PulseTarget``.

    A :class:`Port` is the most basic ``PulseTarget`` - simply a hardware port the user can control,
    (typically for playing pulses, but not only, for example data acquisition).

    A :class:`Port` is identified by a string, which is set, and must be recognized, by the
    backend. Therefore, using pulse level control with :class:`Port` requires an extensive
    knowledge of the hardware. Programs with string identifiers which are not recognized by the
    backend will fail to execute.
    """

    def __init__(self, name: str):
        if False:
            return 10
        'Create ``Port``.\n\n        Args:\n            name: A string identifying the port.\n        '
        self._name = name

    @property
    def name(self) -> str:
        if False:
            return 10
        'Return the ``name`` of this port.'
        return self._name

    def __eq__(self, other: 'Port') -> bool:
        if False:
            return 10
        'Return True iff self and other are equal, specifically, iff they have the same type\n        and the same ``name``.\n\n        Args:\n            other: The Port to compare to this one.\n\n        Returns:\n            True iff equal.\n        '
        return type(self) is type(other) and self._name == other._name

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return hash((self._name, type(self)))

    def __repr__(self) -> str:
        if False:
            return 10
        return f'Port({self._name})'

class LogicalElement(PulseTarget, ABC):
    """Base class of logical elements.

    Class :class:`LogicalElement` provides an abstraction layer to ``PulseTarget``. The abstraction
    allows to write pulse level programs with less knowledge of the hardware, and in a level which
    is more similar to the circuit level programing. i.e., instead of specifying specific ports, one
    can use Qubits, Couplers, etc.

    A logical element is identified by its type and index.
    """

    def __init__(self, index: Tuple[int, ...]):
        if False:
            print('Hello World!')
        'Create ``LogicalElement``.\n\n        Args:\n            index: Tuple of indices of the logical element.\n        '
        self._validate_index(index)
        self._index = index

    @property
    def index(self) -> Tuple[int, ...]:
        if False:
            while True:
                i = 10
        'Return the ``index`` of this logical element.'
        return self._index

    @abstractmethod
    def _validate_index(self, index) -> None:
        if False:
            return 10
        'Raise a PulseError if the logical element ``index`` is invalid.\n\n        Raises:\n            PulseError: If ``index`` is not valid.\n        '
        pass

    def __eq__(self, other: 'LogicalElement') -> bool:
        if False:
            i = 10
            return i + 15
        'Return True iff self and other are equal, specifically, iff they have the same type\n        and the same ``index``.\n\n        Args:\n            other: The logical element to compare to this one.\n\n        Returns:\n            True iff equal.\n        '
        return type(self) is type(other) and self._index == other._index

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return hash((self._index, type(self)))

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        ind_str = str(self._index) if len(self._index) > 1 else f'({self._index[0]})'
        return type(self).__name__ + ind_str

class Qubit(LogicalElement):
    """Qubit logical element.

    ``Qubit`` represents the different qubits in the system, as identified by
    their (positive integer) index values.
    """

    def __init__(self, index: int):
        if False:
            print('Hello World!')
        'Qubit logical element.\n\n        Args:\n            index: Qubit index (positive integer).\n        '
        super().__init__((index,))

    @property
    def qubit_index(self):
        if False:
            while True:
                i = 10
        'Index of the Qubit'
        return self.index[0]

    def _validate_index(self, index) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Raise a ``PulseError`` if the qubit index is invalid. Namely, check if the index is a\n        non-negative integer.\n\n        Raises:\n            PulseError: If ``index`` is a negative integer.\n        '
        if not isinstance(index[0], (int, np.integer)) or index[0] < 0:
            raise PulseError('Qubit index must be a non-negative integer')

class Coupler(LogicalElement):
    """Coupler logical element.

    :class:`Coupler` represents an element which couples qubits, and can be controlled on its own.
    It is identified by the tuple of indices of the coupled qubits.
    """

    def __init__(self, *qubits):
        if False:
            return 10
        'Coupler logical element.\n\n        The coupler ``index`` is defined as the ``tuple`` (\\*qubits).\n\n        Args:\n            *qubits: any number of qubit indices coupled by the coupler.\n        '
        super().__init__(tuple(qubits))

    def _validate_index(self, index) -> None:
        if False:
            return 10
        "Raise a ``PulseError`` if the coupler ``index`` is invalid. Namely,\n        check if coupled qubit indices are non-negative integers, at least two indices were provided,\n        and that the indices don't repeat.\n\n        Raises:\n            PulseError: If ``index`` is invalid.\n        "
        if len(index) < 2:
            raise PulseError('At least two qubit indices are needed for a Coupler')
        for qubit_index in index:
            if not isinstance(qubit_index, (int, np.integer)) or qubit_index < 0:
                raise PulseError('Both indices of coupled qubits must be non-negative integers')
        if len(set(index)) != len(index):
            raise PulseError('Indices of a coupler can not repeat')