"""
Frames
"""
from abc import ABC
import numpy as np
from qiskit.pulse.exceptions import PulseError

class Frame(ABC):
    """Base class for pulse module frame.

    Because pulses used in Quantum hardware are typically AC pulses, the carrier frequency and phase
    must be defined. The :class:`Frame` is the object which identifies the frequency and phase for
    the carrier.
    and each pulse and most other instructions are associated with a frame. The different types of frames
    dictate how the frequency and phase duo are defined.

    The default initial phase for every frame is 0.
    """

class GenericFrame(Frame):
    """Pulse module GenericFrame.

    The :class:`GenericFrame` is used for custom user defined frames, which are not associated with any
    backend defaults. It is especially useful when the frame doesn't correspond to any frame of
    the typical qubit model, like qudit control for example. Because no backend defaults exist for
    these frames, during compilation an initial frequency and phase will need to be provided.

    :class:`GenericFrame` objects are identified by their unique name.
    """

    def __init__(self, name: str):
        if False:
            i = 10
            return i + 15
        'Create ``GenericFrame``.\n\n        Args:\n            name: A unique identifier used to identify the frame.\n        '
        self._name = name

    @property
    def name(self) -> str:
        if False:
            return 10
        'Return the name of the frame.'
        return self._name

    def __repr__(self) -> str:
        if False:
            return 10
        return f'GenericFrame({self._name})'

    def __eq__(self, other):
        if False:
            return 10
        return type(self) is type(other) and self._name == other._name

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash((type(self), self._name))

class QubitFrame(Frame):
    """A frame associated with the driving of a qubit.

    :class:`QubitFrame` is a frame associated with the driving of a specific qubit.
    The initial frequency of
    the frame will be taken as the default driving frequency provided by the backend
    during compilation.
    """

    def __init__(self, index: int):
        if False:
            while True:
                i = 10
        'Create ``QubitFrame``.\n\n        Args:\n            index: The index of the qubit represented by the frame.\n        '
        self._validate_index(index)
        self._index = index

    @property
    def index(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Return the qubit index of the qubit frame.'
        return self._index

    def _validate_index(self, index) -> None:
        if False:
            print('Hello World!')
        'Raise a ``PulseError`` if the qubit index is invalid. Namely, check if the index is a\n        non-negative integer.\n\n        Raises:\n            PulseError: If ``identifier`` (index) is a negative integer.\n        '
        if not isinstance(index, (int, np.integer)) or index < 0:
            raise PulseError('Qubit index must be a non-negative integer')

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'QubitFrame({self._index})'

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return type(self) is type(other) and self._index == other._index

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((type(self), self._index))

class MeasurementFrame(Frame):
    """A frame associated with the measurement of a qubit.

    ``MeasurementFrame`` is a frame associated with the readout of a specific qubit,
    which requires a stimulus tone driven at frequency off resonant to qubit drive.

    If not set otherwise, the initial frequency of the frame will be taken as the default
    measurement frequency provided by the backend during compilation.
    """

    def __init__(self, index: int):
        if False:
            print('Hello World!')
        'Create ``MeasurementFrame``.\n\n        Args:\n            index: The index of the qubit represented by the frame.\n        '
        self._validate_index(index)
        self._index = index

    @property
    def index(self) -> int:
        if False:
            print('Hello World!')
        'Return the qubit index of the measurement frame.'
        return self._index

    def _validate_index(self, index) -> None:
        if False:
            while True:
                i = 10
        'Raise a ``PulseError`` if the qubit index is invalid. Namely, check if the index is a\n        non-negative integer.\n\n        Raises:\n            PulseError: If ``index`` is a negative integer.\n        '
        if not isinstance(index, (int, np.integer)) or index < 0:
            raise PulseError('Qubit index must be a non-negative integer')

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'MeasurementFrame({self._index})'

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return type(self) is type(other) and self._index == other._index

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash((type(self), self._index))