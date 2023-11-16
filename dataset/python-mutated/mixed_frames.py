"""
Mixed Frames
"""
from .frames import Frame
from .pulse_target import PulseTarget

class MixedFrame:
    """Representation of a :class:`PulseTarget` and :class:`Frame` combination.

    Most instructions need to be associated with both a :class:`PulseTarget` and a
    :class:`Frame`. The combination
    of the two is called a mixed frame and is represented by a :class:`MixedFrame` object.

    In most cases the :class:`MixedFrame` is used more by the compiler, and a pulse program
    can be written without :class:`MixedFrame` s, by setting :class:`PulseTarget` and
    :class:`Frame` independently. However, in some cases using :class:`MixedFrame` s can
    better convey the meaning of the code, and change the compilation process. One example
    is the use of the shift/set frequency/phase instructions which are not broadcasted to other
    :class:`MixedFrame` s if applied on a specific :class:`MixedFrame` (unlike the behavior
    of :class:`Frame`). User can also use a subclass of :class:`MixedFrame` for a particular
    combination of logical elements and frames as if a syntactic sugar. This might
    increase the readability of a user pulse program. As an example consider the cross
    resonance architecture, in which a pulse is played on a target qubit frame and applied
    to a control qubit logical element.
    """

    def __init__(self, pulse_target: PulseTarget, frame: Frame):
        if False:
            print('Hello World!')
        'Create ``MixedFrame``.\n\n        Args:\n            pulse_target: The ``PulseTarget`` associated with the mixed frame.\n            frame: The frame associated with the mixed frame.\n        '
        self._pulse_target = pulse_target
        self._frame = frame

    @property
    def pulse_target(self) -> PulseTarget:
        if False:
            i = 10
            return i + 15
        'Return the target of this mixed frame.'
        return self._pulse_target

    @property
    def frame(self) -> Frame:
        if False:
            while True:
                i = 10
        'Return the ``Frame`` of this mixed frame.'
        return self._frame

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'MixedFrame({self.pulse_target},{self.frame})'

    def __eq__(self, other: 'MixedFrame') -> bool:
        if False:
            print('Hello World!')
        'Return True iff self and other are equal, specifically, iff they have the same target\n         and frame.\n\n        Args:\n            other: The mixed frame to compare to this one.\n\n        Returns:\n            True iff equal.\n        '
        return self._pulse_target == other._pulse_target and self._frame == other._frame

    def __hash__(self) -> int:
        if False:
            print('Hello World!')
        return hash((self._pulse_target, self._frame, type(self)))