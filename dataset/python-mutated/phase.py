"""The phase instructions update the modulation phase of pulses played on a channel.
This includes ``SetPhase`` instructions which lock the modulation to a particular phase
at that moment, and ``ShiftPhase`` instructions which increase the existing phase by a
relative amount.
"""
from typing import Optional, Union, Tuple
from qiskit.circuit import ParameterExpression
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.instructions.instruction import Instruction
from qiskit.pulse.exceptions import PulseError

class ShiftPhase(Instruction):
    """The shift phase instruction updates the modulation phase of proceeding pulses played on the
    same :py:class:`~qiskit.pulse.channels.Channel`. It is a relative increase in phase determined
    by the ``phase`` operand.

    In particular, a PulseChannel creates pulses of the form

    .. math::
        Re[\\exp(i 2\\pi f jdt + \\phi) d_j].

    The ``ShiftPhase`` instruction causes :math:`\\phi` to be increased by the instruction's
    ``phase`` operand. This will affect all pulses following on the same channel.

    The qubit phase is tracked in software, enabling instantaneous, nearly error-free Z-rotations
    by using a ShiftPhase to update the frame tracking the qubit state.
    """

    def __init__(self, phase: Union[complex, ParameterExpression], channel: PulseChannel, name: Optional[str]=None):
        if False:
            print('Hello World!')
        'Instantiate a shift phase instruction, increasing the output signal phase on ``channel``\n        by ``phase`` [radians].\n\n        Args:\n            phase: The rotation angle in radians.\n            channel: The channel this instruction operates on.\n            name: Display name for this instruction.\n        '
        super().__init__(operands=(phase, channel), name=name)

    def _validate(self):
        if False:
            return 10
        'Called after initialization to validate instruction data.\n\n        Raises:\n            PulseError: If the input ``channel`` is not type :class:`PulseChannel`.\n        '
        if not isinstance(self.channel, PulseChannel):
            raise PulseError(f'Expected a pulse channel, got {self.channel} instead.')

    @property
    def phase(self) -> Union[complex, ParameterExpression]:
        if False:
            i = 10
            return i + 15
        'Return the rotation angle enacted by this instruction in radians.'
        return self.operands[0]

    @property
    def channel(self) -> PulseChannel:
        if False:
            while True:
                i = 10
        'Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is\n        scheduled on.\n        '
        return self.operands[1]

    @property
    def channels(self) -> Tuple[PulseChannel]:
        if False:
            print('Hello World!')
        'Returns the channels that this schedule uses.'
        return (self.channel,)

    @property
    def duration(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Duration of this instruction.'
        return 0

class SetPhase(Instruction):
    """The set phase instruction sets the phase of the proceeding pulses on that channel
    to ``phase`` radians.

    In particular, a PulseChannel creates pulses of the form

    .. math::

        Re[\\exp(i 2\\pi f jdt + \\phi) d_j]

    The ``SetPhase`` instruction sets :math:`\\phi` to the instruction's ``phase`` operand.
    """

    def __init__(self, phase: Union[complex, ParameterExpression], channel: PulseChannel, name: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        'Instantiate a set phase instruction, setting the output signal phase on ``channel``\n        to ``phase`` [radians].\n\n        Args:\n            phase: The rotation angle in radians.\n            channel: The channel this instruction operates on.\n            name: Display name for this instruction.\n        '
        super().__init__(operands=(phase, channel), name=name)

    def _validate(self):
        if False:
            i = 10
            return i + 15
        'Called after initialization to validate instruction data.\n\n        Raises:\n            PulseError: If the input ``channel`` is not type :class:`PulseChannel`.\n        '
        if not isinstance(self.channel, PulseChannel):
            raise PulseError(f'Expected a pulse channel, got {self.channel} instead.')

    @property
    def phase(self) -> Union[complex, ParameterExpression]:
        if False:
            return 10
        'Return the rotation angle enacted by this instruction in radians.'
        return self.operands[0]

    @property
    def channel(self) -> PulseChannel:
        if False:
            while True:
                i = 10
        'Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is\n        scheduled on.\n        '
        return self.operands[1]

    @property
    def channels(self) -> Tuple[PulseChannel]:
        if False:
            while True:
                i = 10
        'Returns the channels that this schedule uses.'
        return (self.channel,)

    @property
    def duration(self) -> int:
        if False:
            print('Hello World!')
        'Duration of this instruction.'
        return 0