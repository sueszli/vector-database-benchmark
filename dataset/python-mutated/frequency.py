"""Frequency instructions module. These instructions allow the user to manipulate
the frequency of a channel.
"""
from typing import Optional, Union, Tuple
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.instructions.instruction import Instruction
from qiskit.pulse.exceptions import PulseError

class SetFrequency(Instruction):
    """Set the channel frequency. This instruction operates on ``PulseChannel`` s.
    A ``PulseChannel`` creates pulses of the form

    .. math::
        Re[\\exp(i 2\\pi f jdt + \\phi) d_j].

    Here, :math:`f` is the frequency of the channel. The instruction ``SetFrequency`` allows
    the user to set the value of :math:`f`. All pulses that are played on a channel
    after SetFrequency has been called will have the corresponding frequency.

    The duration of SetFrequency is 0.
    """

    def __init__(self, frequency: Union[float, ParameterExpression], channel: PulseChannel, name: Optional[str]=None):
        if False:
            return 10
        'Creates a new set channel frequency instruction.\n\n        Args:\n            frequency: New frequency of the channel in Hz.\n            channel: The channel this instruction operates on.\n            name: Name of this set channel frequency instruction.\n        '
        super().__init__(operands=(frequency, channel), name=name)

    def _validate(self):
        if False:
            return 10
        'Called after initialization to validate instruction data.\n\n        Raises:\n            PulseError: If the input ``channel`` is not type :class:`PulseChannel`.\n        '
        if not isinstance(self.channel, PulseChannel):
            raise PulseError(f'Expected a pulse channel, got {self.channel} instead.')

    @property
    def frequency(self) -> Union[float, ParameterExpression]:
        if False:
            for i in range(10):
                print('nop')
        'New frequency.'
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
            print('Hello World!')
        'Duration of this instruction.'
        return 0

class ShiftFrequency(Instruction):
    """Shift the channel frequency away from the current frequency."""

    def __init__(self, frequency: Union[float, ParameterExpression], channel: PulseChannel, name: Optional[str]=None):
        if False:
            while True:
                i = 10
        'Creates a new shift frequency instruction.\n\n        Args:\n            frequency: Frequency shift of the channel in Hz.\n            channel: The channel this instruction operates on.\n            name: Name of this set channel frequency instruction.\n        '
        super().__init__(operands=(frequency, channel), name=name)

    def _validate(self):
        if False:
            while True:
                i = 10
        'Called after initialization to validate instruction data.\n\n        Raises:\n            PulseError: If the input ``channel`` is not type :class:`PulseChannel`.\n        '
        if not isinstance(self.channel, PulseChannel):
            raise PulseError(f'Expected a pulse channel, got {self.channel} instead.')

    @property
    def frequency(self) -> Union[float, ParameterExpression]:
        if False:
            while True:
                i = 10
        'Frequency shift from the set frequency.'
        return self.operands[0]

    @property
    def channel(self) -> PulseChannel:
        if False:
            for i in range(10):
                print('nop')
        'Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is\n        scheduled on.\n        '
        return self.operands[1]

    @property
    def channels(self) -> Tuple[PulseChannel]:
        if False:
            i = 10
            return i + 15
        'Returns the channels that this schedule uses.'
        return (self.channel,)

    @property
    def duration(self) -> int:
        if False:
            print('Hello World!')
        'Duration of this instruction.'
        return 0