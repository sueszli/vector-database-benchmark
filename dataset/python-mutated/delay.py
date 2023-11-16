"""An instruction for blocking time on a channel; useful for scheduling alignment."""
from typing import Optional, Union, Tuple
from qiskit.circuit import ParameterExpression
from qiskit.pulse.channels import Channel
from qiskit.pulse.instructions.instruction import Instruction

class Delay(Instruction):
    """A blocking instruction with no other effect. The delay is used for aligning and scheduling
    other instructions.

    Example:

        To schedule an instruction at time = 10, on a channel assigned to the variable ``channel``,
        the following could be used::

            sched = Schedule(name="Delay instruction example")
            sched += Delay(10, channel)
            sched += Gaussian(duration, amp, sigma, channel)

        The ``channel`` will output no signal from time=0 up until time=10.
    """

    def __init__(self, duration: Union[int, ParameterExpression], channel: Channel, name: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        'Create a new delay instruction.\n\n        No other instruction may be scheduled within a ``Delay``.\n\n        Args:\n            duration: Length of time of the delay in terms of dt.\n            channel: The channel that will have the delay.\n            name: Name of the delay for display purposes.\n        '
        super().__init__(operands=(duration, channel), name=name)

    @property
    def channel(self) -> Channel:
        if False:
            print('Hello World!')
        'Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is\n        scheduled on.\n        '
        return self.operands[1]

    @property
    def channels(self) -> Tuple[Channel]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the channels that this schedule uses.'
        return (self.channel,)

    @property
    def duration(self) -> Union[int, ParameterExpression]:
        if False:
            while True:
                i = 10
        'Duration of this instruction.'
        return self.operands[0]