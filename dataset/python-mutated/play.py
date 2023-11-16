"""An instruction to transmit a given pulse on a ``PulseChannel`` (i.e., those which support
transmitted pulses, such as ``DriveChannel``).
"""
from typing import Optional, Union, Tuple, Set
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.instructions.instruction import Instruction
from qiskit.pulse.library.pulse import Pulse

class Play(Instruction):
    """This instruction is responsible for applying a pulse on a channel.

    The pulse specifies the exact time dynamics of the output signal envelope for a limited
    time. The output is modulated by a phase and frequency which are controlled by separate
    instructions. The pulse duration must be fixed, and is implicitly given in terms of the
    cycle time, dt, of the backend.
    """

    def __init__(self, pulse: Pulse, channel: PulseChannel, name: Optional[str]=None):
        if False:
            return 10
        'Create a new pulse instruction.\n\n        Args:\n            pulse: A pulse waveform description, such as\n                   :py:class:`~qiskit.pulse.library.Waveform`.\n            channel: The channel to which the pulse is applied.\n            name: Name of the instruction for display purposes. Defaults to ``pulse.name``.\n        '
        if name is None:
            name = pulse.name
        super().__init__(operands=(pulse, channel), name=name)

    def _validate(self):
        if False:
            i = 10
            return i + 15
        'Called after initialization to validate instruction data.\n\n        Raises:\n            PulseError: If pulse is not a Pulse type.\n            PulseError: If the input ``channel`` is not type :class:`PulseChannel`.\n        '
        if not isinstance(self.pulse, Pulse):
            raise PulseError('The `pulse` argument to `Play` must be of type `library.Pulse`.')
        if not isinstance(self.channel, PulseChannel):
            raise PulseError(f'Expected a pulse channel, got {self.channel} instead.')

    @property
    def pulse(self) -> Pulse:
        if False:
            i = 10
            return i + 15
        'A description of the samples that will be played.'
        return self.operands[0]

    @property
    def channel(self) -> PulseChannel:
        if False:
            i = 10
            return i + 15
        'Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is\n        scheduled on.\n        '
        return self.operands[1]

    @property
    def channels(self) -> Tuple[PulseChannel]:
        if False:
            return 10
        'Returns the channels that this schedule uses.'
        return (self.channel,)

    @property
    def duration(self) -> Union[int, ParameterExpression]:
        if False:
            while True:
                i = 10
        'Duration of this instruction.'
        return self.pulse.duration

    @property
    def parameters(self) -> Set:
        if False:
            return 10
        'Parameters which determine the instruction behavior.'
        parameters = set()
        for pulse_param_expr in self.pulse.parameters.values():
            if isinstance(pulse_param_expr, ParameterExpression):
                parameters = parameters | pulse_param_expr.parameters
        if self.channel.is_parameterized():
            parameters = parameters | self.channel.parameters
        return parameters