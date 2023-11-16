"""Helper class used to convert a pulse instruction into PulseQobjInstruction."""
import hashlib
import re
import warnings
from enum import Enum
from functools import singledispatchmethod
from typing import Union, List, Iterator, Optional
import numpy as np
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse import channels, instructions, library
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.exceptions import QiskitError
from qiskit.pulse.parser import parse_string_expr
from qiskit.pulse.schedule import Schedule
from qiskit.qobj import QobjMeasurementOption, PulseLibraryItem, PulseQobjInstruction
from qiskit.qobj.utils import MeasLevel
from qiskit.utils.deprecation import deprecate_func

class ParametricPulseShapes(Enum):
    """Map the assembled pulse names to the pulse module waveforms.

    The enum name is the transport layer name for pulse shapes, the
    value is its mapping to the OpenPulse Command in Qiskit.
    """
    gaussian = 'Gaussian'
    gaussian_square = 'GaussianSquare'
    gaussian_square_drag = 'GaussianSquareDrag'
    gaussian_square_echo = 'gaussian_square_echo'
    drag = 'Drag'
    constant = 'Constant'

    @classmethod
    def from_instance(cls, instance: Union[library.ParametricPulse, library.SymbolicPulse]) -> 'ParametricPulseShapes':
        if False:
            for i in range(10):
                print('nop')
        'Get Qobj name from the pulse class instance.\n\n        Args:\n            instance: Symbolic or ParametricPulse class.\n\n        Returns:\n            Qobj name.\n\n        Raises:\n            QiskitError: When pulse instance is not recognizable type.\n        '
        if isinstance(instance, library.SymbolicPulse):
            return cls(instance.pulse_type)
        if isinstance(instance, library.parametric_pulses.Gaussian):
            return ParametricPulseShapes.gaussian
        if isinstance(instance, library.parametric_pulses.GaussianSquare):
            return ParametricPulseShapes.gaussian_square
        if isinstance(instance, library.parametric_pulses.Drag):
            return ParametricPulseShapes.drag
        if isinstance(instance, library.parametric_pulses.Constant):
            return ParametricPulseShapes.constant
        raise QiskitError(f"'{instance}' is not valid pulse type.")

    @classmethod
    def to_type(cls, name: str) -> library.SymbolicPulse:
        if False:
            i = 10
            return i + 15
        'Get symbolic pulse class from the name.\n\n        Args:\n            name: Qobj name of the pulse.\n\n        Returns:\n            Corresponding class.\n        '
        return getattr(library, cls[name].value)

class InstructionToQobjConverter:
    """Converts Qiskit Pulse in-memory representation into Qobj data.

    This converter converts the Qiskit Pulse in-memory representation into
    the transfer layer format to submit the data from client to the server.

    The transfer layer format must be the text representation that coforms to
    the `OpenPulse specification<https://arxiv.org/abs/1809.03452>`__.
    Extention to the OpenPulse can be achieved by subclassing this this with
    extra methods corresponding to each augumented instruction. For example,

    .. code-block:: python

        class MyConverter(InstructionToQobjConverter):

            def _convert_NewInstruction(self, instruction, time_offset):
                command_dict = {
                    'name': 'new_inst',
                    't0': time_offset + instruction.start_time,
                    'param1': instruction.param1,
                    'param2': instruction.param2
                }
                return self._qobj_model(**command_dict)

    where ``NewInstruction`` must be a class name of Qiskit Pulse instruction.
    """

    def __init__(self, qobj_model: PulseQobjInstruction, **run_config):
        if False:
            print('Hello World!')
        'Create new converter.\n\n        Args:\n             qobj_model: Transfer layer data schema.\n             run_config: Run configuration.\n        '
        self._qobj_model = qobj_model
        self._run_config = run_config

    def __call__(self, shift: int, instruction: Union[instructions.Instruction, List[instructions.Acquire]]) -> PulseQobjInstruction:
        if False:
            while True:
                i = 10
        'Convert Qiskit in-memory representation to Qobj instruction.\n\n        Args:\n            instruction: Instruction data in Qiskit Pulse.\n\n        Returns:\n            Qobj instruction data.\n\n        Raises:\n            QiskitError: When list of instruction is provided except for Acquire.\n        '
        if isinstance(instruction, list):
            if all((isinstance(inst, instructions.Acquire) for inst in instruction)):
                return self._convert_bundled_acquire(instruction_bundle=instruction, time_offset=shift)
            raise QiskitError('Bundle of instruction is not supported except for Acquire.')
        return self._convert_instruction(instruction, shift)

    @singledispatchmethod
    def _convert_instruction(self, instruction, time_offset: int) -> PulseQobjInstruction:
        if False:
            return 10
        raise QiskitError(f"Pulse Qobj doesn't support {instruction.__class__.__name__}. This instruction cannot be submitted with Qobj.")

    @_convert_instruction.register(instructions.Acquire)
    def _convert_acquire(self, instruction, time_offset: int) -> PulseQobjInstruction:
        if False:
            print('Hello World!')
        'Return converted `Acquire`.\n\n        Args:\n            instruction: Qiskit Pulse acquire instruction.\n            time_offset: Offset time.\n\n        Returns:\n            Qobj instruction data.\n        '
        meas_level = self._run_config.get('meas_level', 2)
        mem_slot = []
        if instruction.mem_slot:
            mem_slot = [instruction.mem_slot.index]
        command_dict = {'name': 'acquire', 't0': time_offset + instruction.start_time, 'duration': instruction.duration, 'qubits': [instruction.channel.index], 'memory_slot': mem_slot}
        if meas_level == MeasLevel.CLASSIFIED:
            if instruction.discriminator:
                command_dict.update({'discriminators': [QobjMeasurementOption(name=instruction.discriminator.name, params=instruction.discriminator.params)]})
            if instruction.reg_slot:
                command_dict.update({'register_slot': [instruction.reg_slot.index]})
        if meas_level in [MeasLevel.KERNELED, MeasLevel.CLASSIFIED]:
            if instruction.kernel:
                command_dict.update({'kernels': [QobjMeasurementOption(name=instruction.kernel.name, params=instruction.kernel.params)]})
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.SetFrequency)
    def _convert_set_frequency(self, instruction, time_offset: int) -> PulseQobjInstruction:
        if False:
            for i in range(10):
                print('nop')
        'Return converted `SetFrequency`.\n\n        Args:\n            instruction: Qiskit Pulse set frequency instruction.\n            time_offset: Offset time.\n\n        Returns:\n            Qobj instruction data.\n        '
        command_dict = {'name': 'setf', 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name, 'frequency': instruction.frequency / 1000000000.0}
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.ShiftFrequency)
    def _convert_shift_frequency(self, instruction, time_offset: int) -> PulseQobjInstruction:
        if False:
            for i in range(10):
                print('nop')
        'Return converted `ShiftFrequency`.\n\n        Args:\n            instruction: Qiskit Pulse shift frequency instruction.\n            time_offset: Offset time.\n\n        Returns:\n            Qobj instruction data.\n        '
        command_dict = {'name': 'shiftf', 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name, 'frequency': instruction.frequency / 1000000000.0}
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.SetPhase)
    def _convert_set_phase(self, instruction, time_offset: int) -> PulseQobjInstruction:
        if False:
            return 10
        'Return converted `SetPhase`.\n\n        Args:\n            instruction: Qiskit Pulse set phase instruction.\n            time_offset: Offset time.\n\n        Returns:\n            Qobj instruction data.\n        '
        command_dict = {'name': 'setp', 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name, 'phase': instruction.phase}
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.ShiftPhase)
    def _convert_shift_phase(self, instruction, time_offset: int) -> PulseQobjInstruction:
        if False:
            for i in range(10):
                print('nop')
        'Return converted `ShiftPhase`.\n\n        Args:\n            instruction: Qiskit Pulse shift phase instruction.\n            time_offset: Offset time.\n\n        Returns:\n            Qobj instruction data.\n        '
        command_dict = {'name': 'fc', 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name, 'phase': instruction.phase}
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.Delay)
    def _convert_delay(self, instruction, time_offset: int) -> PulseQobjInstruction:
        if False:
            print('Hello World!')
        'Return converted `Delay`.\n\n        Args:\n            instruction: Qiskit Pulse delay instruction.\n            time_offset: Offset time.\n\n        Returns:\n            Qobj instruction data.\n        '
        command_dict = {'name': 'delay', 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name, 'duration': instruction.duration}
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.Play)
    def _convert_play(self, instruction, time_offset: int) -> PulseQobjInstruction:
        if False:
            for i in range(10):
                print('nop')
        'Return converted `Play`.\n\n        Args:\n            instruction: Qiskit Pulse play instruction.\n            time_offset: Offset time.\n\n        Returns:\n            Qobj instruction data.\n        '
        if isinstance(instruction.pulse, (library.ParametricPulse, library.SymbolicPulse)):
            params = dict(instruction.pulse.parameters)
            if 'amp' in params and 'angle' in params:
                params['amp'] = complex(params['amp'] * np.exp(1j * params['angle']))
                del params['angle']
            command_dict = {'name': 'parametric_pulse', 'pulse_shape': ParametricPulseShapes.from_instance(instruction.pulse).name, 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name, 'parameters': params}
        else:
            command_dict = {'name': instruction.name, 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name}
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.Snapshot)
    def _convert_snapshot(self, instruction, time_offset: int) -> PulseQobjInstruction:
        if False:
            i = 10
            return i + 15
        'Return converted `Snapshot`.\n\n        Args:\n            time_offset: Offset time.\n            instruction: Qiskit Pulse snapshot instruction.\n\n        Returns:\n            Qobj instruction data.\n        '
        command_dict = {'name': 'snapshot', 't0': time_offset + instruction.start_time, 'label': instruction.label, 'type': instruction.type}
        return self._qobj_model(**command_dict)

    def _convert_bundled_acquire(self, instruction_bundle: List[instructions.Acquire], time_offset: int) -> PulseQobjInstruction:
        if False:
            while True:
                i = 10
        'Return converted list of parallel `Acquire` instructions.\n\n        Args:\n            instruction_bundle: List of Qiskit Pulse acquire instruction.\n            time_offset: Offset time.\n\n        Returns:\n            Qobj instruction data.\n\n        Raises:\n            QiskitError: When instructions are not aligned.\n            QiskitError: When instructions have different duration.\n            QiskitError: When discriminator or kernel is missing in a part of instructions.\n        '
        meas_level = self._run_config.get('meas_level', 2)
        t0 = instruction_bundle[0].start_time
        duration = instruction_bundle[0].duration
        memory_slots = []
        register_slots = []
        qubits = []
        discriminators = []
        kernels = []
        for instruction in instruction_bundle:
            qubits.append(instruction.channel.index)
            if instruction.start_time != t0:
                raise QiskitError('The supplied acquire instructions have different starting times. Something has gone wrong calling this code. Please report this issue.')
            if instruction.duration != duration:
                raise QiskitError('Acquire instructions beginning at the same time must have same duration.')
            if instruction.mem_slot:
                memory_slots.append(instruction.mem_slot.index)
            if meas_level == MeasLevel.CLASSIFIED:
                if instruction.discriminator:
                    discriminators.append(QobjMeasurementOption(name=instruction.discriminator.name, params=instruction.discriminator.params))
                if instruction.reg_slot:
                    register_slots.append(instruction.reg_slot.index)
            if meas_level in [MeasLevel.KERNELED, MeasLevel.CLASSIFIED]:
                if instruction.kernel:
                    kernels.append(QobjMeasurementOption(name=instruction.kernel.name, params=instruction.kernel.params))
        command_dict = {'name': 'acquire', 't0': time_offset + t0, 'duration': duration, 'qubits': qubits}
        if memory_slots:
            command_dict['memory_slot'] = memory_slots
        if register_slots:
            command_dict['register_slot'] = register_slots
        if discriminators:
            num_discriminators = len(discriminators)
            if num_discriminators == len(qubits) or num_discriminators == 1:
                command_dict['discriminators'] = discriminators
            else:
                raise QiskitError('A discriminator must be supplied for every acquisition or a single discriminator for all acquisitions.')
        if kernels:
            num_kernels = len(kernels)
            if num_kernels == len(qubits) or num_kernels == 1:
                command_dict['kernels'] = kernels
            else:
                raise QiskitError('A kernel must be supplied for every acquisition or a single kernel for all acquisitions.')
        return self._qobj_model(**command_dict)

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_acquire(self, shift, instruction):
        if False:
            while True:
                i = 10
        return self._convert_instruction(instruction, shift)

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_bundled_acquires(self, shift, instructions_):
        if False:
            while True:
                i = 10
        return self._convert_bundled_acquire(instructions_, shift)

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_set_frequency(self, shift, instruction):
        if False:
            i = 10
            return i + 15
        return self._convert_instruction(instruction, shift)

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_shift_frequency(self, shift, instruction):
        if False:
            for i in range(10):
                print('nop')
        return self._convert_instruction(instruction, shift)

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_set_phase(self, shift, instruction):
        if False:
            while True:
                i = 10
        return self._convert_instruction(instruction, shift)

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_shift_phase(self, shift, instruction):
        if False:
            while True:
                i = 10
        return self._convert_instruction(instruction, shift)

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_delay(self, shift, instruction):
        if False:
            while True:
                i = 10
        return self._convert_instruction(instruction, shift)

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_play(self, shift, instruction):
        if False:
            return 10
        return self._convert_instruction(instruction, shift)

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_snapshot(self, shift, instruction):
        if False:
            print('Hello World!')
        return self._convert_instruction(instruction, shift)

class QobjToInstructionConverter:
    """Converts Qobj data into Qiskit Pulse in-memory representation.

    This converter converts data from transfer layer into the in-memory representation of
    the front-end of Qiskit Pulse.

    The transfer layer format must be the text representation that coforms to
    the `OpenPulse specification<https://arxiv.org/abs/1809.03452>`__.
    Extention to the OpenPulse can be achieved by subclassing this this with
    extra methods corresponding to each augumented instruction. For example,

    .. code-block:: python

        class MyConverter(QobjToInstructionConverter):

            def get_supported_instructions(self):
                instructions = super().get_supported_instructions()
                instructions += ["new_inst"]

                return instructions

            def _convert_new_inst(self, instruction):
                return NewInstruction(...)

    where ``NewInstruction`` must be a subclass of :class:`~qiskit.pulse.instructions.Instruction`.
    """
    __chan_regex__ = re.compile('([a-zA-Z]+)(\\d+)')

    def __init__(self, pulse_library: Optional[List[PulseLibraryItem]]=None, **run_config):
        if False:
            for i in range(10):
                print('nop')
        'Create new converter.\n\n        Args:\n            pulse_library: Pulse library in Qobj format.\n             run_config: Run configuration.\n        '
        pulse_library_dict = {}
        for lib_item in pulse_library:
            pulse_library_dict[lib_item.name] = lib_item.samples
        self._pulse_library = pulse_library_dict
        self._run_config = run_config

    def __call__(self, instruction: PulseQobjInstruction) -> Schedule:
        if False:
            return 10
        'Convert Qobj instruction to Qiskit in-memory representation.\n\n        Args:\n            instruction: Instruction data in Qobj format.\n\n        Returns:\n            Scheduled Qiskit Pulse instruction in Schedule format.\n        '
        schedule = Schedule()
        for inst in self._get_sequences(instruction):
            schedule.insert(instruction.t0, inst, inplace=True)
        return schedule

    def _get_sequences(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        if False:
            print('Hello World!')
        'A method to iterate over pulse instructions without creating Schedule.\n\n        .. note::\n\n            This is internal fast-path function, and callers other than this converter class\n            might directly use this method to generate schedule from multiple\n            Qobj instructions. Because __call__ always returns a schedule with the time offset\n            parsed instruction, composing multiple Qobj instructions to create\n            a gate schedule is somewhat inefficient due to composing overhead of schedules.\n            Directly combining instructions with this method is much performant.\n\n        Args:\n            instruction: Instruction data in Qobj format.\n\n        Yields:\n            Qiskit Pulse instructions.\n\n        :meta public:\n        '
        try:
            method = getattr(self, f'_convert_{instruction.name}')
        except AttributeError:
            method = self._convert_generic
        yield from method(instruction)

    def get_supported_instructions(self) -> List[str]:
        if False:
            print('Hello World!')
        'Retrun a list of supported instructions.'
        return ['acquire', 'setp', 'fc', 'setf', 'shiftf', 'delay', 'parametric_pulse', 'snapshot']

    def get_channel(self, channel: str) -> channels.PulseChannel:
        if False:
            return 10
        'Parse and retrieve channel from ch string.\n\n        Args:\n            channel: String identifier of pulse instruction channel.\n\n        Returns:\n            Matched channel object.\n\n        Raises:\n            QiskitError: Is raised if valid channel is not matched\n        '
        match = self.__chan_regex__.match(channel)
        if match:
            (prefix, index) = (match.group(1), int(match.group(2)))
            if prefix == channels.DriveChannel.prefix:
                return channels.DriveChannel(index)
            elif prefix == channels.MeasureChannel.prefix:
                return channels.MeasureChannel(index)
            elif prefix == channels.ControlChannel.prefix:
                return channels.ControlChannel(index)
        raise QiskitError('Channel %s is not valid' % channel)

    @staticmethod
    def disassemble_value(value_expr: Union[float, str]) -> Union[float, ParameterExpression]:
        if False:
            return 10
        'A helper function to format instruction operand.\n\n        If parameter in string representation is specified, this method parses the\n        input string and generates Qiskit ParameterExpression object.\n\n        Args:\n            value_expr: Operand value in Qobj.\n\n        Returns:\n            Parsed operand value. ParameterExpression object is returned if value is not number.\n        '
        if isinstance(value_expr, str):
            str_expr = parse_string_expr(value_expr, partial_binding=False)
            value_expr = str_expr(**{pname: Parameter(pname) for pname in str_expr.params})
        return value_expr

    def _convert_acquire(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        if False:
            for i in range(10):
                print('nop')
        'Return converted `Acquire` instruction.\n\n        Args:\n            instruction: Acquire qobj\n\n        Yields:\n            Qiskit Pulse acquire instructions\n        '
        duration = instruction.duration
        qubits = instruction.qubits
        acquire_channels = [channels.AcquireChannel(qubit) for qubit in qubits]
        mem_slots = [channels.MemorySlot(instruction.memory_slot[i]) for i in range(len(qubits))]
        if hasattr(instruction, 'register_slot'):
            register_slots = [channels.RegisterSlot(instruction.register_slot[i]) for i in range(len(qubits))]
        else:
            register_slots = [None] * len(qubits)
        discriminators = instruction.discriminators if hasattr(instruction, 'discriminators') else None
        if not isinstance(discriminators, list):
            discriminators = [discriminators]
        if any((discriminators[i] != discriminators[0] for i in range(len(discriminators)))):
            warnings.warn('Can currently only support one discriminator per acquire. Defaulting to first discriminator entry.')
        discriminator = discriminators[0]
        if discriminator:
            discriminator = Discriminator(name=discriminators[0].name, **discriminators[0].params)
        kernels = instruction.kernels if hasattr(instruction, 'kernels') else None
        if not isinstance(kernels, list):
            kernels = [kernels]
        if any((kernels[0] != kernels[i] for i in range(len(kernels)))):
            warnings.warn('Can currently only support one kernel per acquire. Defaulting to first kernel entry.')
        kernel = kernels[0]
        if kernel:
            kernel = Kernel(name=kernels[0].name, **kernels[0].params)
        for (acquire_channel, mem_slot, reg_slot) in zip(acquire_channels, mem_slots, register_slots):
            yield instructions.Acquire(duration, acquire_channel, mem_slot=mem_slot, reg_slot=reg_slot, kernel=kernel, discriminator=discriminator)

    def _convert_setp(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        if False:
            for i in range(10):
                print('nop')
        'Return converted `SetPhase` instruction.\n\n        Args:\n            instruction: SetPhase qobj instruction\n\n        Yields:\n            Qiskit Pulse set phase instructions\n        '
        channel = self.get_channel(instruction.ch)
        phase = self.disassemble_value(instruction.phase)
        yield instructions.SetPhase(phase, channel)

    def _convert_fc(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        if False:
            return 10
        'Return converted `ShiftPhase` instruction.\n\n        Args:\n            instruction: ShiftPhase qobj instruction\n\n        Yields:\n            Qiskit Pulse shift phase schedule instructions\n        '
        channel = self.get_channel(instruction.ch)
        phase = self.disassemble_value(instruction.phase)
        yield instructions.ShiftPhase(phase, channel)

    def _convert_setf(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        if False:
            i = 10
            return i + 15
        'Return converted `SetFrequencyInstruction` instruction.\n\n        .. note::\n\n            We assume frequency value is expressed in string with "GHz".\n            Operand value is thus scaled by a factor of 1e9.\n\n        Args:\n            instruction: SetFrequency qobj instruction\n\n        Yields:\n            Qiskit Pulse set frequency instructions\n        '
        channel = self.get_channel(instruction.ch)
        frequency = self.disassemble_value(instruction.frequency) * 1000000000.0
        yield instructions.SetFrequency(frequency, channel)

    def _convert_shiftf(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        if False:
            for i in range(10):
                print('nop')
        'Return converted `ShiftFrequency` instruction.\n\n        .. note::\n\n            We assume frequency value is expressed in string with "GHz".\n            Operand value is thus scaled by a factor of 1e9.\n\n        Args:\n            instruction: ShiftFrequency qobj instruction\n\n        Yields:\n            Qiskit Pulse shift frequency schedule instructions\n        '
        channel = self.get_channel(instruction.ch)
        frequency = self.disassemble_value(instruction.frequency) * 1000000000.0
        yield instructions.ShiftFrequency(frequency, channel)

    def _convert_delay(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        if False:
            return 10
        'Return converted `Delay` instruction.\n\n        Args:\n            instruction: Delay qobj instruction\n\n        Yields:\n            Qiskit Pulse delay instructions\n        '
        channel = self.get_channel(instruction.ch)
        duration = instruction.duration
        yield instructions.Delay(duration, channel)

    def _convert_parametric_pulse(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        if False:
            while True:
                i = 10
        'Return converted `Play` instruction with parametric pulse operand.\n\n        .. note::\n\n            If parametric pulse label is not provided by the backend, this method naively generates\n            a pulse name based on the pulse shape and bound parameters. This pulse name is formatted\n            to, for example, `gaussian_a4e3`, here the last four digits are a part of\n            the hash string generated based on the pulse shape and the parameters.\n            Because we are using a truncated hash for readability,\n            there may be a small risk of pulse name collision with other pulses.\n            Basically the parametric pulse name is used just for visualization purpose and\n            the pulse module should not have dependency on the parametric pulse names.\n\n        Args:\n            instruction: Play qobj instruction with parametric pulse\n\n        Yields:\n            Qiskit Pulse play schedule instructions\n        '
        channel = self.get_channel(instruction.ch)
        try:
            pulse_name = instruction.label
        except AttributeError:
            sorted_params = sorted(instruction.parameters.items(), key=lambda x: x[0])
            base_str = '{pulse}_{params}'.format(pulse=instruction.pulse_shape, params=str(sorted_params))
            short_pulse_id = hashlib.md5(base_str.encode('utf-8')).hexdigest()[:4]
            pulse_name = f'{instruction.pulse_shape}_{short_pulse_id}'
        params = dict(instruction.parameters)
        if 'amp' in params and isinstance(params['amp'], complex):
            params['angle'] = np.angle(params['amp'])
            params['amp'] = np.abs(params['amp'])
        pulse = ParametricPulseShapes.to_type(instruction.pulse_shape)(**params, name=pulse_name)
        yield instructions.Play(pulse, channel)

    def _convert_snapshot(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        if False:
            i = 10
            return i + 15
        'Return converted `Snapshot` instruction.\n\n        Args:\n            instruction: Snapshot qobj instruction\n\n        Yields:\n            Qiskit Pulse snapshot instructions\n        '
        yield instructions.Snapshot(instruction.label, instruction.type)

    def _convert_generic(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
        if False:
            for i in range(10):
                print('nop')
        'Convert generic pulse instruction.\n\n        Args:\n            instruction: Generic qobj instruction\n\n        Yields:\n            Qiskit Pulse generic instructions\n\n        Raises:\n            QiskitError: When instruction name not found.\n        '
        if instruction.name in self._pulse_library:
            waveform = library.Waveform(samples=self._pulse_library[instruction.name], name=instruction.name)
            channel = self.get_channel(instruction.ch)
            yield instructions.Play(waveform, channel)
        else:
            raise QiskitError(f'Instruction {instruction.name} on qubit {instruction.qubits} is not found  in Qiskit namespace. This instruction cannot be deserialized.')

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_acquire(self, instruction):
        if False:
            return 10
        t0 = instruction.t0
        schedule = Schedule()
        for inst in self._convert_acquire(instruction=instruction):
            schedule.insert(t0, inst, inplace=True)
        return schedule

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_set_phase(self, instruction):
        if False:
            i = 10
            return i + 15
        t0 = instruction.t0
        schedule = Schedule()
        for inst in self._convert_setp(instruction=instruction):
            schedule.insert(t0, inst, inplace=True)
        return schedule

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_shift_phase(self, instruction):
        if False:
            i = 10
            return i + 15
        t0 = instruction.t0
        schedule = Schedule()
        for inst in self._convert_fc(instruction=instruction):
            schedule.insert(t0, inst, inplace=True)
        return schedule

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_set_frequency(self, instruction):
        if False:
            while True:
                i = 10
        t0 = instruction.t0
        schedule = Schedule()
        for inst in self._convert_setf(instruction=instruction):
            schedule.insert(t0, inst, inplace=True)
        return schedule

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_shift_frequency(self, instruction):
        if False:
            return 10
        t0 = instruction.t0
        schedule = Schedule()
        for inst in self._convert_shiftf(instruction=instruction):
            schedule.insert(t0, inst, inplace=True)
        return schedule

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_delay(self, instruction):
        if False:
            i = 10
            return i + 15
        t0 = instruction.t0
        schedule = Schedule()
        for inst in self._convert_delay(instruction=instruction):
            schedule.insert(t0, inst, inplace=True)
        return schedule

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def bind_pulse(self, pulse):
        if False:
            i = 10
            return i + 15
        if pulse.name not in self._pulse_library:
            self._pulse_library[pulse.name] = pulse.samples

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_parametric(self, instruction):
        if False:
            for i in range(10):
                print('nop')
        t0 = instruction.t0
        schedule = Schedule()
        for inst in self._convert_parametric_pulse(instruction=instruction):
            schedule.insert(t0, inst, inplace=True)
        return schedule

    @deprecate_func(additional_msg='Instead, call converter instance directory.', since='0.23.0', package_name='qiskit-terra')
    def convert_snapshot(self, instruction):
        if False:
            for i in range(10):
                print('nop')
        t0 = instruction.t0
        schedule = Schedule()
        for inst in self._convert_snapshot(instruction=instruction):
            schedule.insert(t0, inst, inplace=True)
        return schedule