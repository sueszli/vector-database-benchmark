"""Model and schema for pulse defaults."""
from typing import Any, Dict, List
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap, PulseQobjDef
from qiskit.qobj import PulseLibraryItem, PulseQobjInstruction
from qiskit.qobj.converters import QobjToInstructionConverter

class MeasurementKernel:
    """Class representing a Measurement Kernel."""

    def __init__(self, name, params):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a MeasurementKernel object\n\n        Args:\n            name (str): The name of the measurement kernel\n            params: The parameters of the measurement kernel\n        '
        self.name = name
        self.params = params

    def to_dict(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a dictionary format representation of the MeasurementKernel.\n\n        Returns:\n            dict: The dictionary form of the MeasurementKernel.\n        '
        return {'name': self.name, 'params': self.params}

    @classmethod
    def from_dict(cls, data):
        if False:
            return 10
        'Create a new MeasurementKernel object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the MeasurementKernel\n                         to create. It will be in the same format as output by\n                         :meth:`to_dict`.\n\n        Returns:\n            MeasurementKernel: The MeasurementKernel from the input dictionary.\n        '
        return cls(**data)

class Discriminator:
    """Class representing a Discriminator."""

    def __init__(self, name, params):
        if False:
            return 10
        'Initialize a Discriminator object\n\n        Args:\n            name (str): The name of the discriminator\n            params: The parameters of the discriminator\n        '
        self.name = name
        self.params = params

    def to_dict(self):
        if False:
            while True:
                i = 10
        'Return a dictionary format representation of the Discriminator.\n\n        Returns:\n            dict: The dictionary form of the Discriminator.\n        '
        return {'name': self.name, 'params': self.params}

    @classmethod
    def from_dict(cls, data):
        if False:
            return 10
        'Create a new Discriminator object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the Discriminator\n                         to create. It will be in the same format as output by\n                         :meth:`to_dict`.\n\n        Returns:\n            Discriminator: The Discriminator from the input dictionary.\n        '
        return cls(**data)

class Command:
    """Class representing a Command.

    Attributes:
        name: Pulse command name.
    """
    _data = {}

    def __init__(self, name: str, qubits=None, sequence=None, **kwargs):
        if False:
            return 10
        'Initialize a Command object\n\n        Args:\n            name (str): The name of the command\n            qubits: The qubits for the command\n            sequence (PulseQobjInstruction): The sequence for the Command\n            kwargs: Optional additional fields\n        '
        self._data = {}
        self.name = name
        if qubits is not None:
            self.qubits = qubits
        if sequence is not None:
            self.sequence = sequence
        self._data.update(kwargs)

    def __getattr__(self, name):
        if False:
            return 10
        try:
            return self._data[name]
        except KeyError as ex:
            raise AttributeError(f'Attribute {name} is not defined') from ex

    def to_dict(self):
        if False:
            while True:
                i = 10
        'Return a dictionary format representation of the Command.\n\n        Returns:\n            dict: The dictionary form of the Command.\n        '
        out_dict = {'name': self.name}
        if hasattr(self, 'qubits'):
            out_dict['qubits'] = self.qubits
        if hasattr(self, 'sequence'):
            out_dict['sequence'] = [x.to_dict() for x in self.sequence]
        out_dict.update(self._data)
        return out_dict

    @classmethod
    def from_dict(cls, data):
        if False:
            while True:
                i = 10
        'Create a new Command object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the ``Command``\n                         to create. It will be in the same format as output by\n                         :meth:`to_dict`.\n\n        Returns:\n            Command: The ``Command`` from the input dictionary.\n        '
        in_data = {}
        for (key, value) in data.items():
            if key == 'sequence':
                in_data[key] = list(map(PulseQobjInstruction.from_dict, value))
            else:
                in_data[key] = value
        return cls(**in_data)

class PulseDefaults:
    """Description of default settings for Pulse systems. These are instructions or settings that
    may be good starting points for the Pulse user. The user may modify these defaults for custom
    scheduling.
    """
    _data = {}

    def __init__(self, qubit_freq_est: List[float], meas_freq_est: List[float], buffer: int, pulse_library: List[PulseLibraryItem], cmd_def: List[Command], meas_kernel: MeasurementKernel=None, discriminator: Discriminator=None, **kwargs: Dict[str, Any]):
        if False:
            return 10
        '\n        Validate and reformat transport layer inputs to initialize.\n        Args:\n            qubit_freq_est: Estimated qubit frequencies in GHz.\n            meas_freq_est: Estimated measurement cavity frequencies in GHz.\n            buffer: Default buffer time (in units of dt) between pulses.\n            pulse_library: Pulse name and sample definitions.\n            cmd_def: Operation name and definition in terms of Commands.\n            meas_kernel: The measurement kernels\n            discriminator: The discriminators\n            **kwargs: Other attributes for the super class.\n        '
        self._data = {}
        self.buffer = buffer
        self.qubit_freq_est = [freq * 1000000000.0 for freq in qubit_freq_est]
        'Qubit frequencies in Hertz.'
        self.meas_freq_est = [freq * 1000000000.0 for freq in meas_freq_est]
        'Measurement frequencies in Hertz.'
        self.pulse_library = pulse_library
        self.cmd_def = cmd_def
        self.instruction_schedule_map = InstructionScheduleMap()
        self.converter = QobjToInstructionConverter(pulse_library)
        for inst in cmd_def:
            entry = PulseQobjDef(converter=self.converter, name=inst.name)
            entry.define(inst.sequence, user_provided=False)
            self.instruction_schedule_map._add(instruction_name=inst.name, qubits=tuple(inst.qubits), entry=entry)
        if meas_kernel is not None:
            self.meas_kernel = meas_kernel
        if discriminator is not None:
            self.discriminator = discriminator
        self._data.update(kwargs)

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        try:
            return self._data[name]
        except KeyError as ex:
            raise AttributeError(f'Attribute {name} is not defined') from ex

    def to_dict(self):
        if False:
            i = 10
            return i + 15
        'Return a dictionary format representation of the PulseDefaults.\n        Returns:\n            dict: The dictionary form of the PulseDefaults.\n        '
        out_dict = {'qubit_freq_est': self.qubit_freq_est, 'meas_freq_est': self.qubit_freq_est, 'buffer': self.buffer, 'pulse_library': [x.to_dict() for x in self.pulse_library], 'cmd_def': [x.to_dict() for x in self.cmd_def]}
        if hasattr(self, 'meas_kernel'):
            out_dict['meas_kernel'] = self.meas_kernel.to_dict()
        if hasattr(self, 'discriminator'):
            out_dict['discriminator'] = self.discriminator.to_dict()
        for (key, value) in self.__dict__.items():
            if key not in ['qubit_freq_est', 'meas_freq_est', 'buffer', 'pulse_library', 'cmd_def', 'meas_kernel', 'discriminator', 'converter', 'instruction_schedule_map']:
                out_dict[key] = value
        out_dict.update(self._data)
        out_dict['qubit_freq_est'] = [freq * 1e-09 for freq in self.qubit_freq_est]
        out_dict['meas_freq_est'] = [freq * 1e-09 for freq in self.meas_freq_est]
        return out_dict

    @classmethod
    def from_dict(cls, data):
        if False:
            return 10
        'Create a new PulseDefaults object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the PulseDefaults\n                         to create. It will be in the same format as output by\n                         :meth:`to_dict`.\n        Returns:\n            PulseDefaults: The PulseDefaults from the input dictionary.\n        '
        schema = {'pulse_library': PulseLibraryItem, 'cmd_def': Command, 'meas_kernel': MeasurementKernel, 'discriminator': Discriminator}
        in_data = {}
        for (key, value) in data.items():
            if key in schema:
                if isinstance(value, list):
                    in_data[key] = list(map(schema[key].from_dict, value))
                else:
                    in_data[key] = schema[key].from_dict(value)
            else:
                in_data[key] = value
        return cls(**in_data)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        qubit_freqs = [freq / 1000000000.0 for freq in self.qubit_freq_est]
        meas_freqs = [freq / 1000000000.0 for freq in self.meas_freq_est]
        qfreq = f'Qubit Frequencies [GHz]\n{qubit_freqs}'
        mfreq = f'Measurement Frequencies [GHz]\n{meas_freqs} '
        return '<{name}({insts}{qfreq}\n{mfreq})>'.format(name=self.__class__.__name__, insts=str(self.instruction_schedule_map), qfreq=qfreq, mfreq=mfreq)