"""Module providing definitions of Pulse Qobj classes."""
import copy
import pprint
from typing import Union, List
import numpy
from qiskit.qobj.common import QobjDictField
from qiskit.qobj.common import QobjHeader
from qiskit.qobj.common import QobjExperimentHeader

class QobjMeasurementOption:
    """An individual measurement option."""

    def __init__(self, name, params=None):
        if False:
            i = 10
            return i + 15
        'Instantiate a new QobjMeasurementOption object.\n\n        Args:\n            name (str): The name of the measurement option\n            params (list): The parameters of the measurement option.\n        '
        self.name = name
        if params is not None:
            self.params = params

    def to_dict(self):
        if False:
            i = 10
            return i + 15
        'Return a dict format representation of the QobjMeasurementOption.\n\n        Returns:\n            dict: The dictionary form of the QasmMeasurementOption.\n        '
        out_dict = {'name': self.name}
        if hasattr(self, 'params'):
            out_dict['params'] = self.params
        return out_dict

    @classmethod
    def from_dict(cls, data):
        if False:
            return 10
        'Create a new QobjMeasurementOption object from a dictionary.\n\n        Args:\n            data (dict): A dictionary for the experiment config\n\n        Returns:\n            QobjMeasurementOption: The object from the input dictionary.\n        '
        name = data.pop('name')
        return cls(name, **data)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, QobjMeasurementOption):
            if self.to_dict() == other.to_dict():
                return True
        return False

class PulseQobjInstruction:
    """A class representing a single instruction in an PulseQobj Experiment."""
    _COMMON_ATTRS = ['ch', 'conditional', 'val', 'phase', 'frequency', 'duration', 'qubits', 'memory_slot', 'register_slot', 'label', 'type', 'pulse_shape', 'parameters']

    def __init__(self, name, t0, ch=None, conditional=None, val=None, phase=None, duration=None, qubits=None, memory_slot=None, register_slot=None, kernels=None, discriminators=None, label=None, type=None, pulse_shape=None, parameters=None, frequency=None):
        if False:
            i = 10
            return i + 15
        'Instantiate a new PulseQobjInstruction object.\n\n        Args:\n            name (str): The name of the instruction\n            t0 (int): Pulse start time in integer **dt** units.\n            ch (str): The channel to apply the pulse instruction.\n            conditional (int): The register to use for a conditional for this\n                instruction\n            val (complex): Complex value to apply, bounded by an absolute value\n                of 1.\n            phase (float): if a ``fc`` instruction, the frame change phase in\n                radians.\n            frequency (float): if a ``sf`` instruction, the frequency in Hz.\n            duration (int): The duration of the pulse in **dt** units.\n            qubits (list): A list of ``int`` representing the qubits the\n                instruction operates on\n            memory_slot (list): If a ``measure`` instruction this is a list\n                of ``int`` containing the list of memory slots to store the\n                measurement results in (must be the same length as qubits).\n                If a ``bfunc`` instruction this is a single ``int`` of the\n                memory slot to store the boolean function result in.\n            register_slot (list): If a ``measure`` instruction this is a list\n                of ``int`` containing the list of register slots in which to\n                store the measurement results (must be the same length as\n                qubits). If a ``bfunc`` instruction this is a single ``int``\n                of the register slot in which to store the result.\n            kernels (list): List of :class:`QobjMeasurementOption` objects\n                defining the measurement kernels and set of parameters if the\n                measurement level is 1 or 2. Only used for ``acquire``\n                instructions.\n            discriminators (list): A list of :class:`QobjMeasurementOption`\n                used to set the discriminators to be used if the measurement\n                level is 2. Only used for ``acquire`` instructions.\n            label (str): Label of instruction\n            type (str): Type of instruction\n            pulse_shape (str): The shape of the parametric pulse\n            parameters (dict): The parameters for a parametric pulse\n        '
        self.name = name
        self.t0 = t0
        if ch is not None:
            self.ch = ch
        if conditional is not None:
            self.conditional = conditional
        if val is not None:
            self.val = val
        if phase is not None:
            self.phase = phase
        if frequency is not None:
            self.frequency = frequency
        if duration is not None:
            self.duration = duration
        if qubits is not None:
            self.qubits = qubits
        if memory_slot is not None:
            self.memory_slot = memory_slot
        if register_slot is not None:
            self.register_slot = register_slot
        if kernels is not None:
            self.kernels = kernels
        if discriminators is not None:
            self.discriminators = discriminators
        if label is not None:
            self.label = label
        if type is not None:
            self.type = type
        if pulse_shape is not None:
            self.pulse_shape = pulse_shape
        if parameters is not None:
            self.parameters = parameters

    def to_dict(self):
        if False:
            return 10
        'Return a dictionary format representation of the Instruction.\n\n        Returns:\n            dict: The dictionary form of the PulseQobjInstruction.\n        '
        out_dict = {'name': self.name, 't0': self.t0}
        for attr in self._COMMON_ATTRS:
            if hasattr(self, attr):
                out_dict[attr] = getattr(self, attr)
        if hasattr(self, 'kernels'):
            out_dict['kernels'] = [x.to_dict() for x in self.kernels]
        if hasattr(self, 'discriminators'):
            out_dict['discriminators'] = [x.to_dict() for x in self.discriminators]
        return out_dict

    def __repr__(self):
        if False:
            return 10
        out = f'PulseQobjInstruction(name="{self.name}", t0={self.t0}'
        for attr in self._COMMON_ATTRS:
            attr_val = getattr(self, attr, None)
            if attr_val is not None:
                if isinstance(attr_val, str):
                    out += f', {attr}="{attr_val}"'
                else:
                    out += f', {attr}={attr_val}'
        out += ')'
        return out

    def __str__(self):
        if False:
            while True:
                i = 10
        out = 'Instruction: %s\n' % self.name
        out += '\t\tt0: %s\n' % self.t0
        for attr in self._COMMON_ATTRS:
            if hasattr(self, attr):
                out += f'\t\t{attr}: {getattr(self, attr)}\n'
        return out

    @classmethod
    def from_dict(cls, data):
        if False:
            return 10
        'Create a new PulseQobjExperimentConfig object from a dictionary.\n\n        Args:\n            data (dict): A dictionary for the experiment config\n\n        Returns:\n            PulseQobjInstruction: The object from the input dictionary.\n        '
        schema = {'discriminators': QobjMeasurementOption, 'kernels': QobjMeasurementOption}
        skip = ['t0', 'name']
        in_data = {}
        for (key, value) in data.items():
            if key in skip:
                continue
            if key == 'parameters':
                formatted_value = value.copy()
                if 'amp' in formatted_value:
                    formatted_value['amp'] = _to_complex(formatted_value['amp'])
                in_data[key] = formatted_value
                continue
            if key in schema:
                if isinstance(value, list):
                    in_data[key] = list(map(schema[key].from_dict, value))
                else:
                    in_data[key] = schema[key].from_dict(value)
            else:
                in_data[key] = value
        return cls(data['name'], data['t0'], **in_data)

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, PulseQobjInstruction):
            if self.to_dict() == other.to_dict():
                return True
        return False

def _to_complex(value: Union[List[float], complex]) -> complex:
    if False:
        for i in range(10):
            print('nop')
    'Convert the input value to type ``complex``.\n    Args:\n        value: Value to be converted.\n    Returns:\n        Input value in ``complex``.\n    Raises:\n        TypeError: If the input value is not in the expected format.\n    '
    if isinstance(value, list) and len(value) == 2:
        return complex(value[0], value[1])
    elif isinstance(value, complex):
        return value
    raise TypeError(f'{value} is not in a valid complex number format.')

class PulseQobjConfig(QobjDictField):
    """A configuration for a Pulse Qobj."""

    def __init__(self, meas_level, meas_return, pulse_library, qubit_lo_freq, meas_lo_freq, memory_slot_size=None, rep_time=None, rep_delay=None, shots=None, seed_simulator=None, memory_slots=None, **kwargs):
        if False:
            print('Hello World!')
        "Instantiate a PulseQobjConfig object.\n\n        Args:\n            meas_level (int): The measurement level to use.\n            meas_return (int): The level of measurement information to return.\n            pulse_library (list): A list of :class:`PulseLibraryItem` objects\n                which define the set of primitive pulses\n            qubit_lo_freq (list): List of frequencies (as floats) for the qubit\n                driver LO's in GHz.\n            meas_lo_freq (list): List of frequencies (as floats) for the'\n                measurement driver LO's in GHz.\n            memory_slot_size (int): Size of each memory slot if the output is\n                Level 0.\n            rep_time (int): Time per program execution in sec. Must be from the list provided\n                by the backend (``backend.configuration().rep_times``). Defaults to the first entry\n                in ``backend.configuration().rep_times``.\n            rep_delay (float): Delay between programs in sec. Only supported on certain\n                backends (``backend.configuration().dynamic_reprate_enabled`` ). If supported,\n                ``rep_delay`` will be used instead of ``rep_time`` and must be from the range\n                supplied by the backend (``backend.configuration().rep_delay_range``). Default is\n                ``backend.configuration().default_rep_delay``.\n            shots (int): The number of shots\n            seed_simulator (int): the seed to use in the simulator\n            memory_slots (list): The number of memory slots on the device\n            kwargs: Additional free form key value fields to add to the\n                configuration\n        "
        self.meas_level = meas_level
        self.meas_return = meas_return
        self.pulse_library = pulse_library
        self.qubit_lo_freq = qubit_lo_freq
        self.meas_lo_freq = meas_lo_freq
        if memory_slot_size is not None:
            self.memory_slot_size = memory_slot_size
        if rep_time is not None:
            self.rep_time = rep_time
        if rep_delay is not None:
            self.rep_delay = rep_delay
        if shots is not None:
            self.shots = int(shots)
        if seed_simulator is not None:
            self.seed_simulator = int(seed_simulator)
        if memory_slots is not None:
            self.memory_slots = int(memory_slots)
        if kwargs:
            self.__dict__.update(kwargs)

    def to_dict(self):
        if False:
            i = 10
            return i + 15
        'Return a dictionary format representation of the Pulse Qobj config.\n\n        Returns:\n            dict: The dictionary form of the PulseQobjConfig.\n        '
        out_dict = copy.copy(self.__dict__)
        if hasattr(self, 'pulse_library'):
            out_dict['pulse_library'] = [x.to_dict() for x in self.pulse_library]
        return out_dict

    @classmethod
    def from_dict(cls, data):
        if False:
            i = 10
            return i + 15
        'Create a new PulseQobjConfig object from a dictionary.\n\n        Args:\n            data (dict): A dictionary for the config\n\n        Returns:\n            PulseQobjConfig: The object from the input dictionary.\n        '
        if 'pulse_library' in data:
            pulse_lib = data.pop('pulse_library')
            pulse_lib_obj = [PulseLibraryItem.from_dict(x) for x in pulse_lib]
            data['pulse_library'] = pulse_lib_obj
        return cls(**data)

class PulseQobjExperiment:
    """A Pulse Qobj Experiment.

    Each instance of this class is used to represent an individual Pulse
    experiment as part of a larger Pulse Qobj.
    """

    def __init__(self, instructions, config=None, header=None):
        if False:
            while True:
                i = 10
        'Instantiate a PulseQobjExperiment.\n\n        Args:\n            config (PulseQobjExperimentConfig): A config object for the experiment\n            header (PulseQobjExperimentHeader): A header object for the experiment\n            instructions (list): A list of :class:`PulseQobjInstruction` objects\n        '
        if config is not None:
            self.config = config
        if header is not None:
            self.header = header
        self.instructions = instructions

    def to_dict(self):
        if False:
            i = 10
            return i + 15
        'Return a dictionary format representation of the Experiment.\n\n        Returns:\n            dict: The dictionary form of the PulseQobjExperiment.\n        '
        out_dict = {'instructions': [x.to_dict() for x in self.instructions]}
        if hasattr(self, 'config'):
            out_dict['config'] = self.config.to_dict()
        if hasattr(self, 'header'):
            out_dict['header'] = self.header.to_dict()
        return out_dict

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        instructions_str = [repr(x) for x in self.instructions]
        instructions_repr = '[' + ', '.join(instructions_str) + ']'
        out = 'PulseQobjExperiment('
        out += instructions_repr
        if hasattr(self, 'config') or hasattr(self, 'header'):
            out += ', '
        if hasattr(self, 'config'):
            out += 'config=' + str(repr(self.config)) + ', '
        if hasattr(self, 'header'):
            out += 'header=' + str(repr(self.header)) + ', '
        out += ')'
        return out

    def __str__(self):
        if False:
            while True:
                i = 10
        out = '\nPulse Experiment:\n'
        if hasattr(self, 'config'):
            config = pprint.pformat(self.config.to_dict())
        else:
            config = '{}'
        if hasattr(self, 'header'):
            header = pprint.pformat(self.header.to_dict() or {})
        else:
            header = '{}'
        out += 'Header:\n%s\n' % header
        out += 'Config:\n%s\n\n' % config
        for instruction in self.instructions:
            out += '\t%s\n' % instruction
        return out

    @classmethod
    def from_dict(cls, data):
        if False:
            for i in range(10):
                print('nop')
        'Create a new PulseQobjExperiment object from a dictionary.\n\n        Args:\n            data (dict): A dictionary for the experiment config\n\n        Returns:\n            PulseQobjExperiment: The object from the input dictionary.\n        '
        config = None
        if 'config' in data:
            config = PulseQobjExperimentConfig.from_dict(data.pop('config'))
        header = None
        if 'header' in data:
            header = QobjExperimentHeader.from_dict(data.pop('header'))
        instructions = None
        if 'instructions' in data:
            instructions = [PulseQobjInstruction.from_dict(inst) for inst in data.pop('instructions')]
        return cls(instructions, config, header)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, PulseQobjExperiment):
            if self.to_dict() == other.to_dict():
                return True
        return False

class PulseQobjExperimentConfig(QobjDictField):
    """A config for a single Pulse experiment in the qobj."""

    def __init__(self, qubit_lo_freq=None, meas_lo_freq=None, **kwargs):
        if False:
            while True:
                i = 10
        'Instantiate a PulseQobjExperimentConfig object.\n\n        Args:\n            qubit_lo_freq (List[float]): List of qubit LO frequencies in GHz.\n            meas_lo_freq (List[float]): List of meas readout LO frequencies in GHz.\n            kwargs: Additional free form key value fields to add to the configuration\n        '
        if qubit_lo_freq is not None:
            self.qubit_lo_freq = qubit_lo_freq
        if meas_lo_freq is not None:
            self.meas_lo_freq = meas_lo_freq
        if kwargs:
            self.__dict__.update(kwargs)

class PulseLibraryItem:
    """An item in a pulse library."""

    def __init__(self, name, samples):
        if False:
            while True:
                i = 10
        'Instantiate a pulse library item.\n\n        Args:\n            name (str): A name for the pulse.\n            samples (list[complex]): A list of complex values defining pulse\n                shape.\n        '
        self.name = name
        if isinstance(samples[0], list):
            self.samples = numpy.array([complex(sample[0], sample[1]) for sample in samples])
        else:
            self.samples = samples

    def to_dict(self):
        if False:
            while True:
                i = 10
        'Return a dictionary format representation of the pulse library item.\n\n        Returns:\n            dict: The dictionary form of the PulseLibraryItem.\n        '
        return {'name': self.name, 'samples': self.samples}

    @classmethod
    def from_dict(cls, data):
        if False:
            return 10
        'Create a new PulseLibraryItem object from a dictionary.\n\n        Args:\n            data (dict): A dictionary for the experiment config\n\n        Returns:\n            PulseLibraryItem: The object from the input dictionary.\n        '
        return cls(**data)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'PulseLibraryItem({self.name}, {repr(self.samples)})'

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'Pulse Library Item:\n\tname: {self.name}\n\tsamples: {self.samples}'

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, PulseLibraryItem):
            if self.to_dict() == other.to_dict():
                return True
        return False

class PulseQobj:
    """A Pulse Qobj."""

    def __init__(self, qobj_id, config, experiments, header=None):
        if False:
            i = 10
            return i + 15
        'Instantiate a new Pulse Qobj Object.\n\n        Each Pulse Qobj object is used to represent a single payload that will\n        be passed to a Qiskit provider. It mirrors the Qobj the published\n        `Qobj specification <https://arxiv.org/abs/1809.03452>`_ for Pulse\n        experiments.\n\n        Args:\n            qobj_id (str): An identifier for the qobj\n            config (PulseQobjConfig): A config for the entire run\n            header (QobjHeader): A header for the entire run\n            experiments (list): A list of lists of :class:`PulseQobjExperiment`\n                objects representing an experiment\n        '
        self.qobj_id = qobj_id
        self.config = config
        self.header = header or QobjHeader()
        self.experiments = experiments
        self.type = 'PULSE'
        self.schema_version = '1.2.0'

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        experiments_str = [repr(x) for x in self.experiments]
        experiments_repr = '[' + ', '.join(experiments_str) + ']'
        out = "PulseQobj(qobj_id='{}', config={}, experiments={}, header={})".format(self.qobj_id, repr(self.config), experiments_repr, repr(self.header))
        return out

    def __str__(self):
        if False:
            i = 10
            return i + 15
        out = 'Pulse Qobj: %s:\n' % self.qobj_id
        config = pprint.pformat(self.config.to_dict())
        out += 'Config: %s\n' % str(config)
        header = pprint.pformat(self.header.to_dict())
        out += 'Header: %s\n' % str(header)
        out += 'Experiments:\n'
        for experiment in self.experiments:
            out += '%s' % str(experiment)
        return out

    def to_dict(self):
        if False:
            print('Hello World!')
        'Return a dictionary format representation of the Pulse Qobj.\n\n        Note this dict is not in the json wire format expected by IBMQ and qobj\n        specification because complex numbers are still of type complex. Also\n        this may contain native numpy arrays. When serializing this output\n        for use with IBMQ you can leverage a json encoder that converts these\n        as expected. For example:\n\n        .. code-block::\n\n            import json\n            import numpy\n\n            class QobjEncoder(json.JSONEncoder):\n                def default(self, obj):\n                    if isinstance(obj, numpy.ndarray):\n                        return obj.tolist()\n                    if isinstance(obj, complex):\n                        return (obj.real, obj.imag)\n                    return json.JSONEncoder.default(self, obj)\n\n            json.dumps(qobj.to_dict(), cls=QobjEncoder)\n\n        Returns:\n            dict: A dictionary representation of the PulseQobj object\n        '
        out_dict = {'qobj_id': self.qobj_id, 'header': self.header.to_dict(), 'config': self.config.to_dict(), 'schema_version': self.schema_version, 'type': self.type, 'experiments': [x.to_dict() for x in self.experiments]}
        return out_dict

    @classmethod
    def from_dict(cls, data):
        if False:
            return 10
        'Create a new PulseQobj object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the PulseQobj to create. It\n                will be in the same format as output by :func:`to_dict`.\n\n        Returns:\n            PulseQobj: The PulseQobj from the input dictionary.\n        '
        config = None
        if 'config' in data:
            config = PulseQobjConfig.from_dict(data['config'])
        experiments = None
        if 'experiments' in data:
            experiments = [PulseQobjExperiment.from_dict(exp) for exp in data['experiments']]
        header = None
        if 'header' in data:
            header = QobjHeader.from_dict(data['header'])
        return cls(qobj_id=data.get('qobj_id'), config=config, experiments=experiments, header=header)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, PulseQobj):
            if self.to_dict() == other.to_dict():
                return True
        return False