"""Module providing definitions of OpenQASM 2 Qobj classes."""
import copy
import pprint
from types import SimpleNamespace
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.qobj.pulse_qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.qobj.common import QobjDictField, QobjHeader

class QasmQobjInstruction:
    """A class representing a single instruction in an QasmQobj Experiment."""

    def __init__(self, name, params=None, qubits=None, register=None, memory=None, condition=None, conditional=None, label=None, mask=None, relation=None, val=None, snapshot_type=None):
        if False:
            while True:
                i = 10
        'Instantiate a new QasmQobjInstruction object.\n\n        Args:\n            name (str): The name of the instruction\n            params (list): The list of parameters for the gate\n            qubits (list): A list of ``int`` representing the qubits the\n                instruction operates on\n            register (list): If a ``measure`` instruction this is a list\n                of ``int`` containing the list of register slots in which to\n                store the measurement results (must be the same length as\n                qubits). If a ``bfunc`` instruction this is a single ``int``\n                of the register slot in which to store the result.\n            memory (list): If a ``measure`` instruction this is a list\n                of ``int`` containing the list of memory slots to store the\n                measurement results in (must be the same length as qubits).\n                If a ``bfunc`` instruction this is a single ``int`` of the\n                memory slot to store the boolean function result in.\n            condition (tuple): A tuple of the form ``(int, int)`` where the\n                first ``int`` is the control register and the second ``int`` is\n                the control value if the gate has a condition.\n            conditional (int):  The register index of the condition\n            label (str): An optional label assigned to the instruction\n            mask (int): For a ``bfunc`` instruction the hex value which is\n                applied as an ``AND`` to the register bits.\n            relation (str): Relational  operator  for  comparing  the  masked\n                register to the ``val`` kwarg. Can be either ``==`` (equals) or\n                ``!=`` (not equals).\n            val (int): Value to which to compare the masked register. In other\n                words, the output of the function is ``(register AND mask)``\n            snapshot_type (str): For snapshot instructions the type of snapshot\n                to use\n        '
        self.name = name
        if params is not None:
            self.params = params
        if qubits is not None:
            self.qubits = qubits
        if register is not None:
            self.register = register
        if memory is not None:
            self.memory = memory
        if condition is not None:
            self._condition = condition
        if conditional is not None:
            self.conditional = conditional
        if label is not None:
            self.label = label
        if mask is not None:
            self.mask = mask
        if relation is not None:
            self.relation = relation
        if val is not None:
            self.val = val
        if snapshot_type is not None:
            self.snapshot_type = snapshot_type

    def to_dict(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a dictionary format representation of the Instruction.\n\n        Returns:\n            dict: The dictionary form of the QasmQobjInstruction.\n        '
        out_dict = {'name': self.name}
        for attr in ['params', 'qubits', 'register', 'memory', '_condition', 'conditional', 'label', 'mask', 'relation', 'val', 'snapshot_type']:
            if hasattr(self, attr):
                if attr == 'params':
                    params = []
                    for param in list(getattr(self, attr)):
                        if isinstance(param, ParameterExpression):
                            params.append(float(param))
                        else:
                            params.append(param)
                    out_dict[attr] = params
                else:
                    out_dict[attr] = getattr(self, attr)
        return out_dict

    def __repr__(self):
        if False:
            return 10
        out = "QasmQobjInstruction(name='%s'" % self.name
        for attr in ['params', 'qubits', 'register', 'memory', '_condition', 'conditional', 'label', 'mask', 'relation', 'val', 'snapshot_type']:
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
            i = 10
            return i + 15
        out = 'Instruction: %s\n' % self.name
        for attr in ['params', 'qubits', 'register', 'memory', '_condition', 'conditional', 'label', 'mask', 'relation', 'val', 'snapshot_type']:
            if hasattr(self, attr):
                out += f'\t\t{attr}: {getattr(self, attr)}\n'
        return out

    @classmethod
    def from_dict(cls, data):
        if False:
            print('Hello World!')
        'Create a new QasmQobjInstruction object from a dictionary.\n\n        Args:\n            data (dict): A dictionary for the experiment config\n\n        Returns:\n            QasmQobjInstruction: The object from the input dictionary.\n        '
        name = data.pop('name')
        return cls(name, **data)

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, QasmQobjInstruction):
            if self.to_dict() == other.to_dict():
                return True
        return False

class QasmQobjExperiment:
    """An OpenQASM 2 Qobj Experiment.

    Each instance of this class is used to represent an OpenQASM 2 experiment as
    part of a larger OpenQASM 2 qobj.
    """

    def __init__(self, config=None, header=None, instructions=None):
        if False:
            i = 10
            return i + 15
        'Instantiate a QasmQobjExperiment.\n\n        Args:\n            config (QasmQobjExperimentConfig): A config object for the experiment\n            header (QasmQobjExperimentHeader): A header object for the experiment\n            instructions (list): A list of :class:`QasmQobjInstruction` objects\n        '
        self.config = config or QasmQobjExperimentConfig()
        self.header = header or QasmQobjExperimentHeader()
        self.instructions = instructions or []

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        instructions_str = [repr(x) for x in self.instructions]
        instructions_repr = '[' + ', '.join(instructions_str) + ']'
        out = 'QasmQobjExperiment(config={}, header={}, instructions={})'.format(repr(self.config), repr(self.header), instructions_repr)
        return out

    def __str__(self):
        if False:
            i = 10
            return i + 15
        out = '\nOpenQASM2 Experiment:\n'
        config = pprint.pformat(self.config.to_dict())
        header = pprint.pformat(self.header.to_dict())
        out += 'Header:\n%s\n' % header
        out += 'Config:\n%s\n\n' % config
        for instruction in self.instructions:
            out += '\t%s\n' % instruction
        return out

    def to_dict(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a dictionary format representation of the Experiment.\n\n        Returns:\n            dict: The dictionary form of the QasmQObjExperiment.\n        '
        out_dict = {'config': self.config.to_dict(), 'header': self.header.to_dict(), 'instructions': [x.to_dict() for x in self.instructions]}
        return out_dict

    @classmethod
    def from_dict(cls, data):
        if False:
            i = 10
            return i + 15
        'Create a new QasmQobjExperiment object from a dictionary.\n\n        Args:\n            data (dict): A dictionary for the experiment config\n\n        Returns:\n            QasmQobjExperiment: The object from the input dictionary.\n        '
        config = None
        if 'config' in data:
            config = QasmQobjExperimentConfig.from_dict(data.pop('config'))
        header = None
        if 'header' in data:
            header = QasmQobjExperimentHeader.from_dict(data.pop('header'))
        instructions = None
        if 'instructions' in data:
            instructions = [QasmQobjInstruction.from_dict(inst) for inst in data.pop('instructions')]
        return cls(config, header, instructions)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, QasmQobjExperiment):
            if self.to_dict() == other.to_dict():
                return True
        return False

class QasmQobjConfig(SimpleNamespace):
    """A configuration for an OpenQASM 2 Qobj."""

    def __init__(self, shots=None, seed_simulator=None, memory=None, parameter_binds=None, meas_level=None, meas_return=None, memory_slots=None, n_qubits=None, pulse_library=None, calibrations=None, rep_delay=None, qubit_lo_freq=None, meas_lo_freq=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Model for RunConfig.\n\n        Args:\n            shots (int): the number of shots.\n            seed_simulator (int): the seed to use in the simulator\n            memory (bool): whether to request memory from backend (per-shot readouts)\n            parameter_binds (list[dict]): List of parameter bindings\n            meas_level (int): Measurement level 0, 1, or 2\n            meas_return (str): For measurement level < 2, whether single or avg shots are returned\n            memory_slots (int): The number of memory slots on the device\n            n_qubits (int): The number of qubits on the device\n            pulse_library (list): List of :class:`PulseLibraryItem`.\n            calibrations (QasmExperimentCalibrations): Information required for Pulse gates.\n            rep_delay (float): Delay between programs in sec. Only supported on certain\n                backends (``backend.configuration().dynamic_reprate_enabled`` ). Must be from the\n                range supplied by the backend (``backend.configuration().rep_delay_range``). Default\n                is ``backend.configuration().default_rep_delay``.\n            qubit_lo_freq (list): List of frequencies (as floats) for the qubit driver LO's in GHz.\n            meas_lo_freq (list): List of frequencies (as floats) for the measurement driver LO's in\n                GHz.\n            kwargs: Additional free form key value fields to add to the\n                configuration.\n        "
        if shots is not None:
            self.shots = int(shots)
        if seed_simulator is not None:
            self.seed_simulator = int(seed_simulator)
        if memory is not None:
            self.memory = bool(memory)
        if parameter_binds is not None:
            self.parameter_binds = parameter_binds
        if meas_level is not None:
            self.meas_level = meas_level
        if meas_return is not None:
            self.meas_return = meas_return
        if memory_slots is not None:
            self.memory_slots = memory_slots
        if n_qubits is not None:
            self.n_qubits = n_qubits
        if pulse_library is not None:
            self.pulse_library = pulse_library
        if calibrations is not None:
            self.calibrations = calibrations
        if rep_delay is not None:
            self.rep_delay = rep_delay
        if qubit_lo_freq is not None:
            self.qubit_lo_freq = qubit_lo_freq
        if meas_lo_freq is not None:
            self.meas_lo_freq = meas_lo_freq
        if kwargs:
            self.__dict__.update(kwargs)

    def to_dict(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a dictionary format representation of the OpenQASM 2 Qobj config.\n\n        Returns:\n            dict: The dictionary form of the QasmQobjConfig.\n        '
        out_dict = copy.copy(self.__dict__)
        if hasattr(self, 'pulse_library'):
            out_dict['pulse_library'] = [x.to_dict() for x in self.pulse_library]
        if hasattr(self, 'calibrations'):
            out_dict['calibrations'] = self.calibrations.to_dict()
        return out_dict

    @classmethod
    def from_dict(cls, data):
        if False:
            return 10
        'Create a new QasmQobjConfig object from a dictionary.\n\n        Args:\n            data (dict): A dictionary for the config\n\n        Returns:\n            QasmQobjConfig: The object from the input dictionary.\n        '
        if 'pulse_library' in data:
            pulse_lib = data.pop('pulse_library')
            pulse_lib_obj = [PulseLibraryItem.from_dict(x) for x in pulse_lib]
            data['pulse_library'] = pulse_lib_obj
        if 'calibrations' in data:
            calibrations = data.pop('calibrations')
            data['calibrations'] = QasmExperimentCalibrations.from_dict(calibrations)
        return cls(**data)

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, QasmQobjConfig):
            if self.to_dict() == other.to_dict():
                return True
        return False

class QasmQobjExperimentHeader(QobjDictField):
    """A header for a single OpenQASM 2 experiment in the qobj."""
    pass

class QasmQobjExperimentConfig(QobjDictField):
    """Configuration for a single OpenQASM 2 experiment in the qobj."""

    def __init__(self, calibrations=None, qubit_lo_freq=None, meas_lo_freq=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Args:\n            calibrations (QasmExperimentCalibrations): Information required for Pulse gates.\n            qubit_lo_freq (List[float]): List of qubit LO frequencies in GHz.\n            meas_lo_freq (List[float]): List of meas readout LO frequencies in GHz.\n            kwargs: Additional free form key value fields to add to the configuration\n        '
        if calibrations:
            self.calibrations = calibrations
        if qubit_lo_freq is not None:
            self.qubit_lo_freq = qubit_lo_freq
        if meas_lo_freq is not None:
            self.meas_lo_freq = meas_lo_freq
        super().__init__(**kwargs)

    def to_dict(self):
        if False:
            return 10
        out_dict = copy.copy(self.__dict__)
        if hasattr(self, 'calibrations'):
            out_dict['calibrations'] = self.calibrations.to_dict()
        return out_dict

    @classmethod
    def from_dict(cls, data):
        if False:
            print('Hello World!')
        if 'calibrations' in data:
            calibrations = data.pop('calibrations')
            data['calibrations'] = QasmExperimentCalibrations.from_dict(calibrations)
        return cls(**data)

class QasmExperimentCalibrations:
    """A container for any calibrations data. The gates attribute contains a list of
    GateCalibrations.
    """

    def __init__(self, gates):
        if False:
            print('Hello World!')
        '\n        Initialize a container for calibrations.\n\n        Args:\n            gates (list(GateCalibration))\n        '
        self.gates = gates

    def to_dict(self):
        if False:
            i = 10
            return i + 15
        'Return a dictionary format representation of the calibrations.\n\n        Returns:\n            dict: The dictionary form of the GateCalibration.\n\n        '
        out_dict = copy.copy(self.__dict__)
        out_dict['gates'] = [x.to_dict() for x in self.gates]
        return out_dict

    @classmethod
    def from_dict(cls, data):
        if False:
            while True:
                i = 10
        'Create a new GateCalibration object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the QasmExperimentCalibrations to\n                         create. It will be in the same format as output by :func:`to_dict`.\n\n        Returns:\n            QasmExperimentCalibrations: The QasmExperimentCalibrations from the input dictionary.\n        '
        gates = data.pop('gates')
        data['gates'] = [GateCalibration.from_dict(x) for x in gates]
        return cls(**data)

class GateCalibration:
    """Each calibration specifies a unique gate by name, qubits and params, and
    contains the Pulse instructions to implement it."""

    def __init__(self, name, qubits, params, instructions):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize a single gate calibration. Instructions may reference waveforms which should be\n        made available in the pulse_library.\n\n        Args:\n            name (str): Gate name.\n            qubits (list(int)): Qubits the gate applies to.\n            params (list(complex)): Gate parameter values, if any.\n            instructions (list(PulseQobjInstruction)): The gate implementation.\n        '
        self.name = name
        self.qubits = qubits
        self.params = params
        self.instructions = instructions

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash((self.name, tuple(self.qubits), tuple(self.params), tuple((str(inst) for inst in self.instructions))))

    def to_dict(self):
        if False:
            return 10
        'Return a dictionary format representation of the Gate Calibration.\n\n        Returns:\n            dict: The dictionary form of the GateCalibration.\n        '
        out_dict = copy.copy(self.__dict__)
        out_dict['instructions'] = [x.to_dict() for x in self.instructions]
        return out_dict

    @classmethod
    def from_dict(cls, data):
        if False:
            print('Hello World!')
        'Create a new GateCalibration object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the GateCalibration to create. It\n                will be in the same format as output by :func:`to_dict`.\n\n        Returns:\n            GateCalibration: The GateCalibration from the input dictionary.\n        '
        instructions = data.pop('instructions')
        data['instructions'] = [PulseQobjInstruction.from_dict(x) for x in instructions]
        return cls(**data)

class QasmQobj:
    """An OpenQASM 2 Qobj."""

    def __init__(self, qobj_id=None, config=None, experiments=None, header=None):
        if False:
            i = 10
            return i + 15
        'Instantiate a new OpenQASM 2 Qobj Object.\n\n        Each OpenQASM 2 Qobj object is used to represent a single payload that will\n        be passed to a Qiskit provider. It mirrors the Qobj the published\n        `Qobj specification <https://arxiv.org/abs/1809.03452>`_ for OpenQASM\n        experiments.\n\n        Args:\n            qobj_id (str): An identifier for the qobj\n            config (QasmQobjRunConfig): A config for the entire run\n            header (QobjHeader): A header for the entire run\n            experiments (list): A list of lists of :class:`QasmQobjExperiment`\n                objects representing an experiment\n        '
        self.header = header or QobjHeader()
        self.config = config or QasmQobjConfig()
        self.experiments = experiments or []
        self.qobj_id = qobj_id
        self.type = 'QASM'
        self.schema_version = '1.3.0'

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        experiments_str = [repr(x) for x in self.experiments]
        experiments_repr = '[' + ', '.join(experiments_str) + ']'
        out = "QasmQobj(qobj_id='{}', config={}, experiments={}, header={})".format(self.qobj_id, repr(self.config), experiments_repr, repr(self.header))
        return out

    def __str__(self):
        if False:
            i = 10
            return i + 15
        out = 'QASM Qobj: %s:\n' % self.qobj_id
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
            i = 10
            return i + 15
        'Return a dictionary format representation of the OpenQASM 2 Qobj.\n\n        Note this dict is not in the json wire format expected by IBM and Qobj\n        specification because complex numbers are still of type complex. Also,\n        this may contain native numpy arrays. When serializing this output\n        for use with IBM systems, you can leverage a json encoder that converts these\n        as expected. For example:\n\n        .. code-block::\n\n            import json\n            import numpy\n\n            class QobjEncoder(json.JSONEncoder):\n                def default(self, obj):\n                    if isinstance(obj, numpy.ndarray):\n                        return obj.tolist()\n                    if isinstance(obj, complex):\n                        return (obj.real, obj.imag)\n                    return json.JSONEncoder.default(self, obj)\n\n            json.dumps(qobj.to_dict(), cls=QobjEncoder)\n\n        Returns:\n            dict: A dictionary representation of the QasmQobj object\n        '
        out_dict = {'qobj_id': self.qobj_id, 'header': self.header.to_dict(), 'config': self.config.to_dict(), 'schema_version': self.schema_version, 'type': 'QASM', 'experiments': [x.to_dict() for x in self.experiments]}
        return out_dict

    @classmethod
    def from_dict(cls, data):
        if False:
            while True:
                i = 10
        'Create a new QASMQobj object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the QasmQobj to create. It\n                will be in the same format as output by :func:`to_dict`.\n\n        Returns:\n            QasmQobj: The QasmQobj from the input dictionary.\n        '
        config = None
        if 'config' in data:
            config = QasmQobjConfig.from_dict(data['config'])
        experiments = None
        if 'experiments' in data:
            experiments = [QasmQobjExperiment.from_dict(exp) for exp in data['experiments']]
        header = None
        if 'header' in data:
            header = QobjHeader.from_dict(data['header'])
        return cls(qobj_id=data.get('qobj_id'), config=config, experiments=experiments, header=header)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, QasmQobj):
            if self.to_dict() == other.to_dict():
                return True
        return False