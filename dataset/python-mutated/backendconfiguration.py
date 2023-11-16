"""Backend Configuration Classes."""
import re
import copy
import numbers
from typing import Dict, List, Any, Iterable, Tuple, Union
from collections import defaultdict
from qiskit.exceptions import QiskitError
from qiskit.providers.exceptions import BackendConfigurationError
from qiskit.pulse.channels import AcquireChannel, Channel, ControlChannel, DriveChannel, MeasureChannel

class GateConfig:
    """Class representing a Gate Configuration

    Attributes:
        name: the gate name as it will be referred to in OpenQASM.
        parameters: variable names for the gate parameters (if any).
        qasm_def: definition of this gate in terms of OpenQASM 2 primitives U
                  and CX.
    """

    def __init__(self, name, parameters, qasm_def, coupling_map=None, latency_map=None, conditional=None, description=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a GateConfig object\n\n        Args:\n            name (str): the gate name as it will be referred to in OpenQASM.\n            parameters (list): variable names for the gate parameters (if any)\n                               as a list of strings.\n            qasm_def (str): definition of this gate in terms of OpenQASM 2 primitives U and CX.\n            coupling_map (list): An optional coupling map for the gate. In\n                the form of a list of lists of integers representing the qubit\n                groupings which are coupled by this gate.\n            latency_map (list): An optional map of latency for the gate. In the\n                the form of a list of lists of integers of either 0 or 1\n                representing an array of dimension\n                len(coupling_map) X n_registers that specifies the register\n                latency (1: fast, 0: slow) conditional operations on the gate\n            conditional (bool): Optionally specify whether this gate supports\n                conditional operations (true/false). If this is not specified,\n                then the gate inherits the conditional property of the backend.\n            description (str): Description of the gate operation\n        '
        self.name = name
        self.parameters = parameters
        self.qasm_def = qasm_def
        if coupling_map:
            self.coupling_map = coupling_map
        if latency_map:
            self.latency_map = latency_map
        if conditional is not None:
            self.conditional = conditional
        if description is not None:
            self.description = description

    @classmethod
    def from_dict(cls, data):
        if False:
            return 10
        'Create a new GateConfig object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the GateConfig to create.\n                         It will be in the same format as output by\n                         :func:`to_dict`.\n\n        Returns:\n            GateConfig: The GateConfig from the input dictionary.\n        '
        return cls(**data)

    def to_dict(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a dictionary format representation of the GateConfig.\n\n        Returns:\n            dict: The dictionary form of the GateConfig.\n        '
        out_dict = {'name': self.name, 'parameters': self.parameters, 'qasm_def': self.qasm_def}
        if hasattr(self, 'coupling_map'):
            out_dict['coupling_map'] = self.coupling_map
        if hasattr(self, 'latency_map'):
            out_dict['latency_map'] = self.latency_map
        if hasattr(self, 'conditional'):
            out_dict['conditional'] = self.conditional
        if hasattr(self, 'description'):
            out_dict['description'] = self.description
        return out_dict

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, GateConfig):
            if self.to_dict() == other.to_dict():
                return True
        return False

    def __repr__(self):
        if False:
            print('Hello World!')
        out_str = f'GateConfig({self.name}, {self.parameters}, {self.qasm_def}'
        for i in ['coupling_map', 'latency_map', 'conditional', 'description']:
            if hasattr(self, i):
                out_str += ', ' + repr(getattr(self, i))
        out_str += ')'
        return out_str

class UchannelLO:
    """Class representing a U Channel LO

    Attributes:
        q: Qubit that scale corresponds too.
        scale: Scale factor for qubit frequency.
    """

    def __init__(self, q, scale):
        if False:
            return 10
        'Initialize a UchannelLOSchema object\n\n        Args:\n            q (int): Qubit that scale corresponds too. Must be >= 0.\n            scale (complex): Scale factor for qubit frequency.\n\n        Raises:\n            QiskitError: If q is < 0\n        '
        if q < 0:
            raise QiskitError('q must be >=0')
        self.q = q
        self.scale = scale

    @classmethod
    def from_dict(cls, data):
        if False:
            print('Hello World!')
        'Create a new UchannelLO object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the UChannelLO to\n                create. It will be in the same format as output by\n                :func:`to_dict`.\n\n        Returns:\n            UchannelLO: The UchannelLO from the input dictionary.\n        '
        return cls(**data)

    def to_dict(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a dictionary format representation of the UChannelLO.\n\n        Returns:\n            dict: The dictionary form of the UChannelLO.\n        '
        out_dict = {'q': self.q, 'scale': self.scale}
        return out_dict

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, UchannelLO):
            if self.to_dict() == other.to_dict():
                return True
        return False

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'UchannelLO({self.q}, {self.scale})'

class QasmBackendConfiguration:
    """Class representing an OpenQASM 2.0 Backend Configuration.

    Attributes:
        backend_name: backend name.
        backend_version: backend version in the form X.Y.Z.
        n_qubits: number of qubits.
        basis_gates: list of basis gates names on the backend.
        gates: list of basis gates on the backend.
        local: backend is local or remote.
        simulator: backend is a simulator.
        conditional: backend supports conditional operations.
        open_pulse: backend supports open pulse.
        memory: backend supports memory.
        max_shots: maximum number of shots supported.
    """
    _data = {}

    def __init__(self, backend_name, backend_version, n_qubits, basis_gates, gates, local, simulator, conditional, open_pulse, memory, max_shots, coupling_map, supported_instructions=None, dynamic_reprate_enabled=False, rep_delay_range=None, default_rep_delay=None, max_experiments=None, sample_name=None, n_registers=None, register_map=None, configurable=None, credits_required=None, online_date=None, display_name=None, description=None, tags=None, dt=None, dtm=None, processor_type=None, parametric_pulses=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a QasmBackendConfiguration Object\n\n        Args:\n            backend_name (str): The backend name\n            backend_version (str): The backend version in the form X.Y.Z\n            n_qubits (int): the number of qubits for the backend\n            basis_gates (list): The list of strings for the basis gates of the\n                backends\n            gates (list): The list of GateConfig objects for the basis gates of\n                the backend\n            local (bool): True if the backend is local or False if remote\n            simulator (bool): True if the backend is a simulator\n            conditional (bool): True if the backend supports conditional\n                operations\n            open_pulse (bool): True if the backend supports OpenPulse\n            memory (bool): True if the backend supports memory\n            max_shots (int): The maximum number of shots allowed on the backend\n            coupling_map (list): The coupling map for the device\n            supported_instructions (List[str]): Instructions supported by the backend.\n            dynamic_reprate_enabled (bool): whether delay between programs can be set dynamically\n                (ie via ``rep_delay``). Defaults to False.\n            rep_delay_range (List[float]): 2d list defining supported range of repetition\n                delays for backend in μs. First entry is lower end of the range, second entry is\n                higher end of the range. Optional, but will be specified when\n                ``dynamic_reprate_enabled=True``.\n            default_rep_delay (float): Value of ``rep_delay`` if not specified by user and\n                ``dynamic_reprate_enabled=True``.\n            max_experiments (int): The maximum number of experiments per job\n            sample_name (str): Sample name for the backend\n            n_registers (int): Number of register slots available for feedback\n                (if conditional is True)\n            register_map (list): An array of dimension n_qubits X\n                n_registers that specifies whether a qubit can store a\n                measurement in a certain register slot.\n            configurable (bool): True if the backend is configurable, if the\n                backend is a simulator\n            credits_required (bool): True if backend requires credits to run a\n                job.\n            online_date (datetime.datetime): The date that the device went online\n            display_name (str): Alternate name field for the backend\n            description (str): A description for the backend\n            tags (list): A list of string tags to describe the backend\n            dt (float): Qubit drive channel timestep in nanoseconds.\n            dtm (float): Measurement drive channel timestep in nanoseconds.\n            processor_type (dict): Processor type for this backend. A dictionary of the\n                form ``{"family": <str>, "revision": <str>, segment: <str>}`` such as\n                ``{"family": "Canary", "revision": "1.0", segment: "A"}``.\n\n                - family: Processor family of this backend.\n                - revision: Revision version of this processor.\n                - segment: Segment this processor belongs to within a larger chip.\n            parametric_pulses (list): A list of pulse shapes which are supported on the backend.\n                For example: ``[\'gaussian\', \'constant\']``\n\n            **kwargs: optional fields\n        '
        self._data = {}
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.n_qubits = n_qubits
        self.basis_gates = basis_gates
        self.gates = gates
        self.local = local
        self.simulator = simulator
        self.conditional = conditional
        self.open_pulse = open_pulse
        self.memory = memory
        self.max_shots = max_shots
        self.coupling_map = coupling_map
        if supported_instructions:
            self.supported_instructions = supported_instructions
        self.dynamic_reprate_enabled = dynamic_reprate_enabled
        if rep_delay_range:
            self.rep_delay_range = [_rd * 1e-06 for _rd in rep_delay_range]
        if default_rep_delay is not None:
            self.default_rep_delay = default_rep_delay * 1e-06
        if max_experiments:
            self.max_experiments = max_experiments
        if sample_name is not None:
            self.sample_name = sample_name
        if n_registers:
            self.n_registers = 1
        if register_map:
            self.register_map = register_map
        if configurable is not None:
            self.configurable = configurable
        if credits_required is not None:
            self.credits_required = credits_required
        if online_date is not None:
            self.online_date = online_date
        if display_name is not None:
            self.display_name = display_name
        if description is not None:
            self.description = description
        if tags is not None:
            self.tags = tags
        if dt is not None:
            self.dt = dt * 1e-09
        if dtm is not None:
            self.dtm = dtm * 1e-09
        if processor_type is not None:
            self.processor_type = processor_type
        if parametric_pulses is not None:
            self.parametric_pulses = parametric_pulses
        if 'qubit_lo_range' in kwargs:
            kwargs['qubit_lo_range'] = [[min_range * 1000000000.0, max_range * 1000000000.0] for (min_range, max_range) in kwargs['qubit_lo_range']]
        if 'meas_lo_range' in kwargs:
            kwargs['meas_lo_range'] = [[min_range * 1000000000.0, max_range * 1000000000.0] for (min_range, max_range) in kwargs['meas_lo_range']]
        if 'rep_times' in kwargs:
            kwargs['rep_times'] = [_rt * 1e-06 for _rt in kwargs['rep_times']]
        self._data.update(kwargs)

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._data[name]
        except KeyError as ex:
            raise AttributeError(f'Attribute {name} is not defined') from ex

    @classmethod
    def from_dict(cls, data):
        if False:
            for i in range(10):
                print('nop')
        'Create a new GateConfig object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the GateConfig to create.\n                         It will be in the same format as output by\n                         :func:`to_dict`.\n        Returns:\n            GateConfig: The GateConfig from the input dictionary.\n        '
        in_data = copy.copy(data)
        gates = [GateConfig.from_dict(x) for x in in_data.pop('gates')]
        in_data['gates'] = gates
        return cls(**in_data)

    def to_dict(self):
        if False:
            print('Hello World!')
        'Return a dictionary format representation of the GateConfig.\n\n        Returns:\n            dict: The dictionary form of the GateConfig.\n        '
        out_dict = {'backend_name': self.backend_name, 'backend_version': self.backend_version, 'n_qubits': self.n_qubits, 'basis_gates': self.basis_gates, 'gates': [x.to_dict() for x in self.gates], 'local': self.local, 'simulator': self.simulator, 'conditional': self.conditional, 'open_pulse': self.open_pulse, 'memory': self.memory, 'max_shots': self.max_shots, 'coupling_map': self.coupling_map, 'dynamic_reprate_enabled': self.dynamic_reprate_enabled}
        if hasattr(self, 'supported_instructions'):
            out_dict['supported_instructions'] = self.supported_instructions
        if hasattr(self, 'rep_delay_range'):
            out_dict['rep_delay_range'] = [_rd * 1000000.0 for _rd in self.rep_delay_range]
        if hasattr(self, 'default_rep_delay'):
            out_dict['default_rep_delay'] = self.default_rep_delay * 1000000.0
        for kwarg in ['max_experiments', 'sample_name', 'n_registers', 'register_map', 'configurable', 'credits_required', 'online_date', 'display_name', 'description', 'tags', 'dt', 'dtm', 'processor_type', 'parametric_pulses']:
            if hasattr(self, kwarg):
                out_dict[kwarg] = getattr(self, kwarg)
        out_dict.update(self._data)
        if 'dt' in out_dict:
            out_dict['dt'] *= 1000000000.0
        if 'dtm' in out_dict:
            out_dict['dtm'] *= 1000000000.0
        if 'qubit_lo_range' in out_dict:
            out_dict['qubit_lo_range'] = [[min_range * 1e-09, max_range * 1e-09] for (min_range, max_range) in out_dict['qubit_lo_range']]
        if 'meas_lo_range' in out_dict:
            out_dict['meas_lo_range'] = [[min_range * 1e-09, max_range * 1e-09] for (min_range, max_range) in out_dict['meas_lo_range']]
        return out_dict

    @property
    def num_qubits(self):
        if False:
            i = 10
            return i + 15
        'Returns the number of qubits.\n\n        In future, `n_qubits` should be replaced in favor of `num_qubits` for consistent use\n        throughout Qiskit. Until this is properly refactored, this property serves as intermediate\n        solution.\n        '
        return self.n_qubits

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, QasmBackendConfiguration):
            if self.to_dict() == other.to_dict():
                return True
        return False

    def __contains__(self, item):
        if False:
            return 10
        return item in self.__dict__

class BackendConfiguration(QasmBackendConfiguration):
    """Backwards compat shim representing an abstract backend configuration."""
    pass

class PulseBackendConfiguration(QasmBackendConfiguration):
    """Static configuration state for an OpenPulse enabled backend. This contains information
    about the set up of the device which can be useful for building Pulse programs.
    """

    def __init__(self, backend_name: str, backend_version: str, n_qubits: int, basis_gates: List[str], gates: GateConfig, local: bool, simulator: bool, conditional: bool, open_pulse: bool, memory: bool, max_shots: int, coupling_map, n_uchannels: int, u_channel_lo: List[List[UchannelLO]], meas_levels: List[int], qubit_lo_range: List[List[float]], meas_lo_range: List[List[float]], dt: float, dtm: float, rep_times: List[float], meas_kernels: List[str], discriminators: List[str], hamiltonian: Dict[str, Any]=None, channel_bandwidth=None, acquisition_latency=None, conditional_latency=None, meas_map=None, max_experiments=None, sample_name=None, n_registers=None, register_map=None, configurable=None, credits_required=None, online_date=None, display_name=None, description=None, tags=None, channels: Dict[str, Any]=None, **kwargs):
        if False:
            return 10
        '\n        Initialize a backend configuration that contains all the extra configuration that is made\n        available for OpenPulse backends.\n\n        Args:\n            backend_name: backend name.\n            backend_version: backend version in the form X.Y.Z.\n            n_qubits: number of qubits.\n            basis_gates: list of basis gates names on the backend.\n            gates: list of basis gates on the backend.\n            local: backend is local or remote.\n            simulator: backend is a simulator.\n            conditional: backend supports conditional operations.\n            open_pulse: backend supports open pulse.\n            memory: backend supports memory.\n            max_shots: maximum number of shots supported.\n            coupling_map (list): The coupling map for the device\n            n_uchannels: Number of u-channels.\n            u_channel_lo: U-channel relationship on device los.\n            meas_levels: Supported measurement levels.\n            qubit_lo_range: Qubit lo ranges for each qubit with form (min, max) in GHz.\n            meas_lo_range: Measurement lo ranges for each qubit with form (min, max) in GHz.\n            dt: Qubit drive channel timestep in nanoseconds.\n            dtm: Measurement drive channel timestep in nanoseconds.\n            rep_times: Supported repetition times (program execution time) for backend in μs.\n            meas_kernels: Supported measurement kernels.\n            discriminators: Supported discriminators.\n            hamiltonian: An optional dictionary with fields characterizing the system hamiltonian.\n            channel_bandwidth (list): Bandwidth of all channels\n                (qubit, measurement, and U)\n            acquisition_latency (list): Array of dimension\n                n_qubits x n_registers. Latency (in units of dt) to write a\n                measurement result from qubit n into register slot m.\n            conditional_latency (list): Array of dimension n_channels\n                [d->u->m] x n_registers. Latency (in units of dt) to do a\n                conditional operation on channel n from register slot m\n            meas_map (list): Grouping of measurement which are multiplexed\n            max_experiments (int): The maximum number of experiments per job\n            sample_name (str): Sample name for the backend\n            n_registers (int): Number of register slots available for feedback\n                (if conditional is True)\n            register_map (list): An array of dimension n_qubits X\n                n_registers that specifies whether a qubit can store a\n                measurement in a certain register slot.\n            configurable (bool): True if the backend is configurable, if the\n                backend is a simulator\n            credits_required (bool): True if backend requires credits to run a\n                job.\n            online_date (datetime.datetime): The date that the device went online\n            display_name (str): Alternate name field for the backend\n            description (str): A description for the backend\n            tags (list): A list of string tags to describe the backend\n            channels: An optional dictionary containing information of each channel -- their\n                purpose, type, and qubits operated on.\n            **kwargs: Optional fields.\n        '
        self.n_uchannels = n_uchannels
        self.u_channel_lo = u_channel_lo
        self.meas_levels = meas_levels
        self.qubit_lo_range = [[min_range * 1000000000.0, max_range * 1000000000.0] for (min_range, max_range) in qubit_lo_range]
        self.meas_lo_range = [[min_range * 1000000000.0, max_range * 1000000000.0] for (min_range, max_range) in meas_lo_range]
        self.meas_kernels = meas_kernels
        self.discriminators = discriminators
        self.hamiltonian = hamiltonian
        if hamiltonian is not None:
            self.hamiltonian = dict(hamiltonian)
            self.hamiltonian['vars'] = {k: v * 1000000000.0 if isinstance(v, numbers.Number) else v for (k, v) in self.hamiltonian['vars'].items()}
        self.rep_times = [_rt * 1e-06 for _rt in rep_times]
        self.dt = dt * 1e-09
        self.dtm = dtm * 1e-09
        if channels is not None:
            self.channels = channels
            (self._qubit_channel_map, self._channel_qubit_map, self._control_channels) = self._parse_channels(channels=channels)
        else:
            self._control_channels = defaultdict(list)
        if channel_bandwidth is not None:
            self.channel_bandwidth = [[min_range * 1000000000.0, max_range * 1000000000.0] for (min_range, max_range) in channel_bandwidth]
        if acquisition_latency is not None:
            self.acquisition_latency = acquisition_latency
        if conditional_latency is not None:
            self.conditional_latency = conditional_latency
        if meas_map is not None:
            self.meas_map = meas_map
        super().__init__(backend_name=backend_name, backend_version=backend_version, n_qubits=n_qubits, basis_gates=basis_gates, gates=gates, local=local, simulator=simulator, conditional=conditional, open_pulse=open_pulse, memory=memory, max_shots=max_shots, coupling_map=coupling_map, max_experiments=max_experiments, sample_name=sample_name, n_registers=n_registers, register_map=register_map, configurable=configurable, credits_required=credits_required, online_date=online_date, display_name=display_name, description=description, tags=tags, **kwargs)

    @classmethod
    def from_dict(cls, data):
        if False:
            i = 10
            return i + 15
        'Create a new GateConfig object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the GateConfig to create.\n                It will be in the same format as output by :func:`to_dict`.\n\n        Returns:\n            GateConfig: The GateConfig from the input dictionary.\n        '
        in_data = copy.copy(data)
        gates = [GateConfig.from_dict(x) for x in in_data.pop('gates')]
        in_data['gates'] = gates
        input_uchannels = in_data.pop('u_channel_lo')
        u_channels = []
        for channel in input_uchannels:
            u_channels.append([UchannelLO.from_dict(x) for x in channel])
        in_data['u_channel_lo'] = u_channels
        return cls(**in_data)

    def to_dict(self):
        if False:
            while True:
                i = 10
        'Return a dictionary format representation of the GateConfig.\n\n        Returns:\n            dict: The dictionary form of the GateConfig.\n        '
        out_dict = super().to_dict()
        u_channel_lo = []
        for x in self.u_channel_lo:
            channel = []
            for y in x:
                channel.append(y.to_dict())
            u_channel_lo.append(channel)
        out_dict.update({'n_uchannels': self.n_uchannels, 'u_channel_lo': u_channel_lo, 'meas_levels': self.meas_levels, 'qubit_lo_range': self.qubit_lo_range, 'meas_lo_range': self.meas_lo_range, 'meas_kernels': self.meas_kernels, 'discriminators': self.discriminators, 'rep_times': self.rep_times, 'dt': self.dt, 'dtm': self.dtm})
        if hasattr(self, 'channel_bandwidth'):
            out_dict['channel_bandwidth'] = self.channel_bandwidth
        if hasattr(self, 'meas_map'):
            out_dict['meas_map'] = self.meas_map
        if hasattr(self, 'acquisition_latency'):
            out_dict['acquisition_latency'] = self.acquisition_latency
        if hasattr(self, 'conditional_latency'):
            out_dict['conditional_latency'] = self.conditional_latency
        if 'channels' in out_dict:
            out_dict.pop('_qubit_channel_map')
            out_dict.pop('_channel_qubit_map')
            out_dict.pop('_control_channels')
        if self.qubit_lo_range:
            out_dict['qubit_lo_range'] = [[min_range * 1e-09, max_range * 1e-09] for (min_range, max_range) in self.qubit_lo_range]
        if self.meas_lo_range:
            out_dict['meas_lo_range'] = [[min_range * 1e-09, max_range * 1e-09] for (min_range, max_range) in self.meas_lo_range]
        if self.rep_times:
            out_dict['rep_times'] = [_rt * 1000000.0 for _rt in self.rep_times]
        out_dict['dt'] *= 1000000000.0
        out_dict['dtm'] *= 1000000000.0
        if hasattr(self, 'channel_bandwidth'):
            out_dict['channel_bandwidth'] = [[min_range * 1e-09, max_range * 1e-09] for (min_range, max_range) in self.channel_bandwidth]
        if self.hamiltonian:
            hamiltonian = copy.deepcopy(self.hamiltonian)
            hamiltonian['vars'] = {k: v * 1e-09 if isinstance(v, numbers.Number) else v for (k, v) in hamiltonian['vars'].items()}
            out_dict['hamiltonian'] = hamiltonian
        if hasattr(self, 'channels'):
            out_dict['channels'] = self.channels
        return out_dict

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, QasmBackendConfiguration):
            if self.to_dict() == other.to_dict():
                return True
        return False

    @property
    def sample_rate(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Sample rate of the signal channels in Hz (1/dt).'
        return 1.0 / self.dt

    @property
    def control_channels(self) -> Dict[Tuple[int, ...], List]:
        if False:
            for i in range(10):
                print('nop')
        'Return the control channels'
        return self._control_channels

    def drive(self, qubit: int) -> DriveChannel:
        if False:
            while True:
                i = 10
        '\n        Return the drive channel for the given qubit.\n\n        Raises:\n            BackendConfigurationError: If the qubit is not a part of the system.\n\n        Returns:\n            Qubit drive channel.\n        '
        if not 0 <= qubit < self.n_qubits:
            raise BackendConfigurationError(f'Invalid index for {qubit}-qubit system.')
        return DriveChannel(qubit)

    def measure(self, qubit: int) -> MeasureChannel:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the measure stimulus channel for the given qubit.\n\n        Raises:\n            BackendConfigurationError: If the qubit is not a part of the system.\n        Returns:\n            Qubit measurement stimulus line.\n        '
        if not 0 <= qubit < self.n_qubits:
            raise BackendConfigurationError(f'Invalid index for {qubit}-qubit system.')
        return MeasureChannel(qubit)

    def acquire(self, qubit: int) -> AcquireChannel:
        if False:
            i = 10
            return i + 15
        '\n        Return the acquisition channel for the given qubit.\n\n        Raises:\n            BackendConfigurationError: If the qubit is not a part of the system.\n        Returns:\n            Qubit measurement acquisition line.\n        '
        if not 0 <= qubit < self.n_qubits:
            raise BackendConfigurationError(f'Invalid index for {qubit}-qubit systems.')
        return AcquireChannel(qubit)

    def control(self, qubits: Iterable[int]=None) -> List[ControlChannel]:
        if False:
            i = 10
            return i + 15
        '\n        Return the secondary drive channel for the given qubit -- typically utilized for\n        controlling multiqubit interactions. This channel is derived from other channels.\n\n        Args:\n            qubits: Tuple or list of qubits of the form `(control_qubit, target_qubit)`.\n\n        Raises:\n            BackendConfigurationError: If the ``qubits`` is not a part of the system or if\n                the backend does not provide `channels` information in its configuration.\n\n        Returns:\n            List of control channels.\n        '
        try:
            if isinstance(qubits, list):
                qubits = tuple(qubits)
            return self._control_channels[qubits]
        except KeyError as ex:
            raise BackendConfigurationError(f"Couldn't find the ControlChannel operating on qubits {qubits} on {self.n_qubits}-qubit system. The ControlChannel information is retrieved from the backend.") from ex
        except AttributeError as ex:
            raise BackendConfigurationError(f"This backend - '{self.backend_name}' does not provide channel information.") from ex

    def get_channel_qubits(self, channel: Channel) -> List[int]:
        if False:
            return 10
        '\n        Return a list of indices for qubits which are operated on directly by the given ``channel``.\n\n        Raises:\n            BackendConfigurationError: If ``channel`` is not a found or if\n                the backend does not provide `channels` information in its configuration.\n\n        Returns:\n            List of qubits operated on my the given ``channel``.\n        '
        try:
            return self._channel_qubit_map[channel]
        except KeyError as ex:
            raise BackendConfigurationError(f"Couldn't find the Channel - {channel}") from ex
        except AttributeError as ex:
            raise BackendConfigurationError(f"This backend - '{self.backend_name}' does not provide channel information.") from ex

    def get_qubit_channels(self, qubit: Union[int, Iterable[int]]) -> List[Channel]:
        if False:
            while True:
                i = 10
        'Return a list of channels which operate on the given ``qubit``.\n\n        Raises:\n            BackendConfigurationError: If ``qubit`` is not a found or if\n                the backend does not provide `channels` information in its configuration.\n\n        Returns:\n            List of ``Channel``\\s operated on my the given ``qubit``.\n        '
        channels = set()
        try:
            if isinstance(qubit, int):
                for key in self._qubit_channel_map.keys():
                    if qubit in key:
                        channels.update(self._qubit_channel_map[key])
                if len(channels) == 0:
                    raise KeyError
            elif isinstance(qubit, list):
                qubit = tuple(qubit)
                channels.update(self._qubit_channel_map[qubit])
            elif isinstance(qubit, tuple):
                channels.update(self._qubit_channel_map[qubit])
            return list(channels)
        except KeyError as ex:
            raise BackendConfigurationError(f"Couldn't find the qubit - {qubit}") from ex
        except AttributeError as ex:
            raise BackendConfigurationError(f"This backend - '{self.backend_name}' does not provide channel information.") from ex

    def describe(self, channel: ControlChannel) -> Dict[DriveChannel, complex]:
        if False:
            while True:
                i = 10
        '\n        Return a basic description of the channel dependency. Derived channels are given weights\n        which describe how their frames are linked to other frames.\n        For instance, the backend could be configured with this setting::\n\n            u_channel_lo = [\n                [UchannelLO(q=0, scale=1. + 0.j)],\n                [UchannelLO(q=0, scale=-1. + 0.j), UchannelLO(q=1, scale=1. + 0.j)]\n            ]\n\n        Then, this method can be used as follows::\n\n            backend.configuration().describe(ControlChannel(1))\n            >>> {DriveChannel(0): -1, DriveChannel(1): 1}\n\n        Args:\n            channel: The derived channel to describe.\n        Raises:\n            BackendConfigurationError: If channel is not a ControlChannel.\n        Returns:\n            Control channel derivations.\n        '
        if not isinstance(channel, ControlChannel):
            raise BackendConfigurationError('Can only describe ControlChannels.')
        result = {}
        for u_chan_lo in self.u_channel_lo[channel.index]:
            result[DriveChannel(u_chan_lo.q)] = u_chan_lo.scale
        return result

    def _parse_channels(self, channels: Dict[set, Any]) -> Dict[Any, Any]:
        if False:
            return 10
        '\n        Generates a dictionaries of ``Channel``\\s, and tuple of qubit(s) they operate on.\n\n        Args:\n            channels: An optional dictionary containing information of each channel -- their\n                purpose, type, and qubits operated on.\n\n        Returns:\n            qubit_channel_map: Dictionary mapping tuple of qubit(s) to list of ``Channel``\\s.\n            channel_qubit_map: Dictionary mapping ``Channel`` to list of qubit(s).\n            control_channels: Dictionary mapping tuple of qubit(s), to list of\n                ``ControlChannel``\\s.\n        '
        qubit_channel_map = defaultdict(list)
        channel_qubit_map = defaultdict(list)
        control_channels = defaultdict(list)
        channels_dict = {DriveChannel.prefix: DriveChannel, ControlChannel.prefix: ControlChannel, MeasureChannel.prefix: MeasureChannel, 'acquire': AcquireChannel}
        for (channel, config) in channels.items():
            (channel_prefix, index) = self._get_channel_prefix_index(channel)
            channel_type = channels_dict[channel_prefix]
            qubits = tuple(config['operates']['qubits'])
            if channel_prefix in channels_dict:
                qubit_channel_map[qubits].append(channel_type(index))
                channel_qubit_map[channel_type(index)].extend(list(qubits))
                if channel_prefix == ControlChannel.prefix:
                    control_channels[qubits].append(channel_type(index))
        return (dict(qubit_channel_map), dict(channel_qubit_map), dict(control_channels))

    def _get_channel_prefix_index(self, channel: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return channel prefix and index from the given ``channel``.\n\n        Args:\n            channel: Name of channel.\n\n        Raises:\n            BackendConfigurationError: If invalid channel name is found.\n\n        Return:\n            Channel name and index. For example, if ``channel=acquire0``, this method\n            returns ``acquire`` and ``0``.\n        '
        channel_prefix = re.match('(?P<channel>[a-z]+)(?P<index>[0-9]+)', channel)
        try:
            return (channel_prefix.group('channel'), int(channel_prefix.group('index')))
        except AttributeError as ex:
            raise BackendConfigurationError(f"Invalid channel name - '{channel}' found.") from ex