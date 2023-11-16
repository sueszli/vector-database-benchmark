"""Backend abstract interface for providers."""
from __future__ import annotations
from typing import List, Iterable, Any, Dict, Optional
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.providers.backend import QubitProperties
from qiskit.utils.units import apply_prefix
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.circuit.measure import Measure
from qiskit.providers.models.backendconfiguration import BackendConfiguration
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.providers.models.pulsedefaults import PulseDefaults
from qiskit.providers.options import Options
from qiskit.providers.exceptions import BackendPropertyError

def convert_to_target(configuration: BackendConfiguration, properties: BackendProperties=None, defaults: PulseDefaults=None, custom_name_mapping: Optional[Dict[str, Any]]=None, add_delay: bool=False, filter_faulty: bool=False):
    if False:
        for i in range(10):
            print('nop')
    'Uses configuration, properties and pulse defaults\n    to construct and return Target class.\n\n    In order to convert with a ``defaults.instruction_schedule_map``,\n    which has a custom calibration for an operation,\n    the operation name must be in ``configuration.basis_gates`` and\n    ``custom_name_mapping`` must be supplied for the operation.\n    Otherwise, the operation will be dropped in the resulting ``Target`` object.\n\n    That suggests it is recommended to add custom calibrations **after** creating a target\n    with this function instead of adding them to ``defaults`` in advance. For example::\n\n        target.add_instruction(custom_gate, {(0, 1): InstructionProperties(calibration=custom_sched)})\n    '
    from qiskit.transpiler.target import Target, InstructionProperties
    name_mapping = get_standard_gate_name_mapping()
    target = None
    if custom_name_mapping is not None:
        name_mapping.update(custom_name_mapping)
    faulty_qubits = set()
    if properties is not None:
        if filter_faulty:
            faulty_qubits = set(properties.faulty_qubits())
        qubit_properties = qubit_props_list_from_props(properties=properties)
        target = Target(num_qubits=configuration.n_qubits, qubit_properties=qubit_properties, concurrent_measurements=getattr(configuration, 'meas_map', None))
        gates: Dict[str, Any] = {}
        for gate in properties.gates:
            name = gate.gate
            if name in name_mapping:
                if name not in gates:
                    gates[name] = {}
            else:
                raise QiskitError(f'Operation name {name} does not have a known mapping. Use custom_name_mapping to map this name to an Operation object')
            qubits = tuple(gate.qubits)
            if filter_faulty:
                if any((not properties.is_qubit_operational(qubit) for qubit in qubits)):
                    continue
                if not properties.is_gate_operational(name, gate.qubits):
                    continue
            gate_props = {}
            for param in gate.parameters:
                if param.name == 'gate_error':
                    gate_props['error'] = param.value
                if param.name == 'gate_length':
                    gate_props['duration'] = apply_prefix(param.value, param.unit)
            gates[name][qubits] = InstructionProperties(**gate_props)
        for (gate, props) in gates.items():
            inst = name_mapping[gate]
            target.add_instruction(inst, props)
        measure_props = {}
        for (qubit, _) in enumerate(properties.qubits):
            if filter_faulty:
                if not properties.is_qubit_operational(qubit):
                    continue
            try:
                duration = properties.readout_length(qubit)
            except BackendPropertyError:
                duration = None
            try:
                error = properties.readout_error(qubit)
            except BackendPropertyError:
                error = None
            measure_props[qubit,] = InstructionProperties(duration=duration, error=error)
        target.add_instruction(Measure(), measure_props)
    else:
        target = Target(num_qubits=configuration.n_qubits, concurrent_measurements=getattr(configuration, 'meas_map', None))
        for gate in configuration.gates:
            name = gate.name
            gate_props = {tuple(x): None for x in gate.coupling_map} if hasattr(gate, 'coupling_map') else {None: None}
            if name in name_mapping:
                target.add_instruction(name_mapping[name], gate_props)
            else:
                raise QiskitError(f'Operation name {name} does not have a known mapping. Use custom_name_mapping to map this name to an Operation object')
        target.add_instruction(Measure())
    if hasattr(configuration, 'dt'):
        target.dt = configuration.dt
    if hasattr(configuration, 'timing_constraints'):
        target.granularity = configuration.timing_constraints.get('granularity')
        target.min_length = configuration.timing_constraints.get('min_length')
        target.pulse_alignment = configuration.timing_constraints.get('pulse_alignment')
        target.acquire_alignment = configuration.timing_constraints.get('acquire_alignment')
    if defaults is not None:
        inst_map = defaults.instruction_schedule_map
        for inst in inst_map.instructions:
            for qarg in inst_map.qubits_with_instruction(inst):
                try:
                    qargs = tuple(qarg)
                except TypeError:
                    qargs = (qarg,)
                calibration_entry = inst_map._get_calibration_entry(inst, qargs)
                if inst in target:
                    if inst == 'measure':
                        for qubit in qargs:
                            if filter_faulty and qubit in faulty_qubits:
                                continue
                            target[inst][qubit,].calibration = calibration_entry
                    elif qargs in target[inst]:
                        if filter_faulty and any((qubit in faulty_qubits for qubit in qargs)):
                            continue
                        target[inst][qargs].calibration = calibration_entry
    combined_global_ops = set()
    if configuration.basis_gates:
        combined_global_ops.update(configuration.basis_gates)
    for op in combined_global_ops:
        if op not in target:
            if op in name_mapping:
                target.add_instruction(name_mapping[op], name=op)
            else:
                raise QiskitError(f"Operation name '{op}' does not have a known mapping. Use custom_name_mapping to map this name to an Operation object")
    if add_delay and 'delay' not in target:
        target.add_instruction(name_mapping['delay'], {(bit,): None for bit in range(target.num_qubits) if bit not in faulty_qubits})
    return target

def qubit_props_list_from_props(properties: BackendProperties) -> List[QubitProperties]:
    if False:
        while True:
            i = 10
    'Uses BackendProperties to construct\n    and return a list of QubitProperties.\n    '
    qubit_props: List[QubitProperties] = []
    for (qubit, _) in enumerate(properties.qubits):
        try:
            t_1 = properties.t1(qubit)
        except BackendPropertyError:
            t_1 = None
        try:
            t_2 = properties.t2(qubit)
        except BackendPropertyError:
            t_2 = None
        try:
            frequency = properties.frequency(qubit)
        except BackendPropertyError:
            frequency = None
        qubit_props.append(QubitProperties(t1=t_1, t2=t_2, frequency=frequency))
    return qubit_props

class BackendV2Converter(BackendV2):
    """A converter class that takes a :class:`~.BackendV1` instance and wraps it in a
    :class:`~.BackendV2` interface.

    This class implements the :class:`~.BackendV2` interface and is used to enable
    common access patterns between :class:`~.BackendV1` and :class:`~.BackendV2`. This
    class should only be used if you need a :class:`~.BackendV2` and still need
    compatibility with :class:`~.BackendV1`.

    When using custom calibrations (or other custom workflows) it is **not** recommended
    to mutate the ``BackendV1`` object before applying this converter. For example, in order to
    convert a ``BackendV1`` object with a customized ``defaults().instruction_schedule_map``,
    which has a custom calibration for an operation, the operation name must be in
    ``configuration().basis_gates`` and ``name_mapping`` must be supplied for the operation.
    Otherwise, the operation will be dropped in the resulting ``BackendV2`` object.

    Instead it is typically better to add custom calibrations **after** applying this converter
    instead of updating ``BackendV1.defaults()`` in advance. For example::

        backend_v2 = BackendV2Converter(backend_v1)
        backend_v2.target.add_instruction(
            custom_gate, {(0, 1): InstructionProperties(calibration=custom_sched)}
        )
    """

    def __init__(self, backend: BackendV1, name_mapping: Optional[Dict[str, Any]]=None, add_delay: bool=False, filter_faulty: bool=False):
        if False:
            i = 10
            return i + 15
        'Initialize a BackendV2 converter instance based on a BackendV1 instance.\n\n        Args:\n            backend: The input :class:`~.BackendV1` based backend to wrap in a\n                :class:`~.BackendV2` interface\n            name_mapping: An optional dictionary that maps custom gate/operation names in\n                ``backend`` to an :class:`~.Operation` object representing that\n                gate/operation. By default most standard gates names are mapped to the\n                standard gate object from :mod:`qiskit.circuit.library` this only needs\n                to be specified if the input ``backend`` defines gates in names outside\n                that set.\n            add_delay: If set to true a :class:`~qiskit.circuit.Delay` operation\n                will be added to the target as a supported operation for all\n                qubits\n            filter_faulty: If the :class:`~.BackendProperties` object (if present) for\n                ``backend`` has any qubits or gates flagged as non-operational filter\n                those from the output target.\n        '
        self._backend = backend
        self._config = self._backend.configuration()
        super().__init__(provider=backend.provider, name=backend.name(), description=self._config.description, online_date=getattr(self._config, 'online_date', None), backend_version=self._config.backend_version)
        self._options = self._backend._options
        self._properties = None
        if hasattr(self._backend, 'properties'):
            self._properties = self._backend.properties()
        self._defaults = None
        self._target = None
        self._name_mapping = name_mapping
        self._add_delay = add_delay
        self._filter_faulty = filter_faulty

    @property
    def target(self):
        if False:
            i = 10
            return i + 15
        'A :class:`qiskit.transpiler.Target` object for the backend.\n\n        :rtype: Target\n        '
        if self._target is None:
            if self._defaults is None and hasattr(self._backend, 'defaults'):
                self._defaults = self._backend.defaults()
            if self._properties is None and hasattr(self._backend, 'properties'):
                self._properties = self._backend.properties()
            self._target = convert_to_target(self._config, self._properties, self._defaults, custom_name_mapping=self._name_mapping, add_delay=self._add_delay, filter_faulty=self._filter_faulty)
        return self._target

    @property
    def max_circuits(self):
        if False:
            return 10
        return self._config.max_experiments

    @classmethod
    def _default_options(cls):
        if False:
            print('Hello World!')
        return Options()

    @property
    def dtm(self) -> float:
        if False:
            return 10
        return self._config.dtm

    @property
    def meas_map(self) -> List[List[int]]:
        if False:
            while True:
                i = 10
        return self._config.meas_map

    def drive_channel(self, qubit: int):
        if False:
            print('Hello World!')
        return self._config.drive(qubit)

    def measure_channel(self, qubit: int):
        if False:
            i = 10
            return i + 15
        return self._config.measure(qubit)

    def acquire_channel(self, qubit: int):
        if False:
            return 10
        return self._config.acquire(qubit)

    def control_channel(self, qubits: Iterable[int]):
        if False:
            print('Hello World!')
        return self._config.control(qubits)

    def run(self, run_input, **options):
        if False:
            for i in range(10):
                print('nop')
        return self._backend.run(run_input, **options)