"""Pass Manager Configuration class."""
import pprint
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.instruction_durations import InstructionDurations

class PassManagerConfig:
    """Pass Manager Configuration."""

    def __init__(self, initial_layout=None, basis_gates=None, inst_map=None, coupling_map=None, layout_method=None, routing_method=None, translation_method=None, scheduling_method=None, instruction_durations=None, backend_properties=None, approximation_degree=None, seed_transpiler=None, timing_constraints=None, unitary_synthesis_method='default', unitary_synthesis_plugin_config=None, target=None, hls_config=None, init_method=None, optimization_method=None):
        if False:
            i = 10
            return i + 15
        'Initialize a PassManagerConfig object\n\n        Args:\n            initial_layout (Layout): Initial position of virtual qubits on\n                physical qubits.\n            basis_gates (list): List of basis gate names to unroll to.\n            inst_map (InstructionScheduleMap): Mapping object that maps gate to schedule.\n            coupling_map (CouplingMap): Directed graph represented a coupling\n                map.\n            layout_method (str): the pass to use for choosing initial qubit\n                placement. This will be the plugin name if an external layout stage\n                plugin is being used.\n            routing_method (str): the pass to use for routing qubits on the\n                architecture. This will be a plugin name if an external routing stage\n                plugin is being used.\n            translation_method (str): the pass to use for translating gates to\n                basis_gates. This will be a plugin name if an external translation stage\n                plugin is being used.\n            scheduling_method (str): the pass to use for scheduling instructions. This will\n                be a plugin name if an external scheduling stage plugin is being used.\n            instruction_durations (InstructionDurations): Dictionary of duration\n                (in dt) for each instruction.\n            backend_properties (BackendProperties): Properties returned by a\n                backend, including information on gate errors, readout errors,\n                qubit coherence times, etc.\n            approximation_degree (float): heuristic dial used for circuit approximation\n                (1.0=no approximation, 0.0=maximal approximation)\n            seed_transpiler (int): Sets random seed for the stochastic parts of\n                the transpiler.\n            timing_constraints (TimingConstraints): Hardware time alignment restrictions.\n            unitary_synthesis_method (str): The string method to use for the\n                :class:`~qiskit.transpiler.passes.UnitarySynthesis` pass. Will\n                search installed plugins for a valid method. You can see a list of\n                installed plugins with :func:`.unitary_synthesis_plugin_names`.\n            target (Target): The backend target\n            hls_config (HLSConfig): An optional configuration class to use for\n                :class:`~qiskit.transpiler.passes.HighLevelSynthesis` pass.\n                Specifies how to synthesize various high-level objects.\n            init_method (str): The plugin name for the init stage plugin to use\n            optimization_method (str): The plugin name for the optimization stage plugin\n                to use.\n        '
        self.initial_layout = initial_layout
        self.basis_gates = basis_gates
        self.inst_map = inst_map
        self.coupling_map = coupling_map
        self.init_method = init_method
        self.layout_method = layout_method
        self.routing_method = routing_method
        self.translation_method = translation_method
        self.optimization_method = optimization_method
        self.scheduling_method = scheduling_method
        self.instruction_durations = instruction_durations
        self.backend_properties = backend_properties
        self.approximation_degree = approximation_degree
        self.seed_transpiler = seed_transpiler
        self.timing_constraints = timing_constraints
        self.unitary_synthesis_method = unitary_synthesis_method
        self.unitary_synthesis_plugin_config = unitary_synthesis_plugin_config
        self.target = target
        self.hls_config = hls_config

    @classmethod
    def from_backend(cls, backend, _skip_target=False, **pass_manager_options):
        if False:
            return 10
        "Construct a configuration based on a backend and user input.\n\n        This method automatically gererates a PassManagerConfig object based on the backend's\n        features. User options can be used to overwrite the configuration.\n\n        Args:\n            backend (BackendV1): The backend that provides the configuration.\n            pass_manager_options: User-defined option-value pairs.\n\n        Returns:\n            PassManagerConfig: The configuration generated based on the arguments.\n\n        Raises:\n            AttributeError: If the backend does not support a `configuration()` method.\n        "
        res = cls(**pass_manager_options)
        backend_version = getattr(backend, 'version', 0)
        if not isinstance(backend_version, int):
            backend_version = 0
        if backend_version < 2:
            config = backend.configuration()
        if res.basis_gates is None:
            if backend_version < 2:
                res.basis_gates = getattr(config, 'basis_gates', None)
            else:
                res.basis_gates = backend.operation_names
        if res.inst_map is None:
            if backend_version < 2:
                if hasattr(backend, 'defaults'):
                    defaults = backend.defaults()
                    if defaults is not None:
                        res.inst_map = defaults.instruction_schedule_map
            else:
                res.inst_map = backend.instruction_schedule_map
        if res.coupling_map is None:
            if backend_version < 2:
                cmap_edge_list = getattr(config, 'coupling_map', None)
                if cmap_edge_list is not None:
                    res.coupling_map = CouplingMap(cmap_edge_list)
            else:
                res.coupling_map = backend.coupling_map
        if res.instruction_durations is None:
            if backend_version < 2:
                res.instruction_durations = InstructionDurations.from_backend(backend)
            else:
                res.instruction_durations = backend.instruction_durations
        if res.backend_properties is None and backend_version < 2:
            res.backend_properties = backend.properties()
        if res.target is None and (not _skip_target):
            if backend_version >= 2:
                res.target = backend.target
        if res.scheduling_method is None and hasattr(backend, 'get_scheduling_stage_plugin'):
            res.scheduling_method = backend.get_scheduling_stage_plugin()
        if res.translation_method is None and hasattr(backend, 'get_translation_stage_plugin'):
            res.translation_method = backend.get_translation_stage_plugin()
        return res

    def __str__(self):
        if False:
            while True:
                i = 10
        newline = '\n'
        newline_tab = '\n\t'
        if self.backend_properties is not None:
            backend_props = pprint.pformat(self.backend_properties.to_dict())
            backend_props = backend_props.replace(newline, newline_tab)
        else:
            backend_props = str(None)
        return f'Pass Manager Config:\n\tinitial_layout: {self.initial_layout}\n\tbasis_gates: {self.basis_gates}\n\tinst_map: {str(self.inst_map).replace(newline, newline_tab)}\n\tcoupling_map: {self.coupling_map}\n\tlayout_method: {self.layout_method}\n\trouting_method: {self.routing_method}\n\ttranslation_method: {self.translation_method}\n\tscheduling_method: {self.scheduling_method}\n\tinstruction_durations: {str(self.instruction_durations).replace(newline, newline_tab)}\n\tbackend_properties: {backend_props}\n\tapproximation_degree: {self.approximation_degree}\n\tseed_transpiler: {self.seed_transpiler}\n\ttiming_constraints: {self.timing_constraints}\n\tunitary_synthesis_method: {self.unitary_synthesis_method}\n\tunitary_synthesis_plugin_config: {self.unitary_synthesis_plugin_config}\n\ttarget: {str(self.target).replace(newline, newline_tab)}\n'