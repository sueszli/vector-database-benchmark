"""Synthesize UnitaryGates."""
from __future__ import annotations
from math import pi, inf, isclose
from typing import Any
from copy import deepcopy
from itertools import product
from functools import partial
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, Target
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.quantum_info.synthesis import one_qubit_decompose
from qiskit.quantum_info.synthesis.xx_decompose import XXDecomposer, XXEmbodiments
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitBasisDecomposer, TwoQubitWeylDecomposition
from qiskit.quantum_info import Operator
from qiskit.circuit import ControlFlowOp, Gate, Parameter
from qiskit.circuit.library.standard_gates import iSwapGate, CXGate, CZGate, RXXGate, RZXGate, ECRGate
from qiskit.transpiler.passes.synthesis import plugin
from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import Optimize1qGatesDecomposition
from qiskit.providers.models import BackendProperties
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
KAK_GATE_NAMES = {'cx': CXGate(), 'cz': CZGate(), 'iswap': iSwapGate(), 'rxx': RXXGate(pi / 2), 'ecr': ECRGate(), 'rzx': RZXGate(pi / 4)}
GateNameToGate = get_standard_gate_name_mapping()

def _choose_kak_gate(basis_gates):
    if False:
        while True:
            i = 10
    'Choose the first available 2q gate to use in the KAK decomposition.'
    kak_gate = None
    kak_gates = set(basis_gates or []).intersection(KAK_GATE_NAMES.keys())
    if kak_gates:
        kak_gate = KAK_GATE_NAMES[kak_gates.pop()]
    return kak_gate

def _choose_euler_basis(basis_gates):
    if False:
        while True:
            i = 10
    'Choose the first available 1q basis to use in the Euler decomposition.'
    basis_set = set(basis_gates or [])
    for (basis, gates) in one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES.items():
        if set(gates).issubset(basis_set):
            return basis
    return 'U'

def _find_matching_euler_bases(target, qubit):
    if False:
        for i in range(10):
            print('nop')
    'Find matching available 1q basis to use in the Euler decomposition.'
    euler_basis_gates = []
    basis_set = target.operation_names_for_qargs((qubit,))
    for (basis, gates) in one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES.items():
        if set(gates).issubset(basis_set):
            euler_basis_gates.append(basis)
    return euler_basis_gates

def _choose_bases(basis_gates, basis_dict=None):
    if False:
        print('Hello World!')
    'Find the matching basis string keys from the list of basis gates from the backend.'
    if basis_gates is None:
        basis_set = set()
    else:
        basis_set = set(basis_gates)
    if basis_dict is None:
        basis_dict = one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES
    out_basis = []
    for (basis, gates) in basis_dict.items():
        if set(gates).issubset(basis_set):
            out_basis.append(basis)
    return out_basis

def _decomposer_2q_from_basis_gates(basis_gates, pulse_optimize=None, approximation_degree=None):
    if False:
        for i in range(10):
            print('nop')
    decomposer2q = None
    kak_gate = _choose_kak_gate(basis_gates)
    euler_basis = _choose_euler_basis(basis_gates)
    basis_fidelity = approximation_degree or 1.0
    if isinstance(kak_gate, RZXGate):
        backup_optimizer = TwoQubitBasisDecomposer(CXGate(), basis_fidelity=basis_fidelity, euler_basis=euler_basis, pulse_optimize=pulse_optimize)
        decomposer2q = XXDecomposer(euler_basis=euler_basis, backup_optimizer=backup_optimizer)
    elif kak_gate is not None:
        decomposer2q = TwoQubitBasisDecomposer(kak_gate, basis_fidelity=basis_fidelity, euler_basis=euler_basis, pulse_optimize=pulse_optimize)
    return decomposer2q

def _error(circuit, target=None, qubits=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate a rough error for a `circuit` that runs on specific\n    `qubits` of `target`.\n\n    Use basis errors from target if available, otherwise use length\n    of circuit as a weak proxy for error.\n    '
    if target is None:
        return len(circuit)
    gate_fidelities = []
    gate_durations = []
    for inst in circuit:
        inst_qubits = tuple((qubits[circuit.find_bit(q).index] for q in inst.qubits))
        try:
            keys = target.operation_names_for_qargs(inst_qubits)
            for key in keys:
                target_op = target.operation_from_name(key)
                if isinstance(target_op, inst.operation.base_class) and (target_op.is_parameterized() or all((isclose(float(p1), float(p2)) for (p1, p2) in zip(target_op.params, inst.operation.params)))):
                    inst_props = target[key].get(inst_qubits, None)
                    if inst_props is not None:
                        error = getattr(inst_props, 'error', 0.0) or 0.0
                        duration = getattr(inst_props, 'duration', 0.0) or 0.0
                        gate_fidelities.append(1 - error)
                        gate_durations.append(duration)
                    else:
                        gate_fidelities.append(1.0)
                        gate_durations.append(0.0)
                    break
            else:
                raise KeyError
        except KeyError as error:
            raise TranspilerError(f'Encountered a bad synthesis. Target has no {inst.operation} on qubits {qubits}.') from error
    return 1 - np.prod(gate_fidelities)

def _preferred_direction(decomposer2q, qubits, natural_direction, coupling_map=None, gate_lengths=None, gate_errors=None):
    if False:
        i = 10
        return i + 15
    '\n    `decomposer2q` decomposes an SU(4) over `qubits`. A user sets `natural_direction`\n    to indicate whether they prefer synthesis in a hardware-native direction.\n    If yes, we return the `preferred_direction` here. If no hardware direction is\n    preferred, we raise an error (unless natural_direction is None).\n    We infer this from `coupling_map`, `gate_lengths`, `gate_errors`.\n\n    Returns [0, 1] if qubits are correct in the hardware-native direction.\n    Returns [1, 0] if qubits must be flipped to match hardware-native direction.\n    '
    qubits_tuple = tuple(qubits)
    reverse_tuple = qubits_tuple[::-1]
    preferred_direction = None
    if natural_direction in {None, True}:
        if coupling_map is not None:
            neighbors0 = coupling_map.neighbors(qubits[0])
            zero_one = qubits[1] in neighbors0
            neighbors1 = coupling_map.neighbors(qubits[1])
            one_zero = qubits[0] in neighbors1
            if zero_one and (not one_zero):
                preferred_direction = [0, 1]
            if one_zero and (not zero_one):
                preferred_direction = [1, 0]
        if preferred_direction is None and (gate_lengths or gate_errors):
            cost_0_1 = inf
            cost_1_0 = inf
            try:
                cost_0_1 = next((duration for (gate, duration) in gate_lengths.get(qubits_tuple, []) if gate == decomposer2q.gate))
            except StopIteration:
                pass
            try:
                cost_1_0 = next((duration for (gate, duration) in gate_lengths.get(reverse_tuple, []) if gate == decomposer2q.gate))
            except StopIteration:
                pass
            if not (cost_0_1 < inf or cost_1_0 < inf):
                try:
                    cost_0_1 = next((error for (gate, error) in gate_errors.get(qubits_tuple, []) if gate == decomposer2q.gate))
                except StopIteration:
                    pass
                try:
                    cost_1_0 = next((error for (gate, error) in gate_errors.get(reverse_tuple, []) if gate == decomposer2q.gate))
                except StopIteration:
                    pass
            if cost_0_1 < cost_1_0:
                preferred_direction = [0, 1]
            elif cost_1_0 < cost_0_1:
                preferred_direction = [1, 0]
    if natural_direction is True and preferred_direction is None:
        raise TranspilerError(f'No preferred direction of gate on qubits {qubits} could be determined from coupling map or gate lengths / gate errors.')
    return preferred_direction

class UnitarySynthesis(TransformationPass):
    """Synthesize gates according to their basis gates."""

    def __init__(self, basis_gates: list[str]=None, approximation_degree: float | None=1.0, coupling_map: CouplingMap=None, backend_props: BackendProperties=None, pulse_optimize: bool | None=None, natural_direction: bool | None=None, synth_gates: list[str] | None=None, method: str='default', min_qubits: int=None, plugin_config: dict=None, target: Target=None):
        if False:
            print('Hello World!')
        "Synthesize unitaries over some basis gates.\n\n        This pass can approximate 2-qubit unitaries given some\n        gate fidelities (either via ``backend_props`` or ``target``).\n        More approximation can be forced by setting a heuristic dial\n        ``approximation_degree``.\n\n        Args:\n            basis_gates (list[str]): List of gate names to target. If this is\n                not specified the ``target`` argument must be used. If both this\n                and the ``target`` are specified the value of ``target`` will\n                be used and this will be ignored.\n            approximation_degree (float): heuristic dial used for circuit approximation\n                (1.0=no approximation, 0.0=maximal approximation). Approximation can\n                make the synthesized circuit cheaper at the cost of straying from\n                the original unitary. If None, approximation is done based on gate fidelities.\n            coupling_map (CouplingMap): the coupling map of the backend\n                in case synthesis is done on a physical circuit. The\n                directionality of the coupling_map will be taken into\n                account if ``pulse_optimize`` is ``True``/``None`` and ``natural_direction``\n                is ``True``/``None``.\n            backend_props (BackendProperties): Properties of a backend to\n                synthesize for (e.g. gate fidelities).\n            pulse_optimize (bool): Whether to optimize pulses during\n                synthesis. A value of ``None`` will attempt it but fall\n                back if it does not succeed. A value of ``True`` will raise\n                an error if pulse-optimized synthesis does not succeed.\n            natural_direction (bool): Whether to apply synthesis considering\n                directionality of 2-qubit gates. Only applies when\n                ``pulse_optimize`` is ``True`` or ``None``. The natural direction is\n                determined by first checking to see whether the\n                coupling map is unidirectional.  If there is no\n                coupling map or the coupling map is bidirectional,\n                the gate direction with the shorter\n                duration from the backend properties will be used. If\n                set to True, and a natural direction can not be\n                determined, raises :class:`.TranspilerError`. If set to None, no\n                exception will be raised if a natural direction can\n                not be determined.\n            synth_gates (list[str]): List of gates to synthesize. If None and\n                ``pulse_optimize`` is False or None, default to\n                ``['unitary']``. If ``None`` and ``pulse_optimize == True``,\n                default to ``['unitary', 'swap']``\n            method (str): The unitary synthesis method plugin to use.\n            min_qubits: The minimum number of qubits in the unitary to synthesize. If this is set\n                and the unitary is less than the specified number of qubits it will not be\n                synthesized.\n            plugin_config: Optional extra configuration arguments (as a ``dict``)\n                which are passed directly to the specified unitary synthesis\n                plugin. By default, this will have no effect as the default\n                plugin has no extra arguments. Refer to the documentation of\n                your unitary synthesis plugin on how to use this.\n            target: The optional :class:`~.Target` for the target device the pass\n                is compiling for. If specified this will supersede the values\n                set for ``basis_gates``, ``coupling_map``, and ``backend_props``.\n        "
        super().__init__()
        self._basis_gates = set(basis_gates or ())
        self._approximation_degree = approximation_degree
        self._min_qubits = min_qubits
        self.method = method
        self.plugins = None
        if method != 'default':
            self.plugins = plugin.UnitarySynthesisPluginManager()
        self._coupling_map = coupling_map
        self._backend_props = backend_props
        self._pulse_optimize = pulse_optimize
        self._natural_direction = natural_direction
        self._plugin_config = plugin_config
        self._target = target
        if target is not None:
            self._coupling_map = self._target.build_coupling_map()
        if synth_gates:
            self._synth_gates = synth_gates
        elif pulse_optimize:
            self._synth_gates = ['unitary', 'swap']
        else:
            self._synth_gates = ['unitary']
        self._synth_gates = set(self._synth_gates) - self._basis_gates

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if False:
            while True:
                i = 10
        'Run the UnitarySynthesis pass on ``dag``.\n\n        Args:\n            dag: input dag.\n\n        Returns:\n            Output dag with UnitaryGates synthesized to target basis.\n\n        Raises:\n            TranspilerError: if ``method`` was specified for the class and is not\n                found in the installed plugins list. The list of installed\n                plugins can be queried with\n                :func:`~qiskit.transpiler.passes.synthesis.plugin.unitary_synthesis_plugin_names`\n        '
        if self.method != 'default' and self.method not in self.plugins.ext_plugins:
            raise TranspilerError('Specified method: %s not found in plugin list' % self.method)
        if not set(self._synth_gates).intersection(dag.count_ops()):
            return dag
        if self.plugins:
            plugin_method = self.plugins.ext_plugins[self.method].obj
        else:
            plugin_method = DefaultUnitarySynthesis()
        plugin_kwargs: dict[str, Any] = {'config': self._plugin_config}
        _gate_lengths = _gate_errors = None
        _gate_lengths_by_qubit = _gate_errors_by_qubit = None
        if self.method == 'default':
            default_method = plugin_method
            default_kwargs = plugin_kwargs
            method_list = [(plugin_method, plugin_kwargs)]
        else:
            default_method = self.plugins.ext_plugins['default'].obj
            default_kwargs = {}
            method_list = [(plugin_method, plugin_kwargs), (default_method, default_kwargs)]
        for (method, kwargs) in method_list:
            if method.supports_basis_gates:
                kwargs['basis_gates'] = self._basis_gates
            if method.supports_natural_direction:
                kwargs['natural_direction'] = self._natural_direction
            if method.supports_pulse_optimize:
                kwargs['pulse_optimize'] = self._pulse_optimize
            if method.supports_gate_lengths:
                _gate_lengths = _gate_lengths or _build_gate_lengths(self._backend_props, self._target)
                kwargs['gate_lengths'] = _gate_lengths
            if method.supports_gate_errors:
                _gate_errors = _gate_errors or _build_gate_errors(self._backend_props, self._target)
                kwargs['gate_errors'] = _gate_errors
            if method.supports_gate_lengths_by_qubit:
                _gate_lengths_by_qubit = _gate_lengths_by_qubit or _build_gate_lengths_by_qubit(self._backend_props, self._target)
                kwargs['gate_lengths_by_qubit'] = _gate_lengths_by_qubit
            if method.supports_gate_errors_by_qubit:
                _gate_errors_by_qubit = _gate_errors_by_qubit or _build_gate_errors_by_qubit(self._backend_props, self._target)
                kwargs['gate_errors_by_qubit'] = _gate_errors_by_qubit
            supported_bases = method.supported_bases
            if supported_bases is not None:
                kwargs['matched_basis'] = _choose_bases(self._basis_gates, supported_bases)
            if method.supports_target:
                kwargs['target'] = self._target
        default_method._approximation_degree = self._approximation_degree
        if self.method == 'default':
            plugin_method._approximation_degree = self._approximation_degree
        qubit_indices = {bit: i for (i, bit) in enumerate(dag.qubits)} if plugin_method.supports_coupling_map or default_method.supports_coupling_map else {}
        return self._run_main_loop(dag, qubit_indices, plugin_method, plugin_kwargs, default_method, default_kwargs)

    def _run_main_loop(self, dag, qubit_indices, plugin_method, plugin_kwargs, default_method, default_kwargs):
        if False:
            return 10
        'Inner loop for the optimizer, after all DAG-independent set-up has been completed.'
        for node in dag.op_nodes(ControlFlowOp):
            node.op = node.op.replace_blocks([dag_to_circuit(self._run_main_loop(circuit_to_dag(block), {inner: qubit_indices[outer] for (inner, outer) in zip(block.qubits, node.qargs)}, plugin_method, plugin_kwargs, default_method, default_kwargs), copy_operations=False) for block in node.op.blocks])
        for node in dag.named_nodes(*self._synth_gates):
            if self._min_qubits is not None and len(node.qargs) < self._min_qubits:
                continue
            synth_dag = None
            unitary = node.op.to_matrix()
            n_qubits = len(node.qargs)
            if plugin_method.max_qubits is not None and n_qubits > plugin_method.max_qubits or (plugin_method.min_qubits is not None and n_qubits < plugin_method.min_qubits):
                (method, kwargs) = (default_method, default_kwargs)
            else:
                (method, kwargs) = (plugin_method, plugin_kwargs)
            if method.supports_coupling_map:
                kwargs['coupling_map'] = (self._coupling_map, [qubit_indices[x] for x in node.qargs])
            synth_dag = method.run(unitary, **kwargs)
            if synth_dag is not None:
                dag.substitute_node_with_dag(node, synth_dag)
        return dag

def _build_gate_lengths(props=None, target=None):
    if False:
        for i in range(10):
            print('nop')
    'Builds a ``gate_lengths`` dictionary from either ``props`` (BackendV1)\n    or ``target`` (BackendV2).\n\n    The dictionary has the form:\n    {gate_name: {(qubits,): duration}}\n    '
    gate_lengths = {}
    if target is not None:
        for (gate, prop_dict) in target.items():
            gate_lengths[gate] = {}
            for (qubit, gate_props) in prop_dict.items():
                if gate_props is not None and gate_props.duration is not None:
                    gate_lengths[gate][qubit] = gate_props.duration
    elif props is not None:
        for gate in props._gates:
            gate_lengths[gate] = {}
            for (k, v) in props._gates[gate].items():
                length = v.get('gate_length')
                if length:
                    gate_lengths[gate][k] = length[0]
            if not gate_lengths[gate]:
                del gate_lengths[gate]
    return gate_lengths

def _build_gate_errors(props=None, target=None):
    if False:
        for i in range(10):
            print('nop')
    'Builds a ``gate_error`` dictionary from either ``props`` (BackendV1)\n    or ``target`` (BackendV2).\n\n    The dictionary has the form:\n    {gate_name: {(qubits,): error_rate}}\n    '
    gate_errors = {}
    if target is not None:
        for (gate, prop_dict) in target.items():
            gate_errors[gate] = {}
            for (qubit, gate_props) in prop_dict.items():
                if gate_props is not None and gate_props.error is not None:
                    gate_errors[gate][qubit] = gate_props.error
    if props is not None:
        for gate in props._gates:
            gate_errors[gate] = {}
            for (k, v) in props._gates[gate].items():
                error = v.get('gate_error')
                if error:
                    gate_errors[gate][k] = error[0]
            if not gate_errors[gate]:
                del gate_errors[gate]
    return gate_errors

def _build_gate_lengths_by_qubit(props=None, target=None):
    if False:
        print('Hello World!')
    '\n    Builds a `gate_lengths` dictionary from either `props` (BackendV1)\n    or `target (BackendV2)`.\n\n    The dictionary has the form:\n    {(qubits): [Gate, duration]}\n    '
    gate_lengths = {}
    if target is not None and target.qargs is not None:
        for qubits in target.qargs:
            names = target.operation_names_for_qargs(qubits)
            operation_and_durations = []
            for name in names:
                operation = target.operation_from_name(name)
                duration = getattr(target[name].get(qubits, None), 'duration', None)
                if duration:
                    operation_and_durations.append((operation, duration))
            if operation_and_durations:
                gate_lengths[qubits] = operation_and_durations
    elif props is not None:
        for (gate_name, gate_props) in props._gates.items():
            gate = GateNameToGate[gate_name]
            for (qubits, properties) in gate_props.items():
                duration = properties.get('gate_length', [0.0])[0]
                operation_and_durations = (gate, duration)
                if qubits in gate_lengths:
                    gate_lengths[qubits].append(operation_and_durations)
                else:
                    gate_lengths[qubits] = [operation_and_durations]
    return gate_lengths

def _build_gate_errors_by_qubit(props=None, target=None):
    if False:
        while True:
            i = 10
    '\n    Builds a `gate_error` dictionary from either `props` (BackendV1)\n    or `target (BackendV2)`.\n\n    The dictionary has the form:\n    {(qubits): [Gate, error]}\n    '
    gate_errors = {}
    if target is not None and target.qargs is not None:
        for qubits in target.qargs:
            names = target.operation_names_for_qargs(qubits)
            operation_and_errors = []
            for name in names:
                operation = target.operation_from_name(name)
                error = getattr(target[name].get(qubits, None), 'error', None)
                if error:
                    operation_and_errors.append((operation, error))
            if operation_and_errors:
                gate_errors[qubits] = operation_and_errors
    elif props is not None:
        for (gate_name, gate_props) in props._gates.items():
            gate = GateNameToGate[gate_name]
            for (qubits, properties) in gate_props.items():
                error = properties.get('gate_error', [0.0])[0]
                operation_and_errors = (gate, error)
                if qubits in gate_errors:
                    gate_errors[qubits].append(operation_and_errors)
                else:
                    gate_errors[qubits] = [operation_and_errors]
    return gate_errors

class DefaultUnitarySynthesis(plugin.UnitarySynthesisPlugin):
    """The default unitary synthesis plugin."""

    @property
    def supports_basis_gates(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    @property
    def supports_coupling_map(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    @property
    def supports_natural_direction(self):
        if False:
            print('Hello World!')
        return True

    @property
    def supports_pulse_optimize(self):
        if False:
            return 10
        return True

    @property
    def supports_gate_lengths(self):
        if False:
            return 10
        return False

    @property
    def supports_gate_errors(self):
        if False:
            while True:
                i = 10
        return False

    @property
    def supports_gate_lengths_by_qubit(self):
        if False:
            print('Hello World!')
        return True

    @property
    def supports_gate_errors_by_qubit(self):
        if False:
            return 10
        return True

    @property
    def max_qubits(self):
        if False:
            return 10
        return None

    @property
    def min_qubits(self):
        if False:
            i = 10
            return i + 15
        return None

    @property
    def supported_bases(self):
        if False:
            print('Hello World!')
        return None

    @property
    def supports_target(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self._decomposer_cache = {}

    def _decomposer_2q_from_target(self, target, qubits, approximation_degree):
        if False:
            i = 10
            return i + 15
        qubits_tuple = tuple(sorted(qubits))
        reverse_tuple = qubits_tuple[::-1]
        if qubits_tuple in self._decomposer_cache:
            return self._decomposer_cache[qubits_tuple]
        available_2q_basis = {}
        available_2q_props = {}

        def _replace_parameterized_gate(op):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(op, RXXGate) and isinstance(op.params[0], Parameter):
                op = RXXGate(pi / 2)
            elif isinstance(op, RZXGate) and isinstance(op.params[0], Parameter):
                op = RZXGate(pi / 4)
            return op
        try:
            keys = target.operation_names_for_qargs(qubits_tuple)
            for key in keys:
                op = target.operation_from_name(key)
                if not isinstance(op, Gate):
                    continue
                available_2q_basis[key] = _replace_parameterized_gate(op)
                available_2q_props[key] = target[key][qubits_tuple]
        except KeyError:
            pass
        try:
            keys = target.operation_names_for_qargs(reverse_tuple)
            for key in keys:
                if key not in available_2q_basis:
                    op = target.operation_from_name(key)
                    if not isinstance(op, Gate):
                        continue
                    available_2q_basis[key] = _replace_parameterized_gate(op)
                    available_2q_props[key] = target[key][reverse_tuple]
        except KeyError:
            pass
        if not available_2q_basis:
            raise TranspilerError(f'Target has no gates available on qubits {qubits} to synthesize over.')
        available_1q_basis = _find_matching_euler_bases(target, qubits_tuple[0])
        decomposers = []

        def is_supercontrolled(gate):
            if False:
                for i in range(10):
                    print('nop')
            try:
                operator = Operator(gate)
            except QiskitError:
                return False
            kak = TwoQubitWeylDecomposition(operator.data)
            return isclose(kak.a, pi / 4) and isclose(kak.c, 0.0)

        def is_controlled(gate):
            if False:
                while True:
                    i = 10
            try:
                operator = Operator(gate)
            except QiskitError:
                return False
            kak = TwoQubitWeylDecomposition(operator.data)
            return isclose(kak.b, 0.0) and isclose(kak.c, 0.0)
        supercontrolled_basis = {k: v for (k, v) in available_2q_basis.items() if is_supercontrolled(v)}
        for (basis_1q, basis_2q) in product(available_1q_basis, supercontrolled_basis.keys()):
            props = available_2q_props.get(basis_2q)
            if props is None:
                basis_2q_fidelity = 1.0
            else:
                error = getattr(props, 'error', 0.0)
                if error is None:
                    error = 0.0
                basis_2q_fidelity = 1 - error
            if approximation_degree is not None:
                basis_2q_fidelity *= approximation_degree
            decomposer = TwoQubitBasisDecomposer(supercontrolled_basis[basis_2q], euler_basis=basis_1q, basis_fidelity=basis_2q_fidelity)
            decomposers.append(decomposer)
        controlled_basis = {k: v for (k, v) in available_2q_basis.items() if is_controlled(v)}
        basis_2q_fidelity = {}
        embodiments = {}
        pi2_basis = None
        for (k, v) in controlled_basis.items():
            strength = 2 * TwoQubitWeylDecomposition(Operator(v).data).a
            props = available_2q_props.get(k)
            if props is None:
                basis_2q_fidelity[strength] = 1.0
            else:
                error = getattr(props, 'error', 0.0)
                if error is None:
                    error = 0.0
                basis_2q_fidelity[strength] = 1 - error
            embodiment = XXEmbodiments[v.base_class]
            if len(embodiment.parameters) == 1:
                embodiments[strength] = embodiment.assign_parameters([strength])
            else:
                embodiments[strength] = embodiment
            if isclose(strength, pi / 2) and k in supercontrolled_basis:
                pi2_basis = v
        if approximation_degree is not None:
            basis_2q_fidelity = {k: v * approximation_degree for (k, v) in basis_2q_fidelity.items()}
        if basis_2q_fidelity:
            for basis_1q in available_1q_basis:
                if isinstance(pi2_basis, CXGate) and basis_1q == 'ZSX':
                    pi2_decomposer = TwoQubitBasisDecomposer(pi2_basis, euler_basis=basis_1q, basis_fidelity=basis_2q_fidelity, pulse_optimize=True)
                    embodiments.update({pi / 2: XXEmbodiments[pi2_decomposer.gate.base_class]})
                else:
                    pi2_decomposer = None
                decomposer = XXDecomposer(basis_fidelity=basis_2q_fidelity, euler_basis=basis_1q, embodiments=embodiments, backup_optimizer=pi2_decomposer)
                decomposers.append(decomposer)
        self._decomposer_cache[qubits_tuple] = decomposers
        return decomposers

    def run(self, unitary, **options):
        if False:
            while True:
                i = 10
        approximation_degree = getattr(self, '_approximation_degree', 1.0)
        basis_gates = options['basis_gates']
        coupling_map = options['coupling_map'][0]
        natural_direction = options['natural_direction']
        pulse_optimize = options['pulse_optimize']
        gate_lengths = options['gate_lengths_by_qubit']
        gate_errors = options['gate_errors_by_qubit']
        qubits = options['coupling_map'][1]
        target = options['target']
        if unitary.shape == (2, 2):
            _decomposer1q = Optimize1qGatesDecomposition(basis_gates, target)
            sequence = _decomposer1q._resynthesize_run(unitary, qubits[0])
            if sequence is None:
                return None
            return _decomposer1q._gate_sequence_to_dag(sequence)
        elif unitary.shape == (4, 4):
            if target is not None:
                decomposers2q = self._decomposer_2q_from_target(target, qubits, approximation_degree)
            else:
                decomposer2q = _decomposer_2q_from_basis_gates(basis_gates, pulse_optimize, approximation_degree)
                decomposers2q = [decomposer2q] if decomposer2q is not None else []
            synth_circuits = []
            for decomposer2q in decomposers2q:
                preferred_direction = _preferred_direction(decomposer2q, qubits, natural_direction, coupling_map, gate_lengths, gate_errors)
                synth_circuit = self._synth_su4(unitary, decomposer2q, preferred_direction, approximation_degree)
                synth_circuits.append(synth_circuit)
            synth_circuit = min(synth_circuits, key=partial(_error, target=target, qubits=tuple(qubits)), default=None)
        else:
            from qiskit.quantum_info.synthesis.qsd import qs_decomposition
            synth_circuit = qs_decomposition(unitary) if basis_gates or target else None
        synth_dag = circuit_to_dag(synth_circuit) if synth_circuit is not None else None
        return synth_dag

    def _synth_su4(self, su4_mat, decomposer2q, preferred_direction, approximation_degree):
        if False:
            for i in range(10):
                print('nop')
        approximate = not approximation_degree == 1.0
        synth_circ = decomposer2q(su4_mat, approximate=approximate)
        synth_direction = None
        for inst in synth_circ:
            if inst.operation.num_qubits == 2:
                synth_direction = [synth_circ.find_bit(q).index for q in inst.qubits]
        if preferred_direction and synth_direction != preferred_direction:
            su4_mat_mm = deepcopy(su4_mat)
            su4_mat_mm[[1, 2]] = su4_mat_mm[[2, 1]]
            su4_mat_mm[:, [1, 2]] = su4_mat_mm[:, [2, 1]]
            synth_circ = decomposer2q(su4_mat_mm, approximate=approximate).reverse_bits()
        return synth_circ