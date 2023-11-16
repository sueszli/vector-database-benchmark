"""Synthesize higher-level objects and unroll custom definitions."""
from typing import Optional, Union, List, Tuple
from qiskit.circuit.operation import Operation
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import ControlFlowOp, ControlledGate, EquivalenceLibrary
from qiskit.transpiler.passes.utils import control_flow
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit.annotated_operation import AnnotatedOperation, InverseModifier, ControlModifier, PowerModifier
from qiskit.synthesis.clifford import synth_clifford_full, synth_clifford_layers, synth_clifford_depth_lnn, synth_clifford_greedy, synth_clifford_ag, synth_clifford_bm
from qiskit.synthesis.linear import synth_cnot_count_full_pmh, synth_cnot_depth_line_kms
from qiskit.synthesis.permutation import synth_permutation_basic, synth_permutation_acg, synth_permutation_depth_lnn_kms
from .plugin import HighLevelSynthesisPluginManager, HighLevelSynthesisPlugin

class HLSConfig:
    """The high-level-synthesis config allows to specify a list of "methods" used by
    :class:`~.HighLevelSynthesis` transformation pass to synthesize different types
    of higher-level-objects. A higher-level object is an object of type
    :class:`~.Operation` (e.g., "clifford", "linear_function", etc.), and the list
    of applicable synthesis methods is strictly tied to the name of the operation.
    In the config, each method is specified as a tuple consisting of the name of the
    synthesis algorithm and of a dictionary providing additional arguments for this
    algorithm. Additionally, a synthesis method can be specified as a tuple consisting
    of an instance of :class:`.HighLevelSynthesisPlugin` and additional arguments.
    Moreover, when there are no additional arguments, a synthesis
    method can be specified simply by name or by an instance
    of :class:`.HighLevelSynthesisPlugin`. The following example illustrates different
    ways how a config file can be created::

        from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
        from qiskit.transpiler.passes.synthesis.high_level_synthesis import ACGSynthesisPermutation

        # All the ways to specify hls_config are equivalent
        hls_config = HLSConfig(permutation=[("acg", {})])
        hls_config = HLSConfig(permutation=["acg"])
        hls_config = HLSConfig(permutation=[(ACGSynthesisPermutation(), {})])
        hls_config = HLSConfig(permutation=[ACGSynthesisPermutation()])

    The names of the synthesis algorithms should be declared in ``entry_points`` for
    ``qiskit.synthesis`` in ``setup.py``, in the form
    <higher-level-object-name>.<synthesis-method-name>.

    The standard higher-level-objects are recommended to have a synthesis method
    called "default", which would be called automatically when synthesizing these objects,
    without having to explicitly set these methods in the config.

    To avoid synthesizing a given higher-level-object, one can give it an empty list of methods.

    For an explicit example of using such config files, refer to the
    documentation for :class:`~.HighLevelSynthesis`.
    """

    def __init__(self, use_default_on_unspecified=True, **kwargs):
        if False:
            print('Hello World!')
        'Creates a high-level-synthesis config.\n\n        Args:\n            use_default_on_unspecified (bool): if True, every higher-level-object without an\n                explicitly specified list of methods will be synthesized using the "default"\n                algorithm if it exists.\n            kwargs: a dictionary mapping higher-level-objects to lists of synthesis methods.\n        '
        self.use_default_on_unspecified = use_default_on_unspecified
        self.methods = {}
        for (key, value) in kwargs.items():
            self.set_methods(key, value)

    def set_methods(self, hls_name, hls_methods):
        if False:
            i = 10
            return i + 15
        'Sets the list of synthesis methods for a given higher-level-object. This overwrites\n        the lists of methods if also set previously.'
        self.methods[hls_name] = hls_methods

class HighLevelSynthesis(TransformationPass):
    """Synthesize higher-level objects and unroll custom definitions.

    The input to this pass is a DAG that may contain higher-level objects,
    including abstract mathematical objects (e.g., objects of type :class:`.LinearFunction`),
    annotated operations (objects of type :class:`.AnnotatedOperation`), and
    custom gates.

    In the most common use-case when either ``basis_gates`` or ``target`` is specified,
    all higher-level objects are synthesized, so the output is a :class:`.DAGCircuit`
    without such objects.
    More precisely, every gate in the output DAG is either directly supported by the target,
    or is in ``equivalence_library``.

    The abstract mathematical objects are synthesized using synthesis plugins, applying
    synthesis methods specified in the high-level-synthesis config (refer to the documentation
    for :class:`~.HLSConfig`).

    As an example, let us assume that ``op_a`` and ``op_b`` are names of two higher-level objects,
    that ``op_a``-objects have two synthesis methods ``default`` which does require any additional
    parameters and ``other`` with two optional integer parameters ``option_1`` and ``option_2``,
    that ``op_b``-objects have a single synthesis method ``default``, and ``qc`` is a quantum
    circuit containing ``op_a`` and ``op_b`` objects. The following code snippet::

        hls_config = HLSConfig(op_b=[("other", {"option_1": 7, "option_2": 4})])
        pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
        transpiled_qc = pm.run(qc)

    shows how to run the alternative synthesis method ``other`` for ``op_b``-objects, while using the
    ``default`` methods for all other high-level objects, including ``op_a``-objects.

    The annotated operations (consisting of a base operation and a list of inverse, control and power
    modifiers) are synthesizing recursively, first synthesizing the base operation, and then applying
    synthesis methods for creating inverted, controlled, or powered versions of that).

    The custom gates are synthesized by recursively unrolling their definitions, until every gate
    is either supported by the target or is in the equivalence library.

    When neither ``basis_gates`` nor ``target`` is specified, the pass synthesizes only the top-level
    abstract mathematical objects and annotated operations, without descending into the gate
    ``definitions``. This is consistent with the older behavior of the pass, allowing to synthesize
    some higher-level objects using plugins and leaving the other gates untouched.
    """

    def __init__(self, hls_config: Optional[HLSConfig]=None, coupling_map: Optional[CouplingMap]=None, target: Optional[Target]=None, use_qubit_indices: bool=False, equivalence_library: Optional[EquivalenceLibrary]=None, basis_gates: Optional[List[str]]=None, min_qubits: int=0):
        if False:
            return 10
        "\n        HighLevelSynthesis initializer.\n\n        Args:\n            hls_config: Optional, the high-level-synthesis config that specifies synthesis methods\n                and parameters for various high-level-objects in the circuit. If it is not specified,\n                the default synthesis methods and parameters will be used.\n            coupling_map: Optional, directed graph represented as a coupling map.\n            target: Optional, the backend target to use for this pass. If it is specified,\n                it will be used instead of the coupling map.\n            use_qubit_indices: a flag indicating whether this synthesis pass is running before or after\n                the layout is set, that is, whether the qubit indices of higher-level-objects correspond\n                to qubit indices on the target backend.\n            equivalence_library: The equivalence library used (instructions in this library will not\n                be unrolled by this pass).\n            basis_gates: Optional, target basis names to unroll to, e.g. `['u3', 'cx']`.\n                Ignored if ``target`` is also specified.\n            min_qubits: The minimum number of qubits for operations in the input\n                dag to translate.\n        "
        super().__init__()
        if hls_config is not None:
            self.hls_config = hls_config
        else:
            self.hls_config = HLSConfig(True)
        self.hls_plugin_manager = HighLevelSynthesisPluginManager()
        self._coupling_map = coupling_map
        self._target = target
        self._use_qubit_indices = use_qubit_indices
        if target is not None:
            self._coupling_map = self._target.build_coupling_map()
        self._equiv_lib = equivalence_library
        self._basis_gates = basis_gates
        self._min_qubits = min_qubits
        self._top_level_only = self._basis_gates is None and self._target is None
        if not self._top_level_only and self._target is None:
            basic_insts = {'measure', 'reset', 'barrier', 'snapshot', 'delay'}
            self._device_insts = basic_insts | set(self._basis_gates)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if False:
            while True:
                i = 10
        'Run the HighLevelSynthesis pass on `dag`.\n\n        Args:\n            dag: input dag.\n\n        Returns:\n            Output dag with higher-level operations synthesized.\n\n        Raises:\n            TranspilerError: when the transpiler is unable to synthesize the given DAG\n            (for instance, when the specified synthesis method is not available).\n        '
        dag_op_nodes = dag.op_nodes()
        for node in dag_op_nodes:
            if isinstance(node.op, ControlFlowOp):
                node.op = control_flow.map_blocks(self.run, node.op)
                continue
            if getattr(node.op, '_directive', False):
                continue
            if dag.has_calibration_for(node) or len(node.qargs) < self._min_qubits:
                continue
            qubits = [dag.find_bit(x).index for x in node.qargs] if self._use_qubit_indices else None
            (decomposition, modified) = self._recursively_handle_op(node.op, qubits)
            if not modified:
                continue
            if isinstance(decomposition, QuantumCircuit):
                dag.substitute_node_with_dag(node, circuit_to_dag(decomposition, copy_operations=False))
            elif isinstance(decomposition, DAGCircuit):
                dag.substitute_node_with_dag(node, decomposition)
            elif isinstance(decomposition, Operation):
                dag.substitute_node(node, decomposition)
        return dag

    def _recursively_handle_op(self, op: Operation, qubits: Optional[List]=None) -> Tuple[Union[QuantumCircuit, DAGCircuit, Operation], bool]:
        if False:
            while True:
                i = 10
        'Recursively synthesizes a single operation.\n\n        Note: the reason that this function accepts an operation and not a dag node\n        is that it\'s also used for synthesizing the base operation for an annotated\n        gate (i.e. no dag node is available).\n\n        There are several possible results:\n\n        - The given operation is unchanged: e.g., it is supported by the target or is\n          in the equivalence library\n        - The result is a quantum circuit: e.g., synthesizing Clifford using plugin\n        - The result is a DAGCircuit: e.g., when unrolling custom gates\n        - The result is an Operation: e.g., adding control to CXGate results in CCXGate\n        - The given operation could not be synthesized, raising a transpiler error\n\n        The function returns the result of the synthesis (either a quantum circuit or\n        an Operation), and, as an optimization, a boolean indicating whether\n        synthesis did anything.\n\n        The function is recursive, for example synthesizing an annotated operation\n        involves synthesizing its "base operation" which might also be\n        an annotated operation.\n        '
        decomposition = self._synthesize_op_using_plugins(op, qubits)
        if decomposition:
            return (decomposition, True)
        decomposition = self._synthesize_annotated_op(op)
        if decomposition:
            return (decomposition, True)
        if self._top_level_only:
            return (op, False)
        controlled_gate_open_ctrl = isinstance(op, ControlledGate) and op._open_ctrl
        if not controlled_gate_open_ctrl:
            qargs = tuple(qubits) if qubits is not None else None
            inst_supported = self._target.instruction_supported(operation_name=op.name, qargs=qargs) if self._target is not None else op.name in self._device_insts
            if inst_supported or (self._equiv_lib is not None and self._equiv_lib.has_entry(op)):
                return (op, False)
        try:
            definition = op.definition
        except TypeError as err:
            raise TranspilerError(f'HighLevelSynthesis was unable to extract definition for {op.name}: {err}') from err
        except AttributeError:
            definition = None
        if definition is None:
            raise TranspilerError(f'HighLevelSynthesis was unable to synthesize {op}.')
        dag = circuit_to_dag(definition, copy_operations=False)
        dag = self.run(dag)
        return (dag, True)

    def _synthesize_op_using_plugins(self, op: Operation, qubits: List) -> Union[QuantumCircuit, None]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Attempts to synthesize op using plugin mechanism.\n        Returns either the synthesized circuit or None (which occurs when no\n        synthesis methods are available or specified).\n        '
        hls_plugin_manager = self.hls_plugin_manager
        if op.name in self.hls_config.methods.keys():
            methods = self.hls_config.methods[op.name]
        elif self.hls_config.use_default_on_unspecified and 'default' in hls_plugin_manager.method_names(op.name):
            methods = ['default']
        else:
            methods = []
        for method in methods:
            if isinstance(method, tuple):
                (plugin_specifier, plugin_args) = method
            else:
                plugin_specifier = method
                plugin_args = {}
            if isinstance(plugin_specifier, str):
                if plugin_specifier not in hls_plugin_manager.method_names(op.name):
                    raise TranspilerError('Specified method: %s not found in available plugins for %s' % (plugin_specifier, op.name))
                plugin_method = hls_plugin_manager.method(op.name, plugin_specifier)
            else:
                plugin_method = plugin_specifier
            decomposition = plugin_method.run(op, coupling_map=self._coupling_map, target=self._target, qubits=qubits, **plugin_args)
            if decomposition is not None:
                return decomposition
        return None

    def _synthesize_annotated_op(self, op: Operation) -> Union[Operation, None]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Recursively synthesizes annotated operations.\n        Returns either the synthesized operation or None (which occurs when the operation\n        is not an annotated operation).\n        '
        if isinstance(op, AnnotatedOperation):
            (synthesized_op, _) = self._recursively_handle_op(op.base_op, qubits=None)
            for modifier in op.modifiers:
                if isinstance(synthesized_op, DAGCircuit):
                    synthesized_op = dag_to_circuit(synthesized_op, copy_operations=False)
                if isinstance(modifier, InverseModifier):
                    synthesized_op = synthesized_op.inverse()
                elif isinstance(modifier, ControlModifier):
                    if isinstance(synthesized_op, QuantumCircuit):
                        synthesized_op = synthesized_op.to_gate()
                    synthesized_op = synthesized_op.control(num_ctrl_qubits=modifier.num_ctrl_qubits, label=None, ctrl_state=modifier.ctrl_state)
                    (synthesized_op, _) = self._recursively_handle_op(synthesized_op)
                elif isinstance(modifier, PowerModifier):
                    if isinstance(synthesized_op, QuantumCircuit):
                        qc = synthesized_op
                    else:
                        qc = QuantumCircuit(synthesized_op.num_qubits, synthesized_op.num_clbits)
                        qc.append(synthesized_op, range(synthesized_op.num_qubits), range(synthesized_op.num_clbits))
                    qc = qc.power(modifier.power)
                    synthesized_op = qc.to_gate()
                    (synthesized_op, _) = self._recursively_handle_op(synthesized_op)
                else:
                    raise TranspilerError(f'Unknown modifier {modifier}.')
            return synthesized_op
        return None

class DefaultSynthesisClifford(HighLevelSynthesisPlugin):
    """The default clifford synthesis plugin.

    For N <= 3 qubits this is the optimal CX cost decomposition by Bravyi, Maslov.
    For N > 3 qubits this is done using the general non-optimal greedy compilation
    routine from reference by Bravyi, Hu, Maslov, Shaydulin.

    This plugin name is :``clifford.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if False:
            return 10
        'Run synthesis for the given Clifford.'
        decomposition = synth_clifford_full(high_level_object)
        return decomposition

class AGSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Aaronson-Gottesman method.

    This plugin name is :``clifford.ag`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if False:
            i = 10
            return i + 15
        'Run synthesis for the given Clifford.'
        decomposition = synth_clifford_ag(high_level_object)
        return decomposition

class BMSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method.
    The plugin is named

    The method only works on Cliffords with at most 3 qubits, for which it
    constructs the optimal CX cost decomposition.

    This plugin name is :``clifford.bm`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if False:
            return 10
        'Run synthesis for the given Clifford.'
        if high_level_object.num_qubits <= 3:
            decomposition = synth_clifford_bm(high_level_object)
        else:
            decomposition = None
        return decomposition

class GreedySynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the greedy synthesis
    Bravyi-Hu-Maslov-Shaydulin method.

    This plugin name is :``clifford.greedy`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if False:
            i = 10
            return i + 15
        'Run synthesis for the given Clifford.'
        decomposition = synth_clifford_greedy(high_level_object)
        return decomposition

class LayerSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method
    to synthesize Cliffords into layers.

    This plugin name is :``clifford.layers`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if False:
            print('Hello World!')
        'Run synthesis for the given Clifford.'
        decomposition = synth_clifford_layers(high_level_object)
        return decomposition

class LayerLnnSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method
    to synthesize Cliffords into layers, with each layer synthesized
    adhering to LNN connectivity.

    This plugin name is :``clifford.lnn`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if False:
            print('Hello World!')
        'Run synthesis for the given Clifford.'
        decomposition = synth_clifford_depth_lnn(high_level_object)
        return decomposition

class DefaultSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """The default linear function synthesis plugin.

    This plugin name is :``linear_function.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if False:
            print('Hello World!')
        'Run synthesis for the given LinearFunction.'
        decomposition = synth_cnot_count_full_pmh(high_level_object.linear)
        return decomposition

class KMSSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """Linear function synthesis plugin based on the Kutin-Moulton-Smithline method.

    This plugin name is :``linear_function.kms`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if False:
            print('Hello World!')
        'Run synthesis for the given LinearFunction.'
        decomposition = synth_cnot_depth_line_kms(high_level_object.linear)
        return decomposition

class PMHSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """Linear function synthesis plugin based on the Patel-Markov-Hayes method.

    This plugin name is :``linear_function.pmh`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if False:
            return 10
        'Run synthesis for the given LinearFunction.'
        decomposition = synth_cnot_count_full_pmh(high_level_object.linear)
        return decomposition

class KMSSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the Kutin, Moulton, Smithline method.

    This plugin name is :``permutation.kms`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if False:
            for i in range(10):
                print('nop')
        'Run synthesis for the given Permutation.'
        decomposition = synth_permutation_depth_lnn_kms(high_level_object.pattern)
        return decomposition

class BasicSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on sorting.

    This plugin name is :``permutation.basic`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if False:
            for i in range(10):
                print('nop')
        'Run synthesis for the given Permutation.'
        decomposition = synth_permutation_basic(high_level_object.pattern)
        return decomposition

class ACGSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the Alon, Chung, Graham method.

    This plugin name is :``permutation.acg`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if False:
            while True:
                i = 10
        'Run synthesis for the given Permutation.'
        decomposition = synth_permutation_acg(high_level_object.pattern)
        return decomposition