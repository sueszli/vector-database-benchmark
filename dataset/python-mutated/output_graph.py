import collections
import contextlib
import copy
import functools
import itertools
import logging
import operator
import re
import sys
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union
import sympy
import torch._guards
import torch._logging
import torch.nn
import torch.utils._pytree as pytree
from torch import fx
from torch._guards import Checkpointable, GlobalContextCheckpointState, GuardsCheckpointState, Source, TracingContext
from torch._utils_internal import signpost_event
from torch.fx.experimental.symbolic_shapes import free_symbols, is_symbolic, ShapeEnv
from torch.utils.weak import WeakTensorKeyDictionary
from . import config, logging as torchdynamo_logging, variables
from .backends.registry import CompiledFn, CompilerFn
from .bytecode_transformation import create_call_function, create_instruction, Instruction, unique_id
from .code_context import code_context
from .codegen import PyCodegen
from .current_scope_id import enter_new_scope
from .exc import BackendCompilerFailed, exceptions_allowed_to_be_fallback, SkipFrame, unimplemented, unimplemented_with_warning
from .guards import GuardBuilder, install_guard
from .mutation_guard import is_dynamic_nn_module
from .side_effects import SideEffects
from .source import ConstantSource, GlobalStateSource, is_constant_source, is_from_local_source, LocalSource, ParamBufferSource, ShapeEnvSource, TensorProperty, TensorPropertySource
from .utils import checkpoint_params, CleanupHook, clone_inputs, count_calls, counters, dynamo_timed, get_instruction_source_311, get_static_address_type, graph_break_reasons, increment_op_count, lazy_format_graph_code, lazy_format_graph_tabular, LazyString, same
from .variables.base import VariableTracker
from .variables.builder import GraphArg, TrackedFake, VariableBuilder, wrap_fx_proxy
from .variables.nn_module import NNModuleVariable
from .variables.tensor import NumpyNdarrayVariable, SymNodeVariable, TensorVariable, UnspecializedPythonVariable
from .variables.torch_function import TensorWithTFOverrideVariable
log = logging.getLogger(__name__)
graph_tabular_log = torch._logging.getArtifactLogger(__name__, 'graph')
graph_code_log = torch._logging.getArtifactLogger(__name__, 'graph_code')
graph_sizes_log = torch._logging.getArtifactLogger(__name__, 'graph_sizes')
trace_call_log = torch._logging.getArtifactLogger(__name__, 'trace_call')

class OutputGraphState(NamedTuple):
    input_source_to_var: Dict[Source, VariableTracker]
    tracked_fakes: List[TrackedFake]
    guard_state: GuardsCheckpointState
    nn_modules: Optional[Dict[str, torch.nn.Module]]
    register_finalizer_fns: List[Callable[[fx.GraphModule], None]]
    global_state: Optional[Dict[str, bool]]
    param_name_to_source: Optional[Dict[str, Source]]
    side_effects: SideEffects
    timestamp: int
    non_compliant_ops: Set[torch._ops.OpOverload]

    def diff(self, other: 'OutputGraphState', *, prefix: str='') -> Optional[str]:
        if False:
            print('Hello World!')
        for k in self._fields:
            if k == 'guard_state':
                r = self.guard_state.diff(other.guard_state)
                if r is not None:
                    return r
                continue
            elif k == 'side_effects':
                r = self.side_effects.diff(other.side_effects)
                if r is not None:
                    return r
                continue
            sv = getattr(self, k)
            ov = getattr(other, k)
            if sv != ov:
                return f'{prefix}{k} mismatch: {sv} != {ov}'
        return None

    @property
    def guards(self):
        if False:
            i = 10
            return i + 15
        return self.guard_state.dynamo_guards

@functools.lru_cache(None)
def _step_logger():
    if False:
        print('Hello World!')
    return torchdynamo_logging.get_step_logger(log)

@dataclass
class GraphCompileReason:
    """Stores why a given output graph was compiled; i.e. what caused the graph break."""
    reason: str
    user_stack: List[traceback.FrameSummary]
    graph_break: bool = True

    def __post_init__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.graph_break:
            graph_break_reasons.append(self)

def _get_gen_rand_values_fn(random_calls):
    if False:
        while True:
            i = 10

    def _gen_rand_values():
        if False:
            i = 10
            return i + 15
        return [fn(*args, **kwargs) for (fn, args, kwargs) in random_calls]
    return _gen_rand_values

class FakeRootModule(torch.nn.Module):
    """Trick the constructor of fx.GraphModule"""

    def __init__(self, nn_modules: Dict[str, torch.nn.Module]):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        for (k, v) in nn_modules.items():
            setattr(self, k, v)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'FakeRootModule(...)'

class WrapperBackend:

    def __init__(self, backend: CompilerFn):
        if False:
            i = 10
            return i + 15
        self.backend: CompilerFn = backend

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        if False:
            while True:
                i = 10
        self.restore = checkpoint_params(gm)
        self.gm = gm
        copy_gm = copy.deepcopy(self.gm)
        self.candidate = self.backend(copy_gm, example_inputs)
        if self.candidate is None or self.candidate is self.gm.forward:
            return self.gm.forward
        if not config.verify_correctness:
            return self.candidate
        try:
            correct = self.gm.forward(*clone_inputs(example_inputs))
            result = self.candidate(*clone_inputs(example_inputs))
            if same(correct, result):
                return self.candidate
            raise RuntimeError(f'incorrect results of backend {self}')
            return self.gm.forward
        except Exception:
            log.exception('error in verify_correctness')
            raise
        finally:
            self.restore()
Scope = Dict[str, object]

class OutputGraph(Checkpointable[OutputGraphState]):
    """
    Wrapper class to hold outputs of InstructionTranslator.  Mainly the
    generated fx.Graph.

    OutputGraph is 1:1 with a frame being processed. Each frame is associated
    with some root InstructionTranslator. When user code calls a function,
    we construct a InliningInstructionTranslator that continues to write into
    the root InstructionTranslator's OutputGraph.
    """

    def __init__(self, code_options: Dict[str, Any], compiler_fn: Optional[CompilerFn], root_tx, export: bool, export_constraints, frame_state, local_scope: Scope, global_scope: Scope, f_code):
        if False:
            return 10
        super().__init__()
        self.tracers = [SubgraphTracer(self, export_root=export)]
        self.input_source_to_var: Dict[Source, VariableTracker] = {}
        self.export = export
        self.export_constraints = export_constraints
        self.frame_state = frame_state
        self.tensor_weakref_to_sizes_strides = WeakTensorKeyDictionary()
        self.cleanup_hooks: List[Callable[[], Any]] = []
        self.co_fields = {'co_name': f_code.co_name, 'co_filename': f_code.co_filename, 'co_firstlineno': f_code.co_firstlineno}
        self.tracked_fakes: List[TrackedFake] = []
        shape_env = ShapeEnv(tracked_fakes=self.tracked_fakes, allow_scalar_outputs=config.capture_scalar_outputs, allow_dynamic_output_shape_ops=config.capture_dynamic_output_shape_ops, co_fields=self.co_fields)
        fake_mode = torch._subclasses.FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True if self.export else False)
        self.tracing_context: TracingContext = TracingContext(fake_mode)
        self.init_ambient_guards()
        self.tracked_fakes_id_to_source: Dict[int, List[Source]] = collections.defaultdict(list)
        self.param_name_to_source: Optional[Dict[str, Source]] = dict()
        self.side_effects = SideEffects()
        self.code_options = dict(code_options)
        self.output_instructions: List[Instruction] = []
        self.timestamp = 0
        self.register_finalizer_fns: List[Callable[[fx.GraphModule], None]] = []
        self.compiler_fn: Optional[CompilerFn] = compiler_fn
        self.global_scope = global_scope
        self.local_scope = local_scope
        self.root_tx = root_tx
        from torch._dynamo.symbolic_convert import InstructionTranslatorBase
        self.source_to_user_stacks: Dict[Source, List[traceback.StackSummary]] = {}
        self._current_tx: List[InstructionTranslatorBase] = []
        self.cleanups: List[CleanupHook] = []
        self.should_exit = False
        self.random_values_var = None
        self.unspec_variable_map: Dict[str, UnspecializedPythonVariable] = {}
        self.torch_function_enabled = torch._C._is_torch_function_enabled()
        self.has_user_defined_allowed_in_graph = False
        self.non_compliant_ops: Set[torch._ops.OpOverload] = set({})
        self.save_global_state()

    def init_ambient_guards(self):
        if False:
            i = 10
            return i + 15
        self.guards.add(ShapeEnvSource().make_guard(GuardBuilder.SHAPE_ENV))
        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.DETERMINISTIC_ALGORITHMS))
        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.GRAD_MODE))
        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.DEFAULT_DEVICE))
        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.TORCH_FUNCTION_STATE))
        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.BACKEND_MATCH))

    def add_cleanup_hook(self, fn: Callable[[], Any]):
        if False:
            return 10
        self.cleanup_hooks.append(fn)

    def call_cleanup_hooks(self):
        if False:
            i = 10
            return i + 15
        for hook in reversed(self.cleanup_hooks):
            hook()
        self.cleanup_hooks.clear()

    @property
    def root_tracer(self):
        if False:
            print('Hello World!')
        return self.tracers[0]

    @property
    def current_tracer(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tracers[-1]

    def is_root_tracer(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.tracers) == 1

    @property
    def graph(self):
        if False:
            for i in range(10):
                print('nop')
        return self.current_tracer.graph

    @graph.setter
    def graph(self, value):
        if False:
            while True:
                i = 10
        self.current_tracer.graph = value

    @property
    def input_name_to_proxy(self):
        if False:
            while True:
                i = 10
        return self.current_tracer.input_name_to_proxy

    @property
    def real_value_cache(self):
        if False:
            return 10
        return self.current_tracer.real_value_cache

    def create_proxy(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self.current_tracer.create_proxy(*args, **kwargs)

    def create_node(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.current_tracer.create_node(*args, **kwargs)

    def remove_node(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.current_tracer.remove_node(*args, **kwargs)

    @contextlib.contextmanager
    def subtracer(self, source_target, prior_tracer):
        if False:
            print('Hello World!')
        new_scope_ctx = enter_new_scope()
        try:
            if prior_tracer:
                assert prior_tracer.parent is self.current_tracer
            new_scope_ctx.__enter__()
            tracer = prior_tracer if prior_tracer else SubgraphTracer(self, parent=self.current_tracer, source_target=source_target)
            self.tracers.append(tracer)
            yield tracer
        finally:
            new_scope_ctx.__exit__(None, None, None)
            self.tracers.pop()

    @property
    def output(self):
        if False:
            print('Hello World!')
        return self

    @property
    def fake_mode(self):
        if False:
            for i in range(10):
                print('nop')
        return self.root_tx.fake_mode

    @property
    def shape_env(self):
        if False:
            i = 10
            return i + 15
        return self.tracing_context.fake_mode.shape_env

    @property
    def guards(self) -> torch._guards.GuardsSet:
        if False:
            return 10
        return self.tracing_context.guards_context.dynamo_guards

    @property
    def nn_modules(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        return self.tracing_context.module_context.nn_modules

    def save_global_state(self, out=None):
        if False:
            i = 10
            return i + 15
        "\n        Saves to out if it is provided. Else saves to the tracing context's global_state.\n        "
        global_state = out if out is not None else self.tracing_context.global_context.global_state
        global_state['torch_function_enabled'] = (self.set_torch_function_state, self.torch_function_enabled)
        global_state['grad_enabled'] = (torch.set_grad_enabled, torch.is_grad_enabled())
        global_state['autocast_enabled'] = (torch.set_autocast_enabled, torch.is_autocast_enabled())
        global_state['autocast_cpu_enabled'] = (torch.set_autocast_cpu_enabled, torch.is_autocast_cpu_enabled())
        global_state['autocast_gpu_dtype'] = (torch.set_autocast_gpu_dtype, torch.get_autocast_gpu_dtype())
        global_state['autocast_cpu_dtype'] = (torch.set_autocast_cpu_dtype, torch.get_autocast_cpu_dtype())
        global_state['autocast_cache_enabled'] = (torch.set_autocast_cache_enabled, torch.is_autocast_cache_enabled())

    def push_tx(self, tx):
        if False:
            while True:
                i = 10
        self._current_tx.append(tx)

    def pop_tx(self):
        if False:
            for i in range(10):
                print('nop')
        return self._current_tx.pop()

    @property
    def current_tx(self):
        if False:
            for i in range(10):
                print('nop')
        return self.root_tx if not self._current_tx else self._current_tx[-1]

    def copy_graphstate(self) -> OutputGraphState:
        if False:
            return 10
        'Create a checkpoint of the current state by copying everything'
        assert self.param_name_to_source is not None
        guards_graph_state = self.tracing_context.guards_context.copy_graphstate()
        module_state = self.tracing_context.module_context.copy_graphstate()
        global_state = self.tracing_context.global_context.copy_graphstate()
        state = OutputGraphState(dict(self.input_source_to_var), list(self.tracked_fakes), guards_graph_state, module_state, list(self.register_finalizer_fns), global_state, dict(self.param_name_to_source), self.side_effects.clone(), self.timestamp, set(self.non_compliant_ops))
        self.timestamp += 1
        return state

    def restore_graphstate(self, state: OutputGraphState):
        if False:
            print('Hello World!')
        'Restore a checkpoint created by self.copy_graphstate()'
        (self.input_source_to_var, self.tracked_fakes, guards_state, module_state, self.register_finalizer_fns, global_state, self.param_name_to_source, self.side_effects, self.timestamp, self.non_compliant_ops) = state
        self.tracing_context.guards_context.restore_graphstate(guards_state)
        self.tracing_context.module_context.restore_graphstate(module_state)
        self.tracing_context.global_context.restore_graphstate(global_state)
        removed_nodes = 0
        for node in reversed(list(self.graph.nodes)):
            if node.meta['creation_timestamp'] > self.timestamp and node.op != 'placeholder':
                if 'example_value' in node.meta:
                    del node.meta['example_value']
                self.remove_node(node)
                self.real_value_cache.pop(node, None)
                removed_nodes += 1
        log.debug('restore_graphstate: removed %s nodes', removed_nodes)

    def add_symbol_bindings(self, arg: GraphArg):
        if False:
            while True:
                i = 10
        if self.export:
            return
        assert arg.fake_tensor is not None

        def bind_symint(s, prop):
            if False:
                for i in range(10):
                    print('nop')
            if not (is_symbolic(s) and isinstance(s.node.expr, sympy.Symbol)):
                return
            proxy = self.root_tracer.create_graph_input(str(s.node.expr), torch.SymInt, before=True, source=prop(arg.source))
            proxy.node.meta['grapharg'] = GraphArg(prop(arg.source), s, is_unspecialized=False, fake_tensor=None, is_tensor=False)
        for (i, s) in enumerate(arg.fake_tensor.size()):
            bind_symint(s, lambda src: TensorPropertySource(src, TensorProperty.SIZE, i))
        for (i, s) in enumerate(arg.fake_tensor.stride()):
            bind_symint(s, lambda src: TensorPropertySource(src, TensorProperty.STRIDE, i))
        bind_symint(arg.fake_tensor.storage_offset(), lambda src: TensorPropertySource(src, TensorProperty.STORAGE_OFFSET))

    def count_calls(self):
        if False:
            print('Hello World!')
        return count_calls(self.graph)

    def is_empty_graph(self):
        if False:
            for i in range(10):
                print('nop')
        return len(list(self.graph.nodes)) == 0

    def get_submodule(self, keys):
        if False:
            while True:
                i = 10
        assert keys
        obj: Union[torch.nn.Module, Dict[str, torch.nn.Module]] = self.nn_modules
        for k in keys.split('.'):
            if isinstance(obj, dict):
                obj = obj[k]
            else:
                obj = getattr(obj, k)
        return obj

    def new_var(self, name='tmp'):
        if False:
            print('Hello World!')
        existing = set(self.code_options['co_varnames'])
        for i in itertools.count():
            var = f'{name}_{i}'
            if var not in existing:
                self.code_options['co_varnames'] += (var,)
                return var

    def update_co_names(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Ensure self.code_options.co_names contains name'
        if name not in self.code_options['co_names']:
            self.code_options['co_names'] += (name,)

    @staticmethod
    def module_key_name(*names):
        if False:
            return 10
        name = '_'.join(map(str, names))
        name = re.sub("^[GL]\\['?(.*?)'?\\]$", '\\1', name)
        name = re.sub('\\[(\\d+)\\]', '_\\g<1>', name)
        name = re.sub('[^a-zA-Z0-9]', '_', name)
        if not name or not name[0].isalpha():
            name = 'sub' + name
        return name

    def register_attr_or_module(self, target: Union[torch.nn.Module, torch.Tensor, Any], *names, **options):
        if False:
            for i in range(10):
                print('nop')
        if is_dynamic_nn_module(target):
            return variables.UnspecializedNNModuleVariable(target, **options)
        options = dict(options)
        assert 'source' in options
        source = options['source']
        assert not isinstance(source, ParamBufferSource)
        if isinstance(target, torch.Tensor):
            tracer = self.current_tracer
            if not self.is_root_tracer():
                tracer = self.root_tracer
            if not is_constant_source(source):
                install_guard(source.make_guard(GuardBuilder.TENSOR_MATCH))
            if get_static_address_type(target) == 'guarded':
                install_guard(source.make_guard(GuardBuilder.DATA_PTR_MATCH))

            def wrap_name(module_key):
                if False:
                    while True:
                        i = 10
                assert self.param_name_to_source is not None
                self.param_name_to_source[module_key] = source
                return wrap_fx_proxy(self.root_tx, tracer.create_proxy('get_attr', module_key, tuple(), {}), example_value=target, **options)
        elif isinstance(target, torch.nn.Module):
            assert isinstance(target, torch.nn.Module)
            install_guard(source.make_guard(GuardBuilder.NN_MODULE))

            def wrap_name(module_key):
                if False:
                    while True:
                        i = 10
                return NNModuleVariable(type(target), module_key, **options)
        elif isinstance(target, (torch.SymInt, torch.SymFloat)):

            def wrap_name(module_key):
                if False:
                    return 10
                return SymNodeVariable.create(self, self.create_proxy('get_attr', module_key, tuple(), {}), sym_num=target, **options)
        else:

            def wrap_name(module_key):
                if False:
                    return 10
                self.output.update_co_names(module_key)
                self.global_scope[module_key] = target
                return VariableBuilder(self, ConstantSource(source_name=module_key))(target)
        for (k, v) in self.nn_modules.items():
            if v is target:
                return wrap_name(k)
        name = OutputGraph.module_key_name(*names)
        base = name
        for i in itertools.count():
            if name not in self.nn_modules:
                self.nn_modules[name] = target
                if isinstance(target, torch.nn.Module):

                    def register_leaf_name(leaf_name):
                        if False:
                            i = 10
                            return i + 15
                        assert self.param_name_to_source is not None
                        new_source = ParamBufferSource(source, leaf_name)
                        new_name = f'{name}.{leaf_name}'
                        self.param_name_to_source[new_name] = new_source
                    if hasattr(target, '_parameters'):
                        for (leaf_name, _) in target.named_parameters():
                            register_leaf_name(leaf_name)
                    if hasattr(target, '_buffers'):
                        for (leaf_name, _) in target.named_buffers():
                            register_leaf_name(leaf_name)
                return wrap_name(name)
            name = f'{base}_{i}'
        raise AssertionError('unreachable')

    def compile_subgraph(self, tx, partial_convert=False, reason: Optional[GraphCompileReason]=None):
        if False:
            i = 10
            return i + 15
        '\n        Generate a subgraph to continue execution on user code.\n        Automatically restore live variables.\n        '
        assert reason is not None
        from .decorators import disable
        self.partial_convert = partial_convert
        self.compile_subgraph_reason = reason
        log.debug('COMPILING GRAPH due to %s', reason)
        if not all((block.can_restore() for block in tx.block_stack)):
            unimplemented('compile_subgraph with block_depth != 0')
        prefix_insts: List[Instruction] = []
        if sys.version_info >= (3, 11):
            for inst in tx.prefix_insts:
                if inst.opname == 'MAKE_CELL':
                    prefix_insts.append(create_instruction('MAKE_CELL', argval=inst.argval))
                elif inst.opname == 'COPY_FREE_VARS':
                    prefix_insts.append(create_instruction('COPY_FREE_VARS', arg=len(tx.code_options['co_freevars'])))
                else:
                    prefix_insts.append(copy.copy(inst))

        def append_prefix_insts():
            if False:
                while True:
                    i = 10
            self.add_output_instructions(prefix_insts)
            prefix_insts.clear()
        for block in reversed(tx.block_stack):
            block.exit(tx)
        self.cleanup_graph()
        tx.prune_dead_locals()
        stack_values = list(tx.stack)
        root = FakeRootModule(self.nn_modules)
        restore_vars = []
        val_to_names: Dict[VariableTracker, List[str]] = {}
        if stack_values:
            val_to_names[stack_values[-1]] = list()
        for (k, v) in tx.symbolic_locals.items():
            if isinstance(v.source, LocalSource) and v.source.local_name == k:
                continue
            if v not in val_to_names:
                val_to_names[v] = list()
            val_to_names[v].append(k)
        for v in val_to_names.keys():
            restore_vars.extend(val_to_names[v])
            stack_values.extend([v] * len(val_to_names[v]))
        if len(tx.random_calls) > 0:
            append_prefix_insts()
            random_calls_instructions = []
            self.random_values_var = self.new_var('random_values')
            rand_fn_name = unique_id('__gen_rand_values')
            rand_fn = disable(_get_gen_rand_values_fn(tx.random_calls))
            self.install_global(rand_fn_name, rand_fn)
            codegen = PyCodegen(tx, root)
            random_calls_instructions.extend(codegen.load_function_name(rand_fn_name, True))
            random_calls_instructions.extend(create_call_function(0, False))
            random_calls_instructions.append(codegen.create_store(tx.output.random_values_var))
            self.add_output_instructions(random_calls_instructions)
        if stack_values and all((not isinstance(v, (UnspecializedPythonVariable, NumpyNdarrayVariable, TensorWithTFOverrideVariable)) for v in stack_values)) and all((isinstance(x, TensorVariable) for x in stack_values)) and (len(set(stack_values)) == len(stack_values)) and self.side_effects.is_empty():
            append_prefix_insts()
            self.add_output_instructions(self.compile_and_call_fx_graph(tx, list(reversed(stack_values)), root) + [create_instruction('UNPACK_SEQUENCE', arg=len(stack_values))])
        else:
            graph_output_var = self.new_var('graph_out')
            pass1 = PyCodegen(tx, root, graph_output_var)
            self.side_effects.codegen_hooks(pass1)
            self.side_effects.codegen_save_tempvars(pass1)
            pass1.foreach(stack_values)
            self.side_effects.codegen_update_mutated(pass1)
            pass2 = PyCodegen(tx, root, graph_output_var, tempvars={val: None for (val, count) in pass1.uses.items() if count > 1})
            self.side_effects.codegen_hooks(pass2)
            self.side_effects.codegen_save_tempvars(pass2)
            pass2.foreach(stack_values)
            self.side_effects.codegen_update_mutated(pass2)
            output = []
            if count_calls(self.graph) != 0 or len(pass2.graph_outputs) != 0:
                output.extend(self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root))
                if len(pass2.graph_outputs) != 0:
                    output.append(pass2.create_store(graph_output_var))
                else:
                    output.append(create_instruction('POP_TOP'))
            append_prefix_insts()
            self.add_output_instructions(output + pass2.get_instructions())
        self.add_output_instructions([PyCodegen(tx).create_store(var) for var in reversed(restore_vars)])

    def cleanup_graph(self):
        if False:
            return 10
        '\n        Remove this pattern from the graph:\n            torch._C._set_grad_enabled(False)\n            torch._C._set_grad_enabled(True)\n        '
        nodes = list(self.graph.nodes)
        grad_enabled = torch.is_grad_enabled()
        for (node1, node2) in zip(nodes, nodes[1:]):
            if node1.target is torch._C._set_grad_enabled and tuple(node1.args) == (not grad_enabled,) and (not node1._erased):
                grad_enabled = node1.args[0]
                if node2.target is torch._C._set_grad_enabled and tuple(node2.args) == (not grad_enabled,) and (not node2._erased):
                    grad_enabled = node2.args[0]
                    self.graph.erase_node(node1)
                    self.graph.erase_node(node2)

    def get_graph_sizes_log_str(self, name):
        if False:
            i = 10
            return i + 15
        graph_sizes_str = 'TRACED GRAPH TENSOR SIZES\n'
        graph_sizes_str += f'===== {name} =====\n'
        for node in self.graph.nodes:
            example_value = node.meta.get('example_value', None)
            if isinstance(example_value, torch._subclasses.FakeTensor):
                size = example_value.size()
                graph_sizes_str += f'{node.name}: {tuple(size)}\n'
                concrete_size = []
                has_symint = False
                for sz in size:
                    if isinstance(sz, int):
                        concrete_size.append(sz)
                    elif isinstance(sz, torch.SymInt):
                        has_symint = True
                        concrete_size.append(sz.node.hint)
                    else:
                        break
                else:
                    if has_symint:
                        graph_sizes_str += f'{node.name} (concrete): {tuple(concrete_size)}\n'
        return graph_sizes_str

    @contextlib.contextmanager
    def restore_global_state(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Momentarily restores the global state to what it was prior to tracing the current output\n        '
        prior_global_state = self.tracing_context.global_context.copy_graphstate()
        current_global_state: Dict[str, Tuple[Any, bool]] = {}
        self.save_global_state(out=current_global_state)
        try:
            self.tracing_context.global_context.restore_graphstate(prior_global_state)
            yield
        finally:
            self.tracing_context.global_context.restore_graphstate(GlobalContextCheckpointState(current_global_state))

    @torch._guards.TracingContext.clear_frame()
    def compile_and_call_fx_graph(self, tx, rv, root):
        if False:
            while True:
                i = 10
        '\n        Generate code from self.graph and return the Instruction()s to\n        call that generated code.\n        '
        from .decorators import disable
        assert isinstance(rv, list)
        assert isinstance(root, FakeRootModule)
        self.create_node('output', 'output', (self.current_tracer.create_arg(tuple((x.as_proxy() for x in rv))),), {})
        self.remove_unused_graphargs()
        ncalls = count_calls(self.graph)
        counters['stats']['calls_captured'] += ncalls
        self.real_value_cache.clear()
        gm = fx.GraphModule(root, self.graph)
        for register_finalizer in self.register_finalizer_fns:
            register_finalizer(gm)
        gm.compile_subgraph_reason = self.compile_subgraph_reason
        name = unique_id('__compiled_fn')
        graph_code_log.debug('%s', lazy_format_graph_code(name, gm))
        graph_tabular_log.debug('%s', lazy_format_graph_tabular(name, gm))
        graph_sizes_log.debug('%s', LazyString(lambda : self.get_graph_sizes_log_str(name)))
        self.call_cleanup_hooks()
        with self.restore_global_state():
            compiled_fn = self.call_user_compiler(gm)
        compiled_fn = disable(compiled_fn)
        counters['stats']['unique_graphs'] += 1
        self.install_global(name, compiled_fn)
        cg = PyCodegen(tx)
        cg.make_call_generated_code(name)
        return cg.get_instructions()

    @property
    def placeholders(self) -> List[fx.Node]:
        if False:
            print('Hello World!')
        r = []
        for node in self.graph.nodes:
            if node.op == 'placeholder':
                r.append(node)
                continue
            break
        return r

    @property
    def graphargs(self) -> List[GraphArg]:
        if False:
            i = 10
            return i + 15
        return [node.meta['grapharg'] for node in self.placeholders]

    @dynamo_timed(phase_name='backend_compile')
    def call_user_compiler(self, gm: fx.GraphModule) -> CompiledFn:
        if False:
            print('Hello World!')
        assert self.compiler_fn is not None
        tot = 0
        placeholders = []
        for node in gm.graph.nodes:
            if node.op in ('call_function', 'call_method', 'call_module'):
                tot += 1
            if node.op == 'placeholder':
                placeholders.append(node)
        increment_op_count(tot)
        for pl in placeholders:
            arg = pl.meta['grapharg']
            pl._dynamo_source = arg.source
        gm._param_name_to_source = self.param_name_to_source
        gm._source_to_user_stacks = self.source_to_user_stacks
        try:
            name = self.compiler_fn.__name__ if hasattr(self.compiler_fn, '__name__') else ''
            _step_logger()(logging.INFO, f'calling compiler function {name}')
            compiler_fn = self.compiler_fn
            if config.verify_correctness:
                compiler_fn = WrapperBackend(compiler_fn)
            compiled_fn = compiler_fn(gm, self.example_inputs())
            _step_logger()(logging.INFO, f'done compiler function {name}')
            assert callable(compiled_fn), 'compiler_fn did not return callable'
        except exceptions_allowed_to_be_fallback as e:
            if self.has_user_defined_allowed_in_graph:
                raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(e.__traceback__) from None
            msg = f'Backend compiler failed with a fake tensor exception at \n{self.root_tx.format_frame_summary()}Adding a graph break.'
            unimplemented_with_warning(e, self.root_tx.f_code, msg)
        except SkipFrame as e:
            raise e
        except Exception as e:
            raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(e.__traceback__) from None
        signpost_event('dynamo', 'OutputGraph.call_user_compiler', {**self.co_fields, 'op_count': tot, 'node_count': len(gm.graph.nodes), 'input_count': len(placeholders)})
        return compiled_fn

    def example_inputs(self) -> List[torch.Tensor]:
        if False:
            i = 10
            return i + 15
        result = []
        for arg in self.graphargs:
            result.append(arg.example)
        return result

    def remove_unused_graphargs(self) -> None:
        if False:
            i = 10
            return i + 15
        for node in reversed(list(self.graph.nodes)):
            if len(list(node.users)) == 0:
                if node.op == 'get_attr':
                    self.remove_node(node)
                elif node.op == 'call_function' and node.target is operator.getitem:
                    self.remove_node(node)

        def placeholder_binds_symbol(node):
            if False:
                return 10
            arg = node.meta['grapharg']
            example = arg.example
            if isinstance(example, torch.SymInt) and isinstance(example.node.expr, sympy.Symbol):
                return example.node.expr
            return None

        def remove_unused(node):
            if False:
                while True:
                    i = 10
            log.debug('REMOVE UNUSED GRAPHARG %s', node.meta['grapharg'].source.name())
            del node.meta['grapharg']
            self.remove_node(node)
            self.real_value_cache.pop(node, None)
        used_symbols = set()
        recheck_placeholders = []
        for node in self.placeholders:
            binds_symbol = placeholder_binds_symbol(node) is not None
            if binds_symbol:
                if not node.users:
                    recheck_placeholders.append(node)
            elif not node.users:
                remove_unused(node)
            else:
                arg = node.meta['grapharg']
                fake = arg.fake_tensor if arg.fake_tensor is not None else arg.example
                used_symbols |= free_symbols(fake)
        for node in recheck_placeholders:
            symbol = placeholder_binds_symbol(node)
            if symbol is not None:
                if symbol not in used_symbols:
                    remove_unused(node)
                else:
                    used_symbols.remove(symbol)

    def add_output_instructions(self, prefix: List[Instruction]) -> None:
        if False:
            return 10
        '\n        We call this on the creation of a new compiled subgraph that is inserted\n        before user code.\n        '
        self.output_instructions.extend(prefix)
        self.should_exit = True

    def install_global(self, name, value) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.cleanups.append(CleanupHook.create(self.global_scope, name, value))

    def cleanup(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.root_tx = None
        self.nn_modules.clear()
        self.param_name_to_source = None
        for node in self.graph.nodes:
            if 'grapharg' in node.meta:
                del node.meta['grapharg']
        self.real_value_cache.clear()
        self.input_name_to_proxy.clear()
        self.side_effects.clear()
        self.register_finalizer_fns.clear()

    def set_torch_function_state(self, enabled: bool) -> None:
        if False:
            while True:
                i = 10
        self.torch_function_enabled = enabled

    def add_graph_finalizer(self, register_finalizer: Callable[[fx.GraphModule], None]) -> None:
        if False:
            i = 10
            return i + 15
        self.register_finalizer_fns.append(register_finalizer)
err_epilogue = "With the current config, we will graph break (and fall back to eager-mode PyTorch) on all ops that have do not have the 'pt2_compliant_tag'. Please see the following doc for how to mark this op as PT2 compliant https://docs.google.com/document/d/1W--T6wz8IY8fOI0Vm8BF44PdBgs283QvpelJZWieQWQ"

def check_pt2_compliant_op(output_graph, kind, target, args, kwargs):
    if False:
        for i in range(10):
            print('nop')
    if kind != 'call_function':
        return

    def encountered_non_compliant_op(target, msg):
        if False:
            while True:
                i = 10
        output_graph.non_compliant_ops.add(target)
        if config.only_allow_pt2_compliant_ops:
            unimplemented(msg + ' ' + err_epilogue)
    if isinstance(target, torch._ops.OpOverload):
        if torch.Tag.pt2_compliant_tag in target.tags:
            return
        encountered_non_compliant_op(target, f'Encountered the torch.ops.OpOverload {target} that is not PT2 compliant.')
        return
    if isinstance(target, torch._ops.OpOverloadPacket):
        overloads = tuple(target.overloads())
        if len(overloads) == 1:
            op = getattr(target, overloads[0])
            if torch.Tag.pt2_compliant_tag in op.tags:
                return
            encountered_non_compliant_op(op, f'Encountered the non-overloaded torch.ops.OpOverloadPacket {target} that is not PT2 compliant. ')
            return
        (args, kwargs) = torch._dynamo.utils.get_fake_values_from_nodes(output_graph.current_tx, (args, kwargs))
        try:
            overload = torch._C._jit_resolve_packet(target._qualified_op_name, *args, **kwargs)
        except RuntimeError as e:
            unimplemented(str(e))
        op = getattr(target, overload)
        if torch.Tag.pt2_compliant_tag not in op.tags:
            encountered_non_compliant_op(op, f'Encountered the torch.ops.OpOverloadPacket {target} which resolves to the overload ({overload}) that is not PT2 compliant.')

class SubgraphTracer(fx.Tracer):
    """
    Holds an FX graph that is being traced. OutputGraph owns a SubgraphTracer
    and the separation of responsibilities is that SubgraphTracer is
    responsible for building the graph while OutputGraph is responsible for
    compiling and executing the graph.
    """

    def __init__(self, output_graph, parent=None, export_root=False, source_target=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.output_graph = weakref.proxy(output_graph)
        self.graph = torch.fx.Graph()
        if export_root:
            assert parent is None
        self.export_root = export_root
        self.input_name_to_proxy: Dict[str, fx.Proxy] = {}
        self.real_value_cache: Dict[fx.Node, torch.Tensor] = {}
        self.parent = parent
        self.lifted_freevars = {}
        self.prev_inst = None
        self._cur_code = None
        self._orig_gm_meta = None
        self._orig_gm_lineno_map = None
        self._orig_gm_firstlineno = None
        if self.parent is None:
            self.source_fn_stack = []
        else:
            self.source_fn_stack = self.parent.source_fn_stack + [(self.graph._target_to_str(source_target), source_target)]

    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None):
        if False:
            print('Hello World!')
        if self.parent is not None:
            (flat_args, tree_spec) = pytree.tree_flatten((args, kwargs))
            new_flat_args = []
            for arg in flat_args:
                maybe_new_arg = self.maybe_lift_tracked_freevar_to_input(arg)
                new_flat_args.append(maybe_new_arg)
            (args, kwargs) = pytree.tree_unflatten(new_flat_args, tree_spec)
        rv = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)
        tx = self.output_graph.current_tx
        if sys.version_info >= (3, 11) and kind in ('call_function', 'call_method', 'call_module'):
            cur_inst = tx.current_instruction
            if cur_inst is not self.prev_inst and cur_inst.positions.lineno is not None:
                tx_code = tx.f_code
                header = tx.get_line_of_code_header(lineno=cur_inst.positions.lineno)

                def get_trace_call_log_str():
                    if False:
                        print('Hello World!')
                    line = get_instruction_source_311(tx_code, cur_inst).rstrip()
                    return f'TRACE FX call {rv.node.name} from {header}\n{line}'
                trace_call_log.debug('%s', LazyString(get_trace_call_log_str))
                self.prev_inst = cur_inst
        is_retracing = False
        if tx.f_code is not self._cur_code:
            orig_graphmodule_maybe = code_context.get_context(tx.f_code).get('orig_graphmodule', None)
            if isinstance(orig_graphmodule_maybe, torch.fx.GraphModule):
                is_retracing = True
                self._orig_gm_meta = [nd.meta for nd in orig_graphmodule_maybe.graph.nodes]
                self._orig_gm_lineno_map = orig_graphmodule_maybe._lineno_map
                self._orig_gm_firstlineno = orig_graphmodule_maybe.forward.__code__.co_firstlineno
            else:
                self._orig_gm_meta = None
                self._orig_gm_lineno_map = None
                self._orig_gm_firstlineno = None
        nn_module_stack = tx.nn_module_stack
        if nn_module_stack:
            rv.node.meta['nn_module_stack'] = nn_module_stack.copy()
        if kind in {'call_function', 'call_method'}:
            rv.node.meta['source_fn_stack'] = self.source_fn_stack + [(rv.node.name, target)]
        elif kind == 'call_module':
            if self.parent is not None:
                unimplemented('Invoking an nn.Module inside HigherOrderOperator')
            rv.node.meta['source_fn_stack'] = self.source_fn_stack + [(rv.node.name, rv.node.meta['nn_module_stack'][target][1])]
        if self._orig_gm_meta and self._orig_gm_lineno_map and self._orig_gm_firstlineno:
            lineno = tx.current_instruction.starts_line
            node_idx = None
            if lineno is not None:
                node_idx = self._orig_gm_lineno_map.get(lineno - self._orig_gm_firstlineno, None)
            if node_idx is not None:
                meta = self._orig_gm_meta[node_idx]
                for field in fx.proxy._COPY_META_FIELDS:
                    if field in meta:
                        rv.node.meta[field] = meta[field]
                if 'stack_trace' in meta:
                    rv.node.meta['stack_trace'] = meta['stack_trace']
        if not is_retracing:
            if 'nn_module_stack' not in rv.node.meta:
                nn_module_stack = tx.nn_module_stack
                if nn_module_stack:
                    rv.node.meta['nn_module_stack'] = nn_module_stack.copy()
            if 'source_fn_stack' not in rv.node.meta:
                if kind in {'call_function', 'call_method'}:
                    rv.node.meta['source_fn_stack'] = self.source_fn_stack + [(rv.node.name, target)]
                elif kind == 'call_module':
                    if self.parent is not None:
                        unimplemented('Invoking an nn.Module inside HigherOrderOperator')
                    rv.node.meta['source_fn_stack'] = self.source_fn_stack + [(rv.node.name, rv.node.meta['nn_module_stack'][target][1])]
        if 'stack_trace' not in rv.node.meta:
            frame_summaries: List[traceback.FrameSummary] = []
            while tx:
                frame_summaries.append(tx.frame_summary())
                tx = getattr(tx, 'parent', None)
            frame_summaries.reverse()
            msgs = traceback.StackSummary.from_list(frame_summaries).format()
            rv.node.stack_trace = ''.join(msgs)
        return rv

    def create_node(self, op, target, args=None, kwargs=None, name=None, type_expr=None):
        if False:
            print('Hello World!')
        check_pt2_compliant_op(self.output_graph, op, target, args, kwargs)
        if self.parent is not None:
            flat_args = pytree.arg_tree_leaves(*args, **kwargs)
            for arg in flat_args:
                if not isinstance(arg, torch.fx.Node):
                    continue
                assert arg.graph == self.graph, 'create_node using arg not from this SubgraphTracer'
        node = super().create_node(op, target, args, kwargs, name, type_expr)
        node.meta['creation_timestamp'] = self.output_graph.timestamp
        return node

    def remove_node(self, node):
        if False:
            print('Hello World!')
        if len(node.users) > 0:
            user_graph_nodes: List[torch.fx.Node] = []
            for user in node.users.keys():
                if user.graph != self.graph:
                    user_graph_nodes.extend(reversed(list(user.graph.nodes)))
            for other_graph_node in user_graph_nodes:
                other_graph_node.graph.erase_node(other_graph_node)
        self.graph.erase_node(node)
        self.input_name_to_proxy.pop(node.name, None)

    def create_graph_input(self, name, type_expr=None, before=False, source=None):
        if False:
            for i in range(10):
                print('nop')
        log.debug('create_graph_input %s %s', name, source.name() if source is not None else '(none)')
        if source is None:
            assert self.parent is not None, 'you are required to provide a source for inputs on the root tracer'
        if self.export_root:
            if not is_from_local_source(source, allow_cell_or_freevar=False):
                self.output_graph.source_to_user_stacks.setdefault(source, []).append(TracingContext.extract_stack())
        if name in self.input_name_to_proxy:
            for i in itertools.count():
                candidate_name = f'{name}_{i}'
                if candidate_name not in self.input_name_to_proxy:
                    name = candidate_name
                    break
        if self.input_name_to_proxy:
            prev_name = next(reversed(self.input_name_to_proxy))
            node = self.input_name_to_proxy[prev_name].node
            if before:
                ctx = self.graph.inserting_before(node)
            else:
                ctx = self.graph.inserting_after(node)
        else:
            ctx = self.graph.inserting_before(None)
        with ctx:
            proxy = self.create_proxy('placeholder', name, (), {}, type_expr=type_expr)
            if self.input_name_to_proxy and before:
                (k, v) = self.input_name_to_proxy.popitem()
                self.input_name_to_proxy[name] = proxy
                self.input_name_to_proxy[k] = v
            else:
                self.input_name_to_proxy[name] = proxy
            return proxy

    def lift_tracked_freevar_to_input(self, proxy):
        if False:
            return 10
        assert self.parent is not None, 'lift_tracked_freevar_to_input should not be called on root SubgraphTracer'
        if proxy in self.lifted_freevars:
            return self.lifted_freevars[proxy]
        new_proxy = self.create_graph_input(proxy.node.name)
        new_proxy.node.meta['example_value'] = proxy.node.meta['example_value']
        self.lifted_freevars[proxy] = new_proxy
        if self.parent is not None and proxy.tracer != self.parent:
            self.parent.lift_tracked_freevar_to_input(proxy)
        return new_proxy

    def maybe_lift_tracked_freevar_to_input(self, arg):
        if False:
            print('Hello World!')
        '\n        If arg is a free variable, then lift it to be an input.\n        Returns the new lifted arg (if arg was a freevar), else the\n        original arg.\n        '
        if not isinstance(arg, torch.fx.Proxy):
            return arg
        elif arg.tracer == self:
            return arg
        return self.lift_tracked_freevar_to_input(arg)