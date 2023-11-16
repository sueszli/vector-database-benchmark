"""Implementation for AtomicFunction."""
import dataclasses
import traceback
import typing
from typing import Any, Dict, List, Optional, Sequence, Union
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_stack

@dataclasses.dataclass(frozen=True)
class CallOptions:
    """Specifies additional configuration for an AtomicFunction call."""
    collective_manager_ids_used: List[int] = dataclasses.field(default_factory=list)
    control_captures: List[Any] = dataclasses.field(default_factory=list)
    is_stateful: bool = False
RUNTIME_FUNCTION_REFS = {}

class AtomicFunction(core.AtomicFunction):
    """A Python callable for functions in the TF Runtime.

  Provides core functionality for tf.function including:
    - automatic lifecycle management of runtime functions
    - structured inputs (including captures) and structured outputs
    - calls from both eager and graph mode
    - dependency tracking of children functions
    - runtime error interpolation to identify user code stack traces
    - control dependencies (including automatic)
  """
    __slots__ = ['_name', '_bound_context', '_function_type', '_children', '_call_options', '_cached_definition', '_cached_graph', '_generated_graph']

    def __init__(self, name: Union[str, bytes], bound_context: context.Context, function_type: function_type_lib.FunctionType, children: Optional[List['AtomicFunction']]=None, call_options: CallOptions=CallOptions(), cached_graph: Optional[func_graph_module.FuncGraph]=None):
        if False:
            while True:
                i = 10
        'Construct a new AtomicFunction.\n\n    Args:\n      name: str/bytes name of the runtime function in the bound context.\n      bound_context: interface to the runtime for the AtomicFunction.\n      function_type: input/output contract for the AtomicFunction\n      children: list of AtomicFunctions that are needed to call this one.\n      call_options: extra configuration options for the call.\n      cached_graph: FuncGraph that this AtomicFunction was generated from (if\n        known). Otherwise it will lazily construct a new corresponding FuncGraph\n        if ever needed.\n    '
        self._name = compat.as_bytes(name)
        self._bound_context = bound_context
        self._function_type = function_type
        self._children = children if children else []
        self._call_options = call_options
        self._cached_definition = None
        self._cached_graph = cached_graph
        self._generated_graph = None
        ref_key = (self._bound_context.function_scope_id, self.name)
        if ref_key not in RUNTIME_FUNCTION_REFS:
            RUNTIME_FUNCTION_REFS[ref_key] = 1
        else:
            RUNTIME_FUNCTION_REFS[ref_key] += 1

    @property
    def name(self) -> bytes:
        if False:
            return 10
        'Name represented in UTF-8 encoded bytes.'
        return self._name

    @property
    def function_type(self) -> function_type_lib.FunctionType:
        if False:
            i = 10
            return i + 15
        'Represents the input/output contract of this function.'
        return self._function_type

    @property
    def children(self) -> List['AtomicFunction']:
        if False:
            i = 10
            return i + 15
        'AtomicFunctions needed as dependencies for this one.'
        return self._children

    @property
    def definition(self) -> function_pb2.FunctionDef:
        if False:
            print('Hello World!')
        'Current FunctionDef in the Runtime.'
        return self._bound_context.get_function_def(self.name)

    @property
    def attributes(self) -> Any:
        if False:
            return 10
        'Returns FunctionDef attributes in the Runtime.'
        attrs = self.definition.attr
        attrs.pop(attributes_lib.EAGER_RUNTIME_CONSTRUCTION_CONTEXT, None)
        return attrs

    @property
    def graph_debug_info(self) -> graph_debug_info_pb2.GraphDebugInfo:
        if False:
            while True:
                i = 10
        'A GraphDebugInfo proto mapping nodes to corresponding stack traces.'
        return self._bound_context.get_graph_debug_info(self.name)

    @property
    def call_options(self) -> CallOptions:
        if False:
            while True:
                i = 10
        'Call options declared for this AtomicFunction.'
        return self._call_options

    @property
    def graph_call_attrs(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Returns a dictionary of attributes needed to add a call in graph.'
        attrs = {'is_stateful': self.call_options.is_stateful, 'tout': [o.dtype.as_datatype_enum for o in self.function_type.flat_outputs], 'xla_compile_attr': self.cached_definition.attr.get(attributes_lib.XLA_COMPILE, None)}
        attrs.update(self._bound_context.function_call_options.as_attrs())
        return attrs

    @property
    def _c_func(self) -> Any:
        if False:
            print('Hello World!')
        'Returns a scoped pybind object containing FunctionRecord in runtime.'
        return self._bound_context.get_c_function(self.name)

    @property
    def cached_definition(self) -> function_pb2.FunctionDef:
        if False:
            return 10
        'Cached FunctionDef (not guaranteed to be fresh).'
        if self._cached_definition is None:
            self._cached_definition = self.definition
        return self._cached_definition

    @property
    def graph(self) -> func_graph_module.FuncGraph:
        if False:
            return 10
        'Returns a FuncGraph corresponding to the AtomicFunction.'
        if self._cached_graph:
            return self._cached_graph
        if not self._generated_graph:
            self._generated_graph = to_func_graph(self)
        return self._generated_graph

    def call_with_captures(self, args: Sequence[Any], kwargs: Dict[str, Any], captures: Sequence[Any]) -> Any:
        if False:
            for i in range(10):
                print('nop')
        'Calls with args, kwargs, captures and returns structured output.'
        bound_parameters = self.function_type.bind(*args, **kwargs)
        tensor_inputs = self.function_type.unpack_inputs(bound_parameters)
        capture_inputs = self.function_type.unpack_captures(captures)
        return self.call_preflattened(tensor_inputs + capture_inputs)

    def call_preflattened(self, args: Sequence[core.Tensor]) -> Any:
        if False:
            print('Hello World!')
        'Calls with flattened tensor inputs and returns the structured output.'
        flat_outputs = self.call_flat(*args)
        return self.function_type.pack_output(flat_outputs)

    def call_flat(self, *args: core.Tensor) -> Sequence[core.Tensor]:
        if False:
            print('Hello World!')
        'Calls with flat tensor inputs and returns flat tensor outputs.\n\n    Args:\n      *args: arguments to call this function with.\n\n    Returns:\n      The outputs of the function call.\n\n    Raises:\n      ValueError: if the number of arguments is incorrect.\n      FunctionAlreadyGarbageCollectedError: if the function is no longer\n        available to be called because it has been garbage collected.\n    '
        expected_len = len(self.cached_definition.signature.input_arg)
        if len(args) != expected_len:
            raise ValueError(f'Signature specifies {expected_len} arguments, got: {len(args)}. Expected inputs: {self.cached_definition.signature.input_arg}. Received inputs: {args}. Function Type: {self.function_type!r}')
        with InterpolateRuntimeError(self):
            with ops.control_dependencies(self._call_options.control_captures):
                with record.stop_recording():
                    if self._bound_context.executing_eagerly():
                        outputs = self._bound_context.call_function(self.name, list(args), len(self.function_type.flat_outputs))
                    else:
                        outputs = make_call_op_in_graph(self, list(args), self._bound_context.function_call_options.as_attrs())
        for (i, output_type) in enumerate(self.function_type.flat_outputs):
            handle_data = output_type.dtype._handle_data
            if handle_data:
                handle_data_util.set_handle_data(outputs[i], handle_data.shape_inference)
        if not self._bound_context.executing_eagerly():
            for (i, output_type) in enumerate(self.function_type.flat_outputs):
                outputs[i].set_shape(output_type.shape)
        return outputs

    def __call__(self, *args, **kwargs) -> Any:
        if False:
            print('Hello World!')
        if self.function_type.captures:
            raise ValueError('The FunctionType defines captured inputs. Use call_with_captures instead.')
        return self.call_with_captures(args, kwargs, [])

    def __del__(self):
        if False:
            while True:
                i = 10
        if self._generated_graph:
            func_graph_module.dismantle_func_graph(self._generated_graph)
        key = (self._bound_context.function_scope_id, self.name)
        RUNTIME_FUNCTION_REFS[key] -= 1
        if RUNTIME_FUNCTION_REFS[key] < 0:
            raise RuntimeError(f'AtomicFunction Refcounting for {self.name} is invalid.')
        if RUNTIME_FUNCTION_REFS[key] == 0:
            try:
                self._bound_context.remove_function(self.name)
                RUNTIME_FUNCTION_REFS.pop(key)
            except TypeError:
                pass
            except AttributeError:
                pass

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'<AtomicFunction> {compat.as_str(self.name)}{self.function_type}'

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'AtomicFunction(name={self.name},\nbound_context={self._bound_context},\nfunction_type={self.function_type!r},\nchildren={self._children!s},\ncall_options={self._call_options},\ncached_graph={self._cached_graph})'

def _set_read_only_resource_inputs_attr(op: ops.Operation, func_graph: func_graph_module.FuncGraph):
    if False:
        print('Hello World!')
    'Sets the list of resource inputs which are read-only.\n\n  This is used by AutomaticControlDependencies.\n\n  Args:\n    op: PartitionedCall Operation.\n    func_graph: FuncGraph.\n  '
    read_only_indices = acd.get_read_only_resource_input_indices_graph(func_graph)
    ops.set_int_list_attr(op, acd.READ_ONLY_RESOURCE_INPUTS_ATTR, read_only_indices)

def partitioned_call_op(name: str, args: Sequence[core.Tensor], is_stateful: bool, tout: Sequence[Any], config: Any=None, executor_type: Optional[str]=None, xla_compile_attr: Any=None) -> ops.Operation:
    if False:
        while True:
            i = 10
    'Generates a function call op respecting device annotations.\n\n  Args:\n    name: Name of the function to call.\n    args: The arguments of the function, including captured inputs.\n    is_stateful: If the function is stateful.\n    tout: a list containing the output dtypes enums\n    config: (Optional) A `tensorflow::ConfigProto` proto, serialized. If `None`,\n      all optimizations are disabled. Currently only handled for eager defined\n      functions.\n    executor_type: (Optional) A string for the name of the executor to be used\n      in the function call. If not set, or set to an empty string, the default\n      tensorflow executor will be used.\n    xla_compile_attr: (Optional) value of the XLA compilation attribute.\n\n  Returns:\n    Returns the operation.\n  '
    if config is None:
        config = function_utils.get_disabled_rewriter_config()
    if executor_type is None:
        executor_type = ''
    args = [ops.convert_to_tensor(x) for x in args]
    tin_attr = attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(type=[x.dtype.as_datatype_enum for x in args]))
    tout_attr = attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(type=tout))
    func_attr = attr_value_pb2.AttrValue(func=attr_value_pb2.NameAttrList(name=name))
    executor_type_attr = attr_value_pb2.AttrValue(s=compat.as_bytes(executor_type))
    config_proto = attr_value_pb2.AttrValue(s=config)
    op_name = 'StatefulPartitionedCall' if is_stateful else 'PartitionedCall'
    op_attrs = {'Tin': tin_attr, 'Tout': tout_attr, 'f': func_attr, 'config_proto': config_proto, 'executor_type': executor_type_attr}
    if xla_compile_attr is not None:
        op_attrs[attributes_lib.XLA_COMPILE] = xla_compile_attr
    op = ops.get_default_graph().create_op(op_name, args, tout, name=op_name, attrs=op_attrs)
    return op

def make_call_op_in_graph(atomic: AtomicFunction, tensor_inputs: Sequence[core.Tensor], context_call_attrs: Dict[str, Any]):
    if False:
        i = 10
        return i + 15
    'Adds an AtomicFunction to graph.'
    graph = ops.get_default_graph()
    graph._add_function_recursive(atomic)
    op = partitioned_call_op(name=atomic.name, args=tensor_inputs, is_stateful=atomic.call_options.is_stateful, tout=[o.dtype.as_datatype_enum for o in atomic.function_type.flat_outputs], config=context_call_attrs['config_proto'], executor_type=context_call_attrs['executor_type'], xla_compile_attr=atomic.cached_definition.attr.get(attributes_lib.XLA_COMPILE, None))
    _set_read_only_resource_inputs_attr(op, atomic.graph)
    ops.set_int_list_attr(op, acd.COLLECTIVE_MANAGER_IDS, atomic._call_options.collective_manager_ids_used)
    return op.outputs

def from_function_def(function_def: function_pb2.FunctionDef, function_type: function_type_lib.FunctionType) -> AtomicFunction:
    if False:
        return 10
    'Create a new AtomicFunction from FunctionDef + FunctionType.'
    bound_context = context.context()
    if bound_context.has_function(compat.as_bytes(function_def.signature.name)):
        raise ValueError('Function already registered in context.')
    bound_context.add_function_def(function_def)
    return AtomicFunction(function_def.signature.name, bound_context, function_type)

def from_func_graph(name: Union[str, bytes], graph: func_graph_module.FuncGraph, attrs: Dict[str, attr_value_pb2.AttrValue], function_type: Optional[function_type_lib.FunctionType]=None, overwrite: bool=False) -> AtomicFunction:
    if False:
        return 10
    'Initializes an AtomicFunction from FuncGraph.\n\n  Args:\n    name: str, the name for the created function.\n    graph: Graph, the graph containing the operations in the function\n    attrs: dict mapping names of attributes to their AttrValue values\n    function_type: known FunctionType to use, otherwise one is derived.\n    overwrite: overwrites function definition in the current context if needed\n\n  Returns:\n    An AtomicFunction instance.\n  '
    if attrs and attributes_lib.IMPLEMENTS in attrs:
        has_resource_vars = any((inp.dtype == dtypes.resource for inp in graph.inputs))
        captured_inputs = graph.external_captures + graph.deferred_external_captures
        assert not any((has_resource_vars, captured_inputs)), 'Function {name} has "{attr}={value}" attribute and thus can not depend on any tensors outside of its signature or modify variables. \n\nNote: variables are always captured and cause function re-tracing for every variable called.\n  inputs: {inputs}\n  captures: {captured}\n\nTo pass a variable to such function use  use variable.read_value().'.format(name=graph.name, attr=attributes_lib.IMPLEMENTS, value=attrs[attributes_lib.IMPLEMENTS], inputs=graph.inputs, captured=captured_inputs)
    input_ops = set((arg.op for arg in graph.inputs))
    operations = [op for op in graph.get_operations() if op not in input_ops]
    graph_output_names = graph._output_names
    if graph_output_names is not None and all((ops.tensor_id(t) in graph_output_names for t in graph.outputs)):
        output_names = [compat.as_bytes(graph_output_names[ops.tensor_id(t)]) for t in graph.outputs]
        if len(set(output_names)) != len(output_names):
            output_names = []
    else:
        output_names = []
    with graph._c_graph.get() as c_graph:
        fn = pywrap_tf_session.TF_GraphToFunction_wrapper(c_graph, compat.as_str(name), False, [o._c_op for o in operations], [t._as_tf_output() for t in graph.inputs], [t._as_tf_output() for t in graph.outputs], output_names, [o._c_op for o in graph.control_outputs], [], None, compat.as_str(''))
    attrs = attributes_lib.parse_func_attrs(attrs or {})
    for (attr_name, attr_value) in attrs.items():
        serialized = attr_value.SerializeToString()
        pywrap_tf_session.TF_FunctionSetAttrValueProto(fn, compat.as_str(attr_name), serialized)
    name = compat.as_bytes(name)
    bound_context = context.context()
    if overwrite and bound_context.has_function(name):
        bound_context.remove_function(name)
    bound_context.add_c_function(fn)
    pywrap_tf_session.TF_DeleteFunction(fn)
    call_options = CallOptions(collective_manager_ids_used=getattr(graph, 'collective_manager_ids_used', []), control_captures=graph.function_captures.control, is_stateful=any((op._is_stateful for op in operations)))
    if not function_type:
        function_type = function_type_utils.derive_from_graph(graph)
    return AtomicFunction(name, bound_context, function_type, list(graph._functions.values()), call_options, cached_graph=graph)

def to_func_graph(atomic: AtomicFunction) -> func_graph_module.FuncGraph:
    if False:
        for i in range(10):
            print('nop')
    'Generate a FuncGraph from an AtomicFunction.'
    (input_signature, output_signature) = function_type_lib.to_structured_signature(atomic.function_type)
    with ops.Graph().as_default():
        for f in atomic.children:
            ops.get_default_graph()._add_function(f)
        result = function_def_to_graph.function_def_to_graph(atomic.definition, structured_input_signature=input_signature, structured_outputs=output_signature, propagate_device_spec=True, include_library_functions=False)
        for f in atomic.children:
            result._add_function(f)
    for (i, input_type) in enumerate(atomic.function_type.flat_inputs):
        handle_data = input_type.dtype._handle_data
        if handle_data:
            handle_data_util.set_handle_data(result.inputs[i], handle_data.shape_inference)
        result.inputs[i].set_shape(input_type.shape)
    for (i, output_type) in enumerate(atomic.function_type.flat_outputs):
        handle_data = output_type.dtype._handle_data
        if handle_data:
            handle_data_util.set_handle_data(result.outputs[i], handle_data.shape_inference)
        result.outputs[i].set_shape(output_type.shape)
    result.collective_manager_ids_used = (atomic.call_options.collective_manager_ids_used,)
    return result

class InterpolateRuntimeError(object):
    """Context Manager that interpolates exceptions received by AtomicFunction."""
    DENY_LIST_PHRASES = ['<embedded']

    def __init__(self, top_level_func):
        if False:
            while True:
                i = 10
        self._func = top_level_func

    def interpolate(self, message, node_names, graph_debug_info):
        if False:
            return 10
        'Uses the GraphDebugInfo to generate an error message.'
        error_message = ['Graph execution error:', '']
        traces = tf_stack.LoadTracesFromDebugInfo(graph_debug_info)
        for node_name in node_names:
            error_message.append(f'Detected at node {node_name} defined at (most recent call last):')
            if node_name in traces:
                stack_trace = traces[node_name]
                for formatted_frame in traceback.format_list(stack_trace):
                    if not any((p in formatted_frame for p in self.DENY_LIST_PHRASES)):
                        error_message.append(formatted_frame)
            else:
                error_message.append('<stack traces unavailable>')
        error_message.append(message.strip())
        return '\n'.join(error_message)

    def __enter__(self):
        if False:
            return 10
        pass

    def __exit__(self, typ, exc, tb):
        if False:
            print('Hello World!')
        if not exc or not isinstance(exc, errors.OpError):
            return False
        exc = typing.cast(errors.OpError, exc)
        message = compat.as_text(exc.message)
        (parsed_message, func_tags, node_tags) = error_interpolation.parse_message(message)
        deepest_func = None
        for func_tag in func_tags:
            if func_tag.name == compat.as_str(self._func.name):
                deepest_func = self._func
            elif deepest_func:
                next_func = None
                for child_func in deepest_func.children:
                    if func_tag.name == compat.as_str(child_func.name):
                        next_func = child_func
                        break
                if next_func is not None and isinstance(next_func, AtomicFunction):
                    deepest_func = next_func
        if deepest_func:
            exc._message = self.interpolate(parsed_message, [t.name for t in node_tags], deepest_func.graph_debug_info)
        return False