"""Tools for deserializing `Function`s."""
import collections
import pprint
import re
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_lib
from tensorflow.python.framework import function_def_to_graph as function_def_lib
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

def _is_tensor(t):
    if False:
        while True:
            i = 10
    return isinstance(t, (tensor.Tensor, resource_variable_ops.BaseResourceVariable))

def _call_concrete_function(function, inputs):
    if False:
        print('Hello World!')
    'Calls a restored Function with structured inputs.\n\n  This differs from `function.__call__` in that inputs and outputs are\n  structured and that it casts inputs to tensors if needed.\n\n  Note: this does not checks that non-tensor inputs match. That should be\n  done before via `_concrete_function_callable_with`.\n\n  Args:\n    function: ConcreteFunction to call.\n    inputs: Structured inputs compatible with\n      `function.graph.structured_input_signature`.\n\n  Returns:\n    The structured function output.\n  '
    expected_structure = function.graph.structured_input_signature
    flatten_inputs = nest.flatten_up_to(expected_structure, inputs, expand_composites=True)
    flatten_expected = nest.flatten(expected_structure, expand_composites=True)
    tensor_inputs = []
    for (arg, expected) in zip(flatten_inputs, flatten_expected):
        if isinstance(expected, tensor.TensorSpec):
            tensor_inputs.append(ops.convert_to_tensor(arg, dtype_hint=expected.dtype))
        elif isinstance(expected, resource_variable_ops.VariableSpec):
            tensor_inputs.append(arg.handle)
    result = function._call_flat(tensor_inputs, function.captured_inputs)
    if isinstance(result, ops.Operation):
        return None
    return result

def _try_convert_to_tensor_spec(arg, dtype_hint):
    if False:
        return 10
    'Returns None or TensorSpec obtained if `arg` is converted to tensor.'
    try:
        with func_graph_lib.FuncGraph(name='guess_conversion').as_default():
            result = ops.convert_to_tensor(arg, dtype_hint=dtype_hint)
            return tensor.TensorSpec(shape=result.shape, dtype=result.dtype)
    except (TypeError, ValueError):
        return None

def _concrete_function_callable_with(function, inputs, allow_conversion):
    if False:
        i = 10
        return i + 15
    'Returns whether concrete `function` can be called with `inputs`.'
    expected_structure = function.graph.structured_input_signature
    try:
        flatten_inputs = nest.flatten_up_to(expected_structure, inputs)
    except (TypeError, ValueError):
        return False
    for (arg, expected) in zip(flatten_inputs, nest.flatten(expected_structure)):
        if isinstance(expected, tensor.TensorSpec):
            if allow_conversion:
                arg = _try_convert_to_tensor_spec(arg, dtype_hint=expected.dtype)
            if not _is_tensor(arg) and (not isinstance(arg, tensor.TensorSpec)):
                return False
            if arg.dtype != expected.dtype:
                return False
            if not expected.shape.is_compatible_with(arg.shape):
                return False
        elif isinstance(expected, type_spec.TypeSpec):
            if not expected.is_compatible_with(arg):
                return False
        elif _is_tensor(arg):
            if id(arg) != id(expected):
                return False
        elif arg != expected:
            return False
    return True

def _deserialize_function_spec_as_nonmethod(function_spec_proto):
    if False:
        for i in range(10):
            print('nop')
    'Deserialize a FunctionSpec object from its proto representation.'
    typeless_fullargspec = nested_structure_coder.decode_proto(function_spec_proto.fullargspec)
    if function_spec_proto.is_method or (typeless_fullargspec.args and typeless_fullargspec.args[0] == 'self'):
        if not typeless_fullargspec.args:
            raise NotImplementedError("Cannot deserialize a method function without a named 'self' argument.")
        args = typeless_fullargspec.args[1:]
    else:
        args = typeless_fullargspec.args
    fullargspec = tf_inspect.FullArgSpec(args=args, varargs=typeless_fullargspec.varargs, varkw=typeless_fullargspec.varkw, defaults=typeless_fullargspec.defaults, kwonlyargs=typeless_fullargspec.kwonlyargs, kwonlydefaults=typeless_fullargspec.kwonlydefaults, annotations=typeless_fullargspec.annotations)
    input_signature = nested_structure_coder.decode_proto(function_spec_proto.input_signature)
    jit_compile = {saved_object_graph_pb2.FunctionSpec.JitCompile.DEFAULT: None, saved_object_graph_pb2.FunctionSpec.JitCompile.ON: True, saved_object_graph_pb2.FunctionSpec.JitCompile.OFF: False}.get(function_spec_proto.jit_compile)
    return function_type_utils.FunctionSpec.from_fullargspec_and_signature(fullargspec=fullargspec, input_signature=input_signature, jit_compile=jit_compile)

def set_preinitialized_function_spec(concrete_fn, spec):
    if False:
        for i in range(10):
            print('nop')
    'Set the FunctionType of the ConcreteFunction using FunctionSpec.'
    if spec is None:
        concrete_fn._function_type = None
        return
    unconstrained_type = function_type_lib.FunctionType([function_type_lib.Parameter(p.name, p.kind, p.optional, None) for p in spec.function_type.parameters.values()])
    (arg_specs, kwarg_specs) = concrete_fn.structured_input_signature
    (input_function_type, _) = function_type_lib.canonicalize_to_monomorphic(arg_specs, {function_type_lib.sanitize_arg_name(k): v for (k, v) in kwarg_specs.items()}, spec.default_values, {}, unconstrained_type)
    output_type = trace_type.from_value(concrete_fn.graph.structured_outputs)
    function_type = function_type_lib.FunctionType(input_function_type.parameters.values(), return_annotation=output_type)
    concrete_fn._function_type = function_type

def setup_bare_concrete_function(saved_bare_concrete_function, concrete_functions):
    if False:
        print('Hello World!')
    'Makes a restored bare concrete function callable.'
    concrete_function = concrete_functions[saved_bare_concrete_function.concrete_function_name]
    concrete_function._arg_keywords = saved_bare_concrete_function.argument_keywords
    concrete_function._num_positional_args = saved_bare_concrete_function.allowed_positional_arguments
    if saved_bare_concrete_function.HasField('function_spec'):
        function_spec = _deserialize_function_spec_as_nonmethod(saved_bare_concrete_function.function_spec)
        set_preinitialized_function_spec(concrete_function, function_spec)
    concrete_function.add_to_graph()
    return concrete_function

class RestoredFunction(def_function.Function):
    """Wrapper class for a function that has been restored from saved state.

  See `def_function.Function`.
  """

    def __init__(self, python_function, name, function_spec, concrete_functions):
        if False:
            for i in range(10):
                print('nop')
        super(RestoredFunction, self).__init__(python_function, name, autograph=False, jit_compile=function_spec.jit_compile)
        self.concrete_functions = concrete_functions
        self._function_type = function_spec.function_type
        self._default_values = function_spec.default_values
        self._omit_frequent_tracing_warning = True

    @property
    def _run_functions_eagerly(self):
        if False:
            while True:
                i = 10
        return False

    def _list_all_concrete_functions(self):
        if False:
            for i in range(10):
                print('nop')
        return self.concrete_functions

    def _list_all_concrete_functions_for_serialization(self):
        if False:
            i = 10
            return i + 15
        return self.concrete_functions

def recreate_function(saved_function, concrete_functions):
    if False:
        return 10
    'Creates a `Function` from a `SavedFunction`.\n\n  Args:\n    saved_function: `SavedFunction` proto.\n    concrete_functions: map from function name to `ConcreteFunction`. As a side\n      effect of this function, the `FunctionSpec` from `saved_function` is added\n      to each `ConcreteFunction` in this map.\n\n  Returns:\n    A `Function`.\n  '
    function_spec = _deserialize_function_spec_as_nonmethod(saved_function.function_spec)

    def restored_function_body(*args, **kwargs):
        if False:
            while True:
                i = 10
        'Calls a restored function or raises an error if no matching function.'
        if not saved_function.concrete_functions:
            raise ValueError('Found zero restored functions for caller function.')
        inputs = (args, kwargs)
        for allow_conversion in [False, True]:
            for function_name in saved_function.concrete_functions:
                function = concrete_functions[function_name]
                if any([inp is None for inp in function.captured_inputs]):
                    raise ValueError('Looks like you are trying to run a loaded non-Keras model that was trained using tf.distribute.experimental.ParameterServerStrategy with variable partitioning, which is not currently supported. Try using Keras to define your model if possible.')
                if _concrete_function_callable_with(function, inputs, allow_conversion):
                    return _call_concrete_function(function, inputs)
        signature_descriptions = []

        def _pretty_format_positional(positional):
            if False:
                for i in range(10):
                    print('nop')
            return 'Positional arguments ({} total):\n    * {}'.format(len(positional), '\n    * '.join((pprint.pformat(a) for a in positional)))
        for (index, function_name) in enumerate(saved_function.concrete_functions):
            concrete_function = concrete_functions[function_name]
            (positional, keyword) = concrete_function.structured_input_signature
            signature_descriptions.append('Option {}:\n  {}\n  Keyword arguments: {}'.format(index + 1, _pretty_format_positional(positional), keyword))
        raise ValueError(f'Could not find matching concrete function to call loaded from the SavedModel. Got:\n  {_pretty_format_positional(args)}\n  Keyword arguments: {kwargs}\n\n Expected these arguments to match one of the following {len(saved_function.concrete_functions)} option(s):\n\n{(chr(10) + chr(10)).join(signature_descriptions)}')
    concrete_function_objects = []
    for concrete_function_name in saved_function.concrete_functions:
        concrete_function_objects.append(concrete_functions[concrete_function_name])
    for cf in concrete_function_objects:
        set_preinitialized_function_spec(cf, function_spec)
    restored_function = RestoredFunction(restored_function_body, restored_function_body.__name__, function_spec, concrete_function_objects)
    return tf_decorator.make_decorator(restored_function_body, restored_function, decorator_argspec=function_spec.fullargspec)

def load_function_def_library(library, saved_object_graph=None, load_shared_name_suffix=None, wrapper_function=None):
    if False:
        i = 10
        return i + 15
    'Load a set of functions as concrete functions without captured inputs.\n\n  Functions names are manipulated during load such that they do not overlap\n  with previously created ones.\n\n  Gradients are re-registered under new names. Ops that reference the gradients\n  are updated to reflect the new registered names.\n\n  Args:\n    library: FunctionDefLibrary proto message.\n    saved_object_graph: SavedObjectGraph proto message. If not passed in,\n      concrete function structured signatures and outputs will not be set.\n    load_shared_name_suffix: If specified, used to uniquify shared names.\n      Otherwise, a unique name is generated.\n    wrapper_function: An object that will be wrapped on newly created functions.\n\n  Returns:\n    Map of original function names in the library to instances of\n    `ConcreteFunction` without captured inputs.\n\n  Raises:\n    ValueError: if functions dependencies have a cycle.\n  '
    library_function_names = set((fdef.signature.name for fdef in library.function))
    functions = {}
    renamed_functions = {}
    if ops.executing_eagerly_outside_functions():
        graph = ops.Graph()
    else:
        graph = ops.get_default_graph()
    if load_shared_name_suffix is None:
        load_shared_name_suffix = '_load_{}'.format(ops.uid())
    library_gradient_names = {}
    new_gradient_op_types = {}
    gradients_to_register = {}
    for gdef in library.registered_gradients:
        if gdef.registered_op_type:
            new_op_type = custom_gradient.generate_name()
            old_op_type = compat.as_bytes(gdef.registered_op_type)
            library_gradient_names[old_op_type] = gdef.gradient_func
            new_gradient_op_types[old_op_type] = new_op_type
            gradients_to_register[gdef.gradient_func] = new_op_type
    function_deps = {}
    for fdef in library.function:
        function_deps[fdef.signature.name] = _list_function_deps(fdef, library_function_names, library_gradient_names)
    loaded_gradients = {}
    for fdef in _sort_function_defs(library, function_deps):
        orig_name = _fix_fdef_in_place(fdef, functions, load_shared_name_suffix, new_gradient_op_types)
        structured_input_signature = None
        structured_outputs = None
        if saved_object_graph is not None and orig_name in saved_object_graph.concrete_functions:
            proto = saved_object_graph.concrete_functions[orig_name]
            structured_input_signature = nested_structure_coder.decode_proto(proto.canonicalized_input_signature)
            structured_outputs = nested_structure_coder.decode_proto(proto.output_signature)
        with graph.as_default():
            func_graph = function_def_lib.function_def_to_graph(fdef, structured_input_signature=structured_input_signature, structured_outputs=structured_outputs)
        _restore_gradient_functions(func_graph, renamed_functions, loaded_gradients)
        for dep in function_deps[orig_name]:
            functions[dep].add_to_graph(func_graph)
        if '_input_shapes' in fdef.attr:
            del fdef.attr['_input_shapes']
        function_type = function_type_lib.from_structured_signature(func_graph.structured_input_signature, func_graph.structured_outputs, func_graph.function_captures.capture_types)
        func = function_lib.ConcreteFunction.from_func_graph(func_graph, function_type, attrs=fdef.attr)
        if wrapper_function:
            func = wrapper_function(func)
        func.add_to_graph(graph)
        functions[orig_name] = func
        renamed_functions[func.name] = func
        if any((op.type == 'TRTEngineOp' for op in func_graph.get_operations())):
            func.add_to_graph(ops.get_default_graph())
        if orig_name in gradients_to_register:
            gradient_op_type = gradients_to_register[orig_name]
            loaded_gradients[compat.as_bytes(gradient_op_type)] = func
            ops.RegisterGradient(gradient_op_type)(_gen_gradient_func(func))
    return functions

def _gen_gradient_func(func):
    if False:
        for i in range(10):
            print('nop')
    'Wraps a deserialized function.'

    def gradient_func(unused_op, *result_grads):
        if False:
            while True:
                i = 10

        def none_to_zero(x, t):
            if False:
                i = 10
                return i + 15
            if x is not None:
                return x
            (shape, dtype) = default_gradient.shape_and_dtype(t)
            if shape.is_fully_defined():
                return default_gradient.zeros_like(t)
            dims = []
            if shape.rank is not None:
                dims = [1 if d is None else d for d in shape.as_list()]
            return array_ops.zeros(dims, dtype)
        result_grads = [none_to_zero(x, t) for (x, t) in zip(result_grads, func.graph.inputs)]
        return func(*result_grads)
    return gradient_func

def _restore_gradient_functions(func_graph, renamed_functions, loaded_gradients):
    if False:
        for i in range(10):
            print('nop')
    "Populate function op's _gradient_function with default gradient."
    for op in func_graph.get_operations():
        if op.type in ['StatefulPartitionedCall', 'PartitionedCall']:
            function = renamed_functions[compat.as_bytes(op.node_def.attr['f'].func.name)]
            op._gradient_function = function._get_gradient_function()
        try:
            gradient_op_type = op.get_attr('_gradient_op_type')
        except ValueError:
            pass
        else:
            if gradient_op_type in loaded_gradients:
                grad_fn = loaded_gradients[gradient_op_type]
                grad_fn._num_positional_args = len(op.inputs)
                grad_fn._arg_keywords = [inp.name for inp in op.inputs]

def _sort_function_defs(library, function_deps):
    if False:
        print('Hello World!')
    'Return a topologic sort of FunctionDefs in a library.'
    edges = collections.defaultdict(list)
    in_count = collections.defaultdict(lambda : 0)
    for (fname, deps) in function_deps.items():
        for dep in deps:
            edges[dep].append(fname)
            in_count[fname] += 1
    ready = [fdef.signature.name for fdef in library.function if in_count[fdef.signature.name] == 0]
    output = []
    while ready:
        node = ready.pop()
        output.append(node)
        for dest in edges[node]:
            in_count[dest] -= 1
            if not in_count[dest]:
                ready.append(dest)
    if len(output) != len(library.function):
        failed_to_resolve = sorted(set(in_count.keys()) - set(output))
        raise ValueError('There is a cyclic dependency between functions. ', f'Could not resolve {failed_to_resolve}.')
    reverse = {fdef.signature.name: fdef for fdef in library.function}
    return [reverse[x] for x in output]

def _get_gradient_op_type(node_def):
    if False:
        while True:
            i = 10
    'Returns the custom gradient op type.'
    if '_gradient_op_type' in node_def.attr and node_def.op not in ['StatefulPartitionedCall', 'PartitionedCall']:
        return node_def.attr['_gradient_op_type'].s
    return None

def fix_node_def(node_def, functions, shared_name_suffix):
    if False:
        for i in range(10):
            print('nop')
    'Replace functions calls and shared names in `node_def`.'
    if node_def.op in functions:
        node_def.op = functions[node_def.op].name
    for (_, attr_value) in node_def.attr.items():
        if attr_value.WhichOneof('value') == 'func':
            attr_value.func.name = functions[attr_value.func.name].name
        elif attr_value.WhichOneof('value') == 'list':
            for fn in attr_value.list.func:
                fn.name = functions[fn.name].name
    if node_def.op == 'HashTableV2':
        if 'use_node_name_sharing' not in node_def.attr or not node_def.attr['use_node_name_sharing'].b:
            node_def.attr['use_node_name_sharing'].b = True
            shared_name_suffix += '_{}'.format(ops.uid())
    op_def = op_def_registry.get(node_def.op)
    if op_def:
        attr = next((a for a in op_def.attr if a.name == 'shared_name'), None)
        if attr:
            shared_name = None
            if 'shared_name' in node_def.attr and node_def.attr['shared_name'].s:
                shared_name = node_def.attr['shared_name'].s
            elif attr.default_value.s:
                shared_name = compat.as_bytes(attr.default_value.s)
            if not shared_name:
                shared_name = compat.as_bytes(node_def.name)
            node_def.attr['shared_name'].s = shared_name + compat.as_bytes(shared_name_suffix)

def _fix_fdef_in_place(fdef, functions, shared_name_suffix, new_gradient_op_types):
    if False:
        return 10
    'Fixes a FunctionDef proto to be loaded in current context.\n\n  In particular, when loading a function library into an eager context, one\n  must rename the functions to avoid conflicts with existent functions.\n\n  Args:\n    fdef: FunctionDef proto to fix. It is mutated in-place.\n    functions: map from function name to a ConcreteFunction instance.\n    shared_name_suffix: A unique string for this load which helps to avoid\n      `shared_name` collisions across loads. Two functions from the same load\n      using the same `shared_name` still need to share, but functions from\n      different loads with the same `shared_name` should not.\n    new_gradient_op_types: map from old gradient op type to newly generated op\n      type.\n\n  Returns:\n    orig_name: original value of fdef.signature.name\n  '
    orig_name = fdef.signature.name
    contains_unsaved_custom_gradients = False
    for node_def in fdef.node_def:
        fix_node_def(node_def, functions, shared_name_suffix)
        op_type = _get_gradient_op_type(node_def)
        if op_type is not None:
            if op_type in new_gradient_op_types:
                node_def.attr['_gradient_op_type'].s = compat.as_bytes(new_gradient_op_types[op_type])
            else:
                contains_unsaved_custom_gradients = True
    if contains_unsaved_custom_gradients:
        logging.warning('Importing a function (%s) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.', fdef.signature.name)
    fdef.signature.name = _clean_function_name(fdef.signature.name)
    return orig_name

def _list_function_deps(fdef, library_function_names, library_gradient_names):
    if False:
        for i in range(10):
            print('nop')
    'Find functions referenced in `fdef`.'
    deps = set()
    for node_def in fdef.node_def:
        grad_op_type = _get_gradient_op_type(node_def)
        if node_def.op in library_function_names:
            deps.add(node_def.op)
        elif grad_op_type and grad_op_type in library_gradient_names:
            deps.add(library_gradient_names[grad_op_type])
        else:
            for (_, attr_value) in node_def.attr.items():
                if attr_value.WhichOneof('value') == 'func':
                    deps.add(attr_value.func.name)
                elif attr_value.WhichOneof('value') == 'list':
                    for fn in attr_value.list.func:
                        deps.add(fn.name)
    return deps
_FUNCTION_WRAPPER_NAME_REGEX = '^%s(.*)_\\d+$' % function_lib._INFERENCE_PREFIX

def _clean_function_name(name):
    if False:
        i = 10
        return i + 15
    'Vanity function to keep the function names comprehensible.'
    match = re.search(_FUNCTION_WRAPPER_NAME_REGEX, name)
    if match:
        return match.group(1)
    else:
        return name