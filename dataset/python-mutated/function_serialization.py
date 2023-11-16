"""Tools for serializing `Function`s."""
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import function as defun
from tensorflow.python.eager import wrap_function as wrap_function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest

def _serialize_function_spec(function_spec):
    if False:
        i = 10
        return i + 15
    'Serialize a FunctionSpec object into its proto representation.'
    if function_spec.fullargspec.args and function_spec.fullargspec.args[0] == 'self':
        raise TypeError("Can not serialize tf.function with unbound 'self' parameter.")
    proto = saved_object_graph_pb2.FunctionSpec()
    proto.fullargspec.CopyFrom(nested_structure_coder.encode_structure(function_spec.fullargspec._replace(annotations={})))
    proto.is_method = False
    proto.input_signature.CopyFrom(nested_structure_coder.encode_structure(function_spec.input_signature))
    proto.jit_compile = {None: saved_object_graph_pb2.FunctionSpec.JitCompile.DEFAULT, True: saved_object_graph_pb2.FunctionSpec.JitCompile.ON, False: saved_object_graph_pb2.FunctionSpec.JitCompile.OFF}.get(function_spec.jit_compile)
    return proto

def serialize_concrete_function(concrete_function, node_ids):
    if False:
        for i in range(10):
            print('nop')
    'Build a SavedConcreteFunction.'
    bound_inputs = []
    try:
        for capture in concrete_function.captured_inputs:
            bound_inputs.append(node_ids[capture])
    except KeyError:
        raise KeyError(f"Failed to add concrete function '{concrete_function.name}' to object-based SavedModel as it captures tensor {capture!r} which is unsupported or not reachable from root. One reason could be that a stateful object or a variable that the function depends on is not assigned to an attribute of the serialized trackable object (see SaveTest.test_captures_unreachable_variable).")
    concrete_function_proto = saved_object_graph_pb2.SavedConcreteFunction()
    structured_outputs = func_graph_module.convert_structure_to_signature(concrete_function.structured_outputs)
    concrete_function_proto.canonicalized_input_signature.CopyFrom(nested_structure_coder.encode_structure(concrete_function.structured_input_signature))
    concrete_function_proto.output_signature.CopyFrom(nested_structure_coder.encode_structure(structured_outputs))
    concrete_function_proto.bound_inputs.extend(bound_inputs)
    return concrete_function_proto

def get_preinitialized_function_spec(concrete_function):
    if False:
        return 10
    'Generates an unconstrained FunctionSpec from FunctionType.'
    if concrete_function.structured_input_signature is None or isinstance(concrete_function, wrap_function_lib.WrappedFunction):
        return None
    function_type = concrete_function.function_type
    if function_type is None:
        return None
    unconstrained_type = function_type_lib.FunctionType([function_type_lib.Parameter(p.name, p.kind, p.optional, None) for p in function_type.parameters.values()])
    default_values = {p.default for p in function_type.parameters.values() if p.optional}
    return function_type_utils.FunctionSpec(unconstrained_type, default_values, False, name=concrete_function.name)

def serialize_bare_concrete_function(concrete_function):
    if False:
        print('Hello World!')
    'Build a SavedBareConcreteFunction.'
    proto = saved_object_graph_pb2.SavedBareConcreteFunction(concrete_function_name=concrete_function.name, allowed_positional_arguments=concrete_function._num_positional_args, argument_keywords=concrete_function._arg_keywords)
    function_spec = get_preinitialized_function_spec(concrete_function)
    if function_spec is not None:
        proto.function_spec.CopyFrom(_serialize_function_spec(function_spec))
    return proto

def serialize_function(function, concrete_functions):
    if False:
        while True:
            i = 10
    'Build a SavedFunction proto.'
    proto = saved_object_graph_pb2.SavedFunction()
    function_spec_proto = _serialize_function_spec(function.function_spec)
    proto.function_spec.CopyFrom(function_spec_proto)
    for concrete_function in concrete_functions:
        proto.concrete_functions.append(concrete_function.name)
    return proto

def wrap_cached_variables(concrete_function):
    if False:
        for i in range(10):
            print('nop')
    'Wraps the concrete function if it uses cached read tensors.\n\n  This function creates a new concrete function that captures variables\n  instead of the cached read tensors.\n\n  Args:\n    concrete_function: A Concrete function that maybe captures cached read\n      tensors.\n\n  Returns:\n    A concrete function that wraps the original concrete function, which\n    captures variables instead. If the original function did not capture any\n    cached values, then the function is not wrapped and the original object is\n    returned.\n  '
    outer_graph = func_graph_module.FuncGraph('{}_no_cache'.format(concrete_function.graph.name))
    mapped_captures = None
    remapped_captures = {}
    with outer_graph.as_default():
        for (capture, placeholder) in concrete_function.graph.captures:
            cached_variable = getattr(capture, '_cached_variable', None)
            if cached_variable is None:
                continue
            cached_variable = cached_variable()
            new_cached_value = cached_variable.read_value()
            key = id(capture)
            external = concrete_function.graph.function_captures.by_val_external[key]
            internal = concrete_function.graph.function_captures.by_val_internal[key]
            remapped_captures[key] = [external, internal]
            concrete_function.graph.function_captures.add_or_replace(key=key, external=new_cached_value, internal=placeholder, is_by_ref=False)
            mapped_captures = True
    if not mapped_captures:
        return concrete_function
    inner_concrete = defun.ConcreteFunction.from_func_graph(concrete_function.graph, concrete_function.function_type, {})

    def wrap_function(*args):
        if False:
            i = 10
            return i + 15
        return inner_concrete._call_flat(list(args), inner_concrete.captured_inputs)
    args = nest.flatten(concrete_function.structured_input_signature, expand_composites=True)
    func_graph_module.func_graph_from_py_func(None, wrap_function, args=tuple(args), kwargs={}, func_graph=outer_graph)
    fn = defun.ConcreteFunction.from_func_graph(outer_graph, concrete_function.function_type, {})
    fn._arg_keywords = concrete_function._arg_keywords
    fn._num_positional_args = concrete_function._num_positional_args
    for (key, capture) in remapped_captures.items():
        (external, internal) = capture
        concrete_function.graph._function_captures.add_or_replace(key=key, external=external, internal=internal, is_by_ref=False)
    return fn