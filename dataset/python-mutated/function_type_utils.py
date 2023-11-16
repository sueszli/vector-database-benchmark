"""Utilities for using FunctionType with tf.function."""
import functools
import inspect
from typing import Any, Dict, Tuple
import six
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import nest

def to_fullargspec(function_type: function_type_lib.FunctionType, default_values: Dict[str, Any]) -> inspect.FullArgSpec:
    if False:
        print('Hello World!')
    'Generates backwards compatible FullArgSpec from FunctionType.'
    args = []
    varargs = None
    varkw = None
    defaults = []
    kwonlyargs = []
    kwonlydefaults = {}
    for parameter in function_type.parameters.values():
        if parameter.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]:
            args.append(parameter.name)
            if parameter.default is not inspect.Parameter.empty:
                defaults.append(default_values[parameter.name])
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            kwonlyargs.append(parameter.name)
            if parameter.default is not inspect.Parameter.empty:
                kwonlydefaults[parameter.name] = default_values[parameter.name]
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            varargs = parameter.name
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            varkw = parameter.name
    return inspect.FullArgSpec(args, varargs, varkw, tuple(defaults) if defaults else None, kwonlyargs, kwonlydefaults if kwonlydefaults else None, annotations={})

def _to_default_values(fullargspec):
    if False:
        i = 10
        return i + 15
    "Returns default values from the function's inspected fullargspec."
    if fullargspec.defaults is not None:
        defaults = {name: value for (name, value) in zip(fullargspec.args[-len(fullargspec.defaults):], fullargspec.defaults)}
    else:
        defaults = {}
    if fullargspec.kwonlydefaults is not None:
        defaults.update(fullargspec.kwonlydefaults)
    defaults = {function_type_lib.sanitize_arg_name(name): value for (name, value) in defaults.items()}
    return defaults

def to_function_type(fullargspec):
    if False:
        i = 10
        return i + 15
    'Generates FunctionType and default values from fullargspec.'
    default_values = _to_default_values(fullargspec)
    parameters = []
    for arg in fullargspec.args:
        arg_name = function_type_lib.sanitize_arg_name(arg)
        parameters.append(function_type_lib.Parameter(arg_name, function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, arg_name in default_values, None))
    if fullargspec.varargs is not None:
        parameters.append(function_type_lib.Parameter(fullargspec.varargs, function_type_lib.Parameter.VAR_POSITIONAL, False, None))
    for kwarg in fullargspec.kwonlyargs:
        parameters.append(function_type_lib.Parameter(function_type_lib.sanitize_arg_name(kwarg), function_type_lib.Parameter.KEYWORD_ONLY, kwarg in default_values, None))
    if fullargspec.varkw is not None:
        parameters.append(function_type_lib.Parameter(fullargspec.varkw, function_type_lib.Parameter.VAR_KEYWORD, False, None))
    return (function_type_lib.FunctionType(parameters), default_values)

def to_input_signature(function_type):
    if False:
        return 10
    'Extracts an input_signature from function_type instance.'
    constrained_parameters = list(function_type.parameters.keys())
    if 'self' in constrained_parameters:
        constrained_parameters.pop(0)
    if not constrained_parameters:
        return tuple()
    constraints = []
    is_auto_constrained = False
    for parameter_name in constrained_parameters:
        parameter = function_type.parameters[parameter_name]
        constraint = None
        if parameter.type_constraint:
            constraint = parameter.type_constraint.placeholder_value(trace_type.InternalPlaceholderContext(unnest_only=True))
            if any((not isinstance(arg, tensor.TensorSpec) for arg in nest.flatten([constraint], expand_composites=True))):
                is_auto_constrained = True
                break
            else:
                constraints.append(constraint)
    if is_auto_constrained and (not constraints):
        return tuple()
    return tuple(constraints) if constraints else None

def to_arg_names(function_type):
    if False:
        i = 10
        return i + 15
    'Generates a list of arg names from a FunctionType.'
    arg_names = []
    for p in function_type.parameters.values():
        if p.kind in {function_type_lib.Parameter.POSITIONAL_ONLY, function_type_lib.Parameter.POSITIONAL_OR_KEYWORD}:
            arg_names.append(p.name)
    return arg_names

class FunctionSpec(object):
    """Specification of how to bind arguments to a function.

  Deprecated. Please use FunctionType instead.
  """

    @classmethod
    def from_function_and_signature(cls, python_function, input_signature, is_pure=False, jit_compile=None):
        if False:
            while True:
                i = 10
        'Creates a FunctionSpec instance given a python function and signature.\n\n    Args:\n      python_function: a function to inspect\n      input_signature: a signature of the function (None, if variable)\n      is_pure: if True all input arguments (including variables and constants)\n        will be converted to tensors and no variable changes allowed.\n      jit_compile: see `tf.function`\n\n    Returns:\n      instance of FunctionSpec\n    '
        (function_type, default_values) = make_function_type(python_function, input_signature)
        while isinstance(python_function, functools.partial):
            python_function = python_function.func
        name = getattr(python_function, '__name__', 'f')
        return FunctionSpec(function_type, default_values, is_pure=is_pure, jit_compile=jit_compile, name=name)

    @classmethod
    def from_fullargspec_and_signature(cls, fullargspec, input_signature, is_pure=False, name=None, jit_compile=None):
        if False:
            return 10
        'Construct FunctionSpec from legacy FullArgSpec format.'
        (function_type, default_values) = to_function_type(fullargspec)
        if input_signature:
            input_signature = tuple(input_signature)
            _validate_signature(input_signature)
            function_type = function_type_lib.add_type_constraints(function_type, input_signature, default_values)
        return FunctionSpec(function_type, default_values, is_pure, name, jit_compile)

    def __init__(self, function_type, default_values, is_pure=False, name=None, jit_compile=None):
        if False:
            while True:
                i = 10
        'Constructs a FunctionSpec describing a python function.\n\n    Args:\n      function_type: A FunctionType describing the python function signature.\n      default_values: Dictionary mapping parameter names to default values.\n      is_pure: if True all input arguments (including variables and constants)\n        will be converted to tensors and no variable changes allowed.\n      name: Name of the function\n      jit_compile: see `tf.function`.\n    '
        self._function_type = function_type
        self._default_values = default_values
        self._fullargspec = to_fullargspec(function_type, default_values)
        self._is_pure = is_pure
        self._jit_compile = jit_compile
        self._name = name or 'f'
        self._input_signature = to_input_signature(function_type)

    @property
    def default_values(self):
        if False:
            return 10
        'Returns dict mapping parameter names to default values.'
        return self._default_values

    @property
    def function_type(self):
        if False:
            i = 10
            return i + 15
        'Returns a FunctionType representing the Python function signature.'
        return self._function_type

    @property
    def fullargspec(self):
        if False:
            i = 10
            return i + 15
        return self._fullargspec

    @property
    def input_signature(self):
        if False:
            return 10
        return self._input_signature

    @property
    def flat_input_signature(self):
        if False:
            for i in range(10):
                print('nop')
        return tuple(nest.flatten(self.input_signature, expand_composites=True))

    @property
    def is_pure(self):
        if False:
            for i in range(10):
                print('nop')
        return self._is_pure

    @property
    def jit_compile(self):
        if False:
            print('Hello World!')
        return self._jit_compile

    @property
    def arg_names(self):
        if False:
            for i in range(10):
                print('nop')
        return to_arg_names(self.function_type)

    def signature_summary(self, default_values=False):
        if False:
            print('Hello World!')
        "Returns a string summarizing this function's signature.\n\n    Args:\n      default_values: If true, then include default values in the signature.\n\n    Returns:\n      A `string`.\n    "
        summary = f'{self._function_type!r}'
        if default_values:
            summary += '\nDefaults:'
            if self.default_values:
                for (name, value) in self.default_values.items():
                    summary += f'\n  {name}: {value!r}'
            else:
                summary += '\n  None'
        return summary

def make_function_type(python_function, input_signature):
    if False:
        print('Hello World!')
    'Generates a FunctionType for python_function.'
    _validate_signature(input_signature)
    function_type = function_type_lib.FunctionType.from_callable(python_function)
    default_values = function_type_lib.FunctionType.get_default_values(python_function)
    if input_signature is not None:
        input_signature = tuple(input_signature)
        function_type = function_type_lib.add_type_constraints(function_type, input_signature, default_values)
    return (function_type, default_values)

def make_canonicalized_monomorphic_type(args: Any, kwargs: Any, capture_types: Any, polymorphic_type) -> Tuple[function_type_lib.FunctionType, trace_type.InternalTracingContext]:
    if False:
        return 10
    'Generates function type given the function arguments.'
    kwargs = {function_type_lib.sanitize_arg_name(name): value for (name, value) in kwargs.items()}
    (function_type, type_context) = function_type_lib.canonicalize_to_monomorphic(args, kwargs, {}, capture_types, polymorphic_type)
    return (function_type, type_context)

def canonicalize_function_inputs(args, kwargs, function_type, default_values=None, is_pure=False):
    if False:
        print('Hello World!')
    "Canonicalizes `args` and `kwargs`.\n\n  Canonicalize the inputs to the Python function using FunctionType.\n  In particular, we parse the varargs and kwargs that the\n  original function was called with into a tuple corresponding to the\n  Python function's positional (named) arguments and a dictionary\n  corresponding to its kwargs.  Missing default arguments are added.\n\n  If the FunctionType has an type constraints, then they are used to convert\n  arguments to tensors; otherwise, any inputs containing numpy arrays are\n  converted to tensors.\n\n\n  Args:\n    args: The varargs this object was called with.\n    kwargs: The keyword args this function was called with.\n    function_type: FunctionType to canonicalize against.\n    default_values: Default values to use.\n    is_pure: Force variable inputs to Tensors.\n\n  Returns:\n    A canonicalized ordering of the inputs, as well as full and filtered\n    (Tensors and Variables only) versions of their concatenated flattened\n    representations, represented by a tuple in the form (args, kwargs,\n    flat_args, filtered_flat_args). Here: `args` is a full list of bound\n    arguments, and `kwargs` contains only true keyword arguments, as opposed\n    to named arguments called in a keyword-like fashion.\n\n  Raises:\n    ValueError: If a keyword in `kwargs` cannot be matched with a positional\n      argument when an input signature is specified, or when the inputs\n      do not conform to the input signature.\n  "
    default_values = {} if not default_values else default_values
    if is_pure:
        (args, kwargs) = _convert_variables_to_tensors(args, kwargs)
    bound_arguments = bind_function_inputs(args, kwargs, function_type, default_values)
    return bound_arguments

def bind_function_inputs(args, kwargs, function_type, default_values):
    if False:
        while True:
            i = 10
    'Bind `args` and `kwargs` into a canonicalized signature args, kwargs.'
    sanitized_kwargs = {function_type_lib.sanitize_arg_name(k): v for (k, v) in kwargs.items()}
    if len(kwargs) != len(sanitized_kwargs):
        raise ValueError(f'Name collision after sanitization. Please rename tf.function input parameters. Original: {sorted(kwargs.keys())}, Sanitized: {sorted(sanitized_kwargs.keys())}')
    try:
        bound_arguments = function_type.bind_with_defaults(args, sanitized_kwargs, default_values)
    except Exception as e:
        raise TypeError(f'Binding inputs to tf.function failed due to `{e}`. Received args: {args} and kwargs: {sanitized_kwargs} for signature: {function_type}.') from e
    return bound_arguments

def _validate_signature(signature):
    if False:
        for i in range(10):
            print('nop')
    'Checks the input_signature to be valid.'
    if signature is None:
        return
    if not isinstance(signature, (tuple, list)):
        raise TypeError(f'input_signature must be either a tuple or a list, got {type(signature)}.')
    variable_specs = _get_variable_specs(signature)
    if variable_specs:
        raise TypeError(f"input_signature doesn't support VariableSpec, got {variable_specs}")
    if any((not isinstance(arg, tensor.TensorSpec) for arg in nest.flatten(signature, expand_composites=True))):
        bad_args = [arg for arg in nest.flatten(signature, expand_composites=True) if not isinstance(arg, tensor.TensorSpec)]
        raise TypeError(f'input_signature must be a possibly nested sequence of TensorSpec objects, got invalid args {bad_args} with types {list(six.moves.map(type, bad_args))}.')

def _to_tensor_or_tensor_spec(x):
    if False:
        for i in range(10):
            print('nop')
    return x if isinstance(x, (tensor.Tensor, tensor.TensorSpec)) else ops.convert_to_tensor(x)

def _convert_variables_to_tensors(args, kwargs):
    if False:
        return 10
    args = [_to_tensor_or_tensor_spec(x) for x in args]
    kwargs = {kw: _to_tensor_or_tensor_spec(x) for (kw, x) in kwargs.items()}
    return (tuple(args), kwargs)

def _get_variable_specs(args):
    if False:
        return 10
    'Returns `VariableSpecs` from `args`.'
    variable_specs = []
    for arg in nest.flatten(args):
        if not isinstance(arg, type_spec.TypeSpec):
            continue
        if isinstance(arg, resource_variable_ops.VariableSpec):
            variable_specs.append(arg)
        elif not isinstance(arg, tensor.TensorSpec):
            variable_specs.extend(_get_variable_specs(arg._component_specs))
    return variable_specs

def derive_from_graph(func_graph):
    if False:
        i = 10
        return i + 15
    'Derives a FunctionType from FuncGraph.'
    input_signature = (tuple((trace_type.from_value(i) for i in func_graph.inputs)), {})
    output_signature = tuple((trace_type.from_value(o) for o in func_graph.outputs))
    return function_type_lib.from_structured_signature(input_signature, output_signature, func_graph.function_captures.capture_types)

def is_same_structure(structure1, structure2, check_values=False):
    if False:
        i = 10
        return i + 15
    'Check two structures for equality, optionally of types and of values.'
    try:
        nest.assert_same_structure(structure1, structure2, expand_composites=True)
    except (ValueError, TypeError):
        return False
    if check_values:
        flattened1 = nest.flatten(structure1, expand_composites=True)
        flattened2 = nest.flatten(structure2, expand_composites=True)
        if any((type(f1) is not type(f2) for (f1, f2) in zip(flattened1, flattened2))):
            return False
        return flattened1 == flattened2
    return True