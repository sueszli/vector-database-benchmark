"""Helpers for working with signatures in tf.saved_model.save."""
from absl import logging
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import attributes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.trackable import base
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
DEFAULT_SIGNATURE_ATTR = '_default_save_signature'
SIGNATURE_ATTRIBUTE_NAME = 'signatures'
_NUM_DISPLAY_NORMALIZED_SIGNATURES = 5

def _get_signature(function):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(function, def_function.Function) and function.input_signature is not None:
        function = function._get_concrete_function_garbage_collected()
    if not isinstance(function, defun.ConcreteFunction):
        return None
    return function

def _valid_signature(concrete_function):
    if False:
        return 10
    'Returns whether concrete function can be converted to a signature.'
    if not concrete_function.outputs:
        return False
    try:
        _validate_inputs(concrete_function)
        _normalize_outputs(concrete_function.structured_outputs, 'unused', 'unused')
    except ValueError:
        return False
    return True

def _validate_inputs(concrete_function):
    if False:
        print('Hello World!')
    'Raises error if input type is tf.Variable.'
    if any((isinstance(inp, resource_variable_ops.VariableSpec) for inp in nest.flatten(concrete_function.structured_input_signature))):
        raise ValueError(f"Unable to serialize concrete_function '{concrete_function.name}'with tf.Variable input. Functions that expect tf.Variable inputs cannot be exported as signatures.")

def _get_signature_name_changes(concrete_function):
    if False:
        print('Hello World!')
    'Checks for user-specified signature input names that are normalized.'
    name_changes = {}
    for (signature_input_name, graph_input) in zip(concrete_function.function_def.signature.input_arg, concrete_function.graph.inputs):
        try:
            user_specified_name = compat.as_str(graph_input.op.get_attr('_user_specified_name'))
            if signature_input_name.name != user_specified_name:
                name_changes[user_specified_name] = signature_input_name.name
        except ValueError:
            pass
    return name_changes

def find_function_to_export(saveable_view):
    if False:
        i = 10
        return i + 15
    'Function to export, None if no suitable function was found.'
    children = saveable_view.list_children(saveable_view.root)
    possible_signatures = []
    for (name, child) in children:
        if not isinstance(child, (def_function.Function, defun.ConcreteFunction)):
            continue
        if name == DEFAULT_SIGNATURE_ATTR:
            return child
        concrete = _get_signature(child)
        if concrete is not None and _valid_signature(concrete):
            possible_signatures.append(concrete)
    if len(possible_signatures) == 1:
        single_function = possible_signatures[0]
        signature = _get_signature(single_function)
        if signature and _valid_signature(signature):
            return signature
    return None

def canonicalize_signatures(signatures):
    if False:
        return 10
    'Converts `signatures` into a dictionary of concrete functions.'
    if signatures is None:
        return ({}, {}, {})
    if not isinstance(signatures, collections_abc.Mapping):
        signatures = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signatures}
    num_normalized_signatures_counter = 0
    concrete_signatures = {}
    wrapped_functions = {}
    defaults = {}
    for (signature_key, function) in signatures.items():
        original_function = signature_function = _get_signature(function)
        if signature_function is None:
            raise ValueError(f'Expected a TensorFlow function for which to generate a signature, but got {function}. Only `tf.functions` with an input signature or concrete functions can be used as a signature.')
        wrapped_functions[original_function] = signature_function = wrapped_functions.get(original_function) or function_serialization.wrap_cached_variables(original_function)
        _validate_inputs(signature_function)
        if num_normalized_signatures_counter < _NUM_DISPLAY_NORMALIZED_SIGNATURES:
            signature_name_changes = _get_signature_name_changes(signature_function)
            if signature_name_changes:
                num_normalized_signatures_counter += 1
                logging.info('Function `%s` contains input name(s) %s with unsupported characters which will be renamed to %s in the SavedModel.', compat.as_str(signature_function.graph.name), ', '.join(signature_name_changes.keys()), ', '.join(signature_name_changes.values()))

        def signature_wrapper(**kwargs):
            if False:
                print('Hello World!')
            structured_outputs = signature_function(**kwargs)
            return _normalize_outputs(structured_outputs, signature_function.name, signature_key)
        if hasattr(function, '__name__'):
            signature_wrapper.__name__ = 'signature_wrapper_' + function.__name__
        experimental_attributes = {}
        for attr in attributes.POLYMORPHIC_FUNCTION_ALLOWLIST:
            attr_value = signature_function.function_def.attr.get(attr, None)
            if attr != attributes.NO_INLINE and attr_value is not None:
                experimental_attributes[attr] = attr_value
        if not experimental_attributes:
            experimental_attributes = None
        wrapped_function = def_function.function(signature_wrapper, experimental_attributes=experimental_attributes)
        tensor_spec_signature = {}
        if signature_function.structured_input_signature is not None:
            inputs = filter(lambda x: isinstance(x, tensor.TensorSpec), nest.flatten(signature_function.structured_input_signature, expand_composites=True))
        else:
            inputs = signature_function.inputs
        for (keyword, inp) in zip(signature_function._arg_keywords, inputs):
            keyword = compat.as_str(keyword)
            if isinstance(inp, tensor.TensorSpec):
                spec = tensor.TensorSpec(inp.shape, inp.dtype, name=keyword)
            else:
                spec = tensor.TensorSpec.from_tensor(inp, name=keyword)
            tensor_spec_signature[keyword] = spec
        final_concrete = wrapped_function._get_concrete_function_garbage_collected(**tensor_spec_signature)
        if len(final_concrete._arg_keywords) == 1:
            final_concrete._num_positional_args = 1
        else:
            final_concrete._num_positional_args = 0
        concrete_signatures[signature_key] = final_concrete
        if isinstance(function, core.PolymorphicFunction):
            flattened_defaults = nest.flatten(function.function_spec.fullargspec.defaults)
            len_default = len(flattened_defaults or [])
            arg_names = list(tensor_spec_signature.keys())
            if len_default > 0:
                for (arg, default) in zip(arg_names[-len_default:], flattened_defaults or []):
                    if not isinstance(default, tensor.Tensor):
                        continue
                    defaults.setdefault(signature_key, {})[arg] = default
    return (concrete_signatures, wrapped_functions, defaults)

def _normalize_outputs(outputs, function_name, signature_key):
    if False:
        i = 10
        return i + 15
    'Normalize outputs if necessary and check that they are tensors.'
    if not isinstance(outputs, collections_abc.Mapping):
        if hasattr(outputs, '_asdict'):
            outputs = outputs._asdict()
        else:
            if not isinstance(outputs, collections_abc.Sequence):
                outputs = [outputs]
            outputs = {'output_{}'.format(output_index): output for (output_index, output) in enumerate(outputs)}
    for (key, value) in outputs.items():
        if not isinstance(key, compat.bytes_or_text_types):
            raise ValueError(f'Got a dictionary with a non-string key {key!r} in the output of the function {compat.as_str_any(function_name)} used to generate the SavedModel signature {signature_key!r}.')
        if not isinstance(value, (tensor.Tensor, composite_tensor.CompositeTensor)):
            raise ValueError(f'Got a non-Tensor value {value!r} for key {key!r} in the output of the function {compat.as_str_any(function_name)} used to generate the SavedModel signature {signature_key!r}. Outputs for functions used as signatures must be a single Tensor, a sequence of Tensors, or a dictionary from string to Tensor.')
    return outputs

class _SignatureMap(collections_abc.Mapping, base.Trackable):
    """A collection of SavedModel signatures."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._signatures = {}

    def _add_signature(self, name, concrete_function):
        if False:
            for i in range(10):
                print('nop')
        'Adds a signature to the _SignatureMap.'
        self._signatures[name] = concrete_function

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return self._signatures[key]

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self._signatures)

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self._signatures)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '_SignatureMap({})'.format(self._signatures)

    def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
        if False:
            i = 10
            return i + 15
        if save_type != base.SaveType.SAVEDMODEL:
            return {}
        return {key: value for (key, value) in self.items() if isinstance(value, (def_function.Function, defun.ConcreteFunction))}
revived_types.register_revived_type('signature_map', lambda obj: isinstance(obj, _SignatureMap), versions=[revived_types.VersionedTypeRegistration(object_factory=lambda proto: _SignatureMap(), version=1, min_producer_version=1, min_consumer_version=1, setter=_SignatureMap._add_signature)])

def create_signature_map(signatures):
    if False:
        print('Hello World!')
    'Creates an object containing `signatures`.'
    signature_map = _SignatureMap()
    for (name, func) in signatures.items():
        assert isinstance(func, defun.ConcreteFunction)
        assert isinstance(func.structured_outputs, collections_abc.Mapping)
        if len(func._arg_keywords) == 1:
            assert 1 == func._num_positional_args
        else:
            assert 0 == func._num_positional_args
        signature_map._add_signature(name, func)
    return signature_map

def validate_augmented_graph_view(augmented_graph_view):
    if False:
        i = 10
        return i + 15
    'Performs signature-related sanity checks on `augmented_graph_view`.'
    for (name, dep) in augmented_graph_view.list_children(augmented_graph_view.root):
        if name == SIGNATURE_ATTRIBUTE_NAME:
            if not isinstance(dep, _SignatureMap):
                raise ValueError(f"Exporting an object {augmented_graph_view.root} which has an attribute named '{SIGNATURE_ATTRIBUTE_NAME}'. This is a reserved attribute used to store SavedModel signatures in objects which come from `tf.saved_model.load`. Delete this attribute (e.g. `del obj.{SIGNATURE_ATTRIBUTE_NAME}`) before saving if this shadowing is acceptable.")
            break