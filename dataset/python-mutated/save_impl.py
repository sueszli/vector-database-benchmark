"""Keras SavedModel serialization.

TODO (kathywu): Move to layer_serialization.py. Some model-specific logic should
go to model_serialization.py.
"""
import functools
import threading
import weakref
from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import load as keras_load
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
base_layer = LazyLoader('base_layer', globals(), 'tensorflow.python.keras.engine.base_layer')
metrics = LazyLoader('metrics', globals(), 'tensorflow.python.keras.metrics')
input_layer = LazyLoader('input_layer', globals(), 'tensorflow.python.keras.engine.input_layer')
training_lib = LazyLoader('training_lib', globals(), 'tensorflow.python.keras.engine.training')
sequential_lib = LazyLoader('sequential_lib', globals(), 'tensorflow.python.keras.engine.sequential')

def should_skip_serialization(layer):
    if False:
        return 10
    "Skip serializing extra objects and functions if layer inputs aren't set."
    saved_model_input_spec_set = isinstance(layer, training_lib.Model) and layer._saved_model_inputs_spec is not None
    if not layer.built and (not saved_model_input_spec_set):
        logging.warning('Skipping full serialization of Keras layer {}, because it is not built.'.format(layer))
        return True
    return False

def wrap_layer_objects(layer, serialization_cache):
    if False:
        i = 10
        return i + 15
    'Returns extra trackable objects to attach to the serialized layer.\n\n  Args:\n    layer: Keras Layer object.\n    serialization_cache: Dictionary shared between all objects during\n      serialization.\n\n  Returns:\n    A dictionary containing all checkpointable objects from a\n    SerializedAttributes object. See LayerAttributes and ModelAttributes for\n    entire list of objects\n  '
    all_losses = layer._callable_losses[:]
    for child_layer in utils.list_all_layers(layer):
        all_losses.extend(child_layer._callable_losses)
    keras_loss_cache = serialization_cache.setdefault('keras_losses', {})
    wrapped_loss_functions = []
    for loss_fn in all_losses:
        if loss_fn in keras_loss_cache:
            wrapped_loss_functions.append(keras_loss_cache[loss_fn])
        else:
            wrapped_loss = _wrap_unconditional_loss(loss_fn, len(keras_loss_cache))
            keras_loss_cache[loss_fn] = wrapped_loss
            wrapped_loss_functions.append(wrapped_loss)
    wrapped_layer_losses = [keras_loss_cache[fn] for fn in layer._callable_losses[:]]
    layer_metrics = data_structures.wrap_or_unwrap({m.name: m for m in layer._metrics})
    return dict(variables=data_structures.wrap_or_unwrap(layer.variables), trainable_variables=data_structures.wrap_or_unwrap(layer.trainable_variables), non_trainable_variables=data_structures.wrap_or_unwrap(layer.non_trainable_variables), layers=data_structures.wrap_or_unwrap(utils.list_all_layers(layer)), metrics=data_structures.wrap_or_unwrap(layer.metrics), regularization_losses=data_structures.wrap_or_unwrap(wrapped_loss_functions), layer_regularization_losses=data_structures.wrap_or_unwrap(wrapped_layer_losses), layer_metrics=layer_metrics)

def wrap_layer_functions(layer, serialization_cache):
    if False:
        return 10
    'Returns dict of wrapped layer call function and losses in tf.functions.\n\n  Args:\n    layer: Keras Layer object.\n    serialization_cache: Dictionary shared between all objects during\n      serialization.\n\n  Returns:\n    A dictionary containing all keras tf.functions to serialize. See\n    LayerAttributes and ModelAttributes for the list of all attributes.\n  '
    if isinstance(layer, keras_load.RevivedLayer) and (not isinstance(layer, sequential_lib.Sequential)):
        return {fn_name: getattr(layer.keras_api, fn_name, None) for fn_name in serialized_attributes.LayerAttributes.all_functions}
    original_fns = _replace_child_layer_functions(layer, serialization_cache)
    original_losses = _reset_layer_losses(layer)
    call_collection = LayerCallCollection(layer)
    call_fn_with_losses = call_collection.add_function(_wrap_call_and_conditional_losses(layer), '{}_layer_call_and_return_conditional_losses'.format(layer.name), match_layer_training_arg=True)
    call_fn = call_collection.add_function(_extract_outputs_from_fn(layer, call_fn_with_losses), '{}_layer_call_fn'.format(layer.name), match_layer_training_arg=False)
    fns = {'call_and_return_conditional_losses': call_fn_with_losses, '__call__': call_fn}
    if layer._activity_regularizer is not None:
        fns['activity_regularizer_fn'] = _wrap_activity_regularizer(layer)
        fns['call_and_return_all_conditional_losses'] = call_collection.add_function(_append_activity_regularizer_loss(layer, call_fn_with_losses, fns['activity_regularizer_fn']), '{}_layer_call_and_return_all_conditional_losses'.format(layer.name), match_layer_training_arg=False)
    else:
        fns['activity_regularizer_fn'] = None
        fns['call_and_return_all_conditional_losses'] = call_fn_with_losses
    with tracing_scope():
        call_collection.trace_with_input_signature()
        with base_layer_utils.call_context().enter(layer, inputs=None, build_graph=True, training=None, saving=True):
            for fn in fns.values():
                if fn is not None and fn.input_signature is not None:
                    if isinstance(fn, LayerCall):
                        fn = fn.wrapped_call
                    fn.get_concrete_function()
    _restore_child_layer_functions(original_fns)
    _restore_layer_losses(original_losses)
    return fns

def default_save_signature(layer):
    if False:
        for i in range(10):
            print('nop')
    original_losses = _reset_layer_losses(layer)
    fn = saving_utils.trace_model_call(layer)
    fn.get_concrete_function()
    _restore_layer_losses(original_losses)
    return fn

def _replace_child_layer_functions(layer, serialization_cache):
    if False:
        for i in range(10):
            print('nop')
    "Replaces functions in the children layers with wrapped tf.functions.\n\n  This step allows functions from parent layers to reference the wrapped\n  functions from their children layers instead of retracing the ops.\n\n  This function also resets all losses stored in the layer. These are stored in\n  the returned dictionary. Use `_restore_child_layer_functions` to restore\n  the original attributes.\n\n  Args:\n    layer: Keras Layer object.\n    serialization_cache: Dictionary shared between all objects during\n      serialization.\n\n  Returns:\n    Dictionary mapping layer objects -> original functions and losses:\n      { Child layer 1: {\n          'losses': Original losses,\n          'call': Original call function\n          '_activity_regularizer': Original activity regularizer},\n        Child layer 2: ...\n      }\n  "
    original_fns = {}

    def replace_layer_functions(child_layer, serialized_fns):
        if False:
            while True:
                i = 10
        'Replaces layer call and activity regularizer with wrapped functions.'
        original_fns[child_layer] = {'call': child_layer.call, '_activity_regularizer': child_layer._activity_regularizer}
        with utils.no_automatic_dependency_tracking_scope(child_layer):
            try:
                child_layer._activity_regularizer = serialized_fns.get('activity_regularizer_fn')
            except AttributeError:
                pass
            child_layer.call = utils.use_wrapped_call(child_layer, serialized_fns['call_and_return_conditional_losses'], default_training_value=False)

    def replace_metric_functions(child_layer, serialized_fns):
        if False:
            for i in range(10):
                print('nop')
        'Replaces metric functions with wrapped functions.'
        original_fns[child_layer] = {'__call__': child_layer.__call__, 'result': child_layer.result, 'update_state': child_layer.update_state}
        with utils.no_automatic_dependency_tracking_scope(child_layer):
            child_layer.__call__ = serialized_fns['__call__']
            child_layer.result = serialized_fns['result']
            child_layer.update_state = serialized_fns['update_state']
    for child_layer in utils.list_all_layers(layer):
        if isinstance(child_layer, input_layer.InputLayer):
            continue
        if child_layer not in serialization_cache[constants.KERAS_CACHE_KEY]:
            serialized_functions = child_layer._trackable_saved_model_saver._get_serialized_attributes(serialization_cache).functions
        else:
            serialized_functions = serialization_cache[constants.KERAS_CACHE_KEY][child_layer].functions
        if not serialized_functions:
            continue
        if isinstance(child_layer, metrics.Metric):
            replace_metric_functions(child_layer, serialized_functions)
        else:
            replace_layer_functions(child_layer, serialized_functions)
    return original_fns

def _restore_child_layer_functions(original_fns):
    if False:
        for i in range(10):
            print('nop')
    'Restores attributes replaced with `_replace_child_layer_functions`.'
    for (child_layer, fns) in original_fns.items():
        with utils.no_automatic_dependency_tracking_scope(child_layer):
            for (fn_name, fn) in fns.items():
                try:
                    setattr(child_layer, fn_name, fn)
                except AttributeError:
                    pass

def _reset_layer_losses(parent_layer):
    if False:
        for i in range(10):
            print('nop')
    'Resets losses of layer and its sublayers, and returns original losses.'
    losses_dict = {}
    for layer in utils.list_all_layers_and_sublayers(parent_layer):
        losses_dict[layer] = {'losses': layer._losses[:], 'eager_losses': layer._eager_losses[:]}
        with utils.no_automatic_dependency_tracking_scope(layer):
            layer._losses = []
            layer._eager_losses = []
    return losses_dict

def _restore_layer_losses(losses_dict):
    if False:
        while True:
            i = 10
    for layer in losses_dict:
        with utils.no_automatic_dependency_tracking_scope(layer):
            layer._losses = losses_dict[layer]['losses']
            layer._eager_losses = losses_dict[layer]['eager_losses']

class LayerTracingContext(threading.local):

    def __init__(self):
        if False:
            return 10
        super(LayerTracingContext, self).__init__()
        self.enable_call_tracing = False
        self.trace_queue = []
_thread_local_data = LayerTracingContext()

@tf_contextlib.contextmanager
def tracing_scope():
    if False:
        print('Hello World!')
    'Enables tracing scope.'
    previous_value = _thread_local_data.enable_call_tracing
    previous_queue = _thread_local_data.trace_queue
    try:
        _thread_local_data.enable_call_tracing = True
        _thread_local_data.trace_queue = []
        yield
    finally:
        while _thread_local_data.trace_queue:
            (fn, args, kwargs, training) = _thread_local_data.trace_queue.pop()
            if training is not None:
                with K.deprecated_internal_learning_phase_scope(training):
                    fn.get_concrete_function(*args, **kwargs)
            else:
                fn.get_concrete_function(*args, **kwargs)
        _thread_local_data.trace_queue = previous_queue
        _thread_local_data.enable_call_tracing = previous_value

def add_trace_to_queue(fn, args, kwargs, training=None):
    if False:
        return 10
    if tracing_enabled():
        _thread_local_data.trace_queue.append((fn, args[:], kwargs.copy(), training))

def tracing_enabled():
    if False:
        i = 10
        return i + 15
    'Whether to add extra traces to the queue.'
    return _thread_local_data.enable_call_tracing

class LayerCallCollection(object):
    """Groups wrapped layer call functions.

  This is used to ensure that all layer call functions are traced with the same
  inputs-
    - call
    - call_and_return_conditional_losses
    - call_and_return_all_conditional_losses
  """

    def __init__(self, layer):
        if False:
            return 10
        self.layer = layer
        self.layer_call_method = _get_layer_call_method(layer)
        self._expects_training_arg = utils.layer_uses_training_bool(layer)
        self._training_arg_index = utils.get_training_arg_index(self.layer_call_method)
        arg_spec = tf_inspect.getfullargspec(self.layer_call_method)
        self._has_kwargs = bool(self._expects_training_arg or arg_spec.defaults or arg_spec.kwonlyargs or arg_spec.varkw)
        self._input_signature = self._generate_input_signature(layer)
        self._functions = weakref.WeakValueDictionary()
        args = arg_spec.args
        if tf_inspect.ismethod(self.layer_call_method):
            args = args[1:]
        self._input_arg_name = args[0] if args else 'inputs'

    def _generate_input_signature(self, layer):
        if False:
            for i in range(10):
                print('nop')
        'Inspects layer object and returns the inferred input signature.\n\n    Args:\n      layer: Layer object.\n\n    Returns:\n      List of possibly nested TensorSpecs of the layer call function inputs.\n      The list does not contain the `training` argument.\n    '
        if isinstance(layer.call, def_function.Function) and layer.call.input_signature is not None:
            return layer.call.input_signature
        elif isinstance(layer, training_lib.Model):
            return saving_utils.model_input_signature(layer)
        elif layer.input_spec is not None and layer._use_input_spec_as_call_signature:

            def to_tensor_spec_or_none(x):
                if False:
                    for i in range(10):
                        print('nop')
                spec = input_spec.to_tensor_spec(x, layer._compute_dtype)
                if spec.shape == tensor_shape.TensorShape(None):
                    return None
                return spec
            input_signature = [nest.map_structure(to_tensor_spec_or_none, layer.input_spec)]
            return input_signature
        else:
            return None

    def add_trace(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Traces all functions with the same args and kwargs.\n\n    Args:\n      *args: Positional args passed to the original function.\n      **kwargs: Keyword args passed to the original function.\n    '
        args = list(args)
        kwargs = kwargs.copy()
        for fn in self._functions.values():
            if self._expects_training_arg:

                def trace_with_training(value, fn=fn):
                    if False:
                        while True:
                            i = 10
                    utils.set_training_arg(value, self._training_arg_index, args, kwargs)
                    add_trace_to_queue(fn, args, kwargs, value)
                trace_with_training(True)
                trace_with_training(False)
            else:
                add_trace_to_queue(fn, args, kwargs)

    @property
    def fn_input_signature(self):
        if False:
            return 10
        'Returns input signature for the wrapped layer call function.'
        if self._has_kwargs:
            return None
        if None in nest.flatten(self._input_signature):
            return None
        return self._input_signature

    def training_arg_was_passed(self, args, kwargs):
        if False:
            while True:
                i = 10
        if not self.layer._expects_training_arg and self._expects_training_arg:
            return utils.get_training_arg(self._training_arg_index, args, kwargs) is not None
        else:
            return self.layer._call_arg_was_passed('training', args, kwargs, inputs_in_args=True)

    def get_training_arg_value(self, args, kwargs):
        if False:
            return 10
        if not self.layer._expects_training_arg and self._expects_training_arg:
            return utils.get_training_arg(self._training_arg_index, args, kwargs)
        else:
            return self.layer._get_call_arg_value('training', args, kwargs, inputs_in_args=True)

    def get_input_arg_value(self, args, kwargs):
        if False:
            while True:
                i = 10
        return self.layer._get_call_arg_value(self._input_arg_name, args, kwargs, inputs_in_args=True)

    def _maybe_wrap_with_training_arg(self, call_fn, match_layer_training_arg):
        if False:
            for i in range(10):
                print('nop')
        'Wraps call function with added training argument if necessary.'
        if not self.layer._expects_training_arg and self._expects_training_arg:
            arg_spec = tf_inspect.getfullargspec(call_fn)
            args = arg_spec.args + ['training']
            defaults = list(arg_spec.defaults or [])
            defaults.append(False)
            new_arg_spec = tf_inspect.FullArgSpec(args=args, varargs=arg_spec.varargs, varkw=arg_spec.varkw, defaults=defaults, kwonlyargs=arg_spec.kwonlyargs, kwonlydefaults=arg_spec.kwonlydefaults, annotations=arg_spec.annotations)
            self._training_arg_index = len(args) - 1
            if tf_inspect.ismethod(call_fn):
                self._training_arg_index -= 1

            def wrap_with_training_arg(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                if match_layer_training_arg:
                    args = list(args)
                    kwargs = kwargs.copy()
                    utils.remove_training_arg(self._training_arg_index, args, kwargs)
                return call_fn(*args, **kwargs)
            return tf_decorator.make_decorator(target=call_fn, decorator_func=wrap_with_training_arg, decorator_argspec=new_arg_spec)
        return call_fn

    def add_function(self, call_fn, name, match_layer_training_arg):
        if False:
            for i in range(10):
                print('nop')
        'Adds a layer call function to the collection.\n\n    Args:\n      call_fn: a python function\n      name: Name of call function\n      match_layer_training_arg: If True, removes the `training` from the\n        function arguments when calling `call_fn`.\n\n    Returns:\n      LayerCall (tf.function)\n    '
        fn = LayerCall(self, self._maybe_wrap_with_training_arg(call_fn, match_layer_training_arg), name, input_signature=self.fn_input_signature)
        self._functions[name] = fn.wrapped_call
        return fn

    def trace_with_input_signature(self):
        if False:
            print('Hello World!')
        'Trace with the layer/models inferred input signature if possible.'
        if None not in nest.flatten(self._input_signature) and self._has_kwargs:
            self.add_trace(*self._input_signature)

def _filtered_inputs(inputs):
    if False:
        i = 10
        return i + 15
    return list(filter(tf_utils.is_tensor_or_variable, nest.flatten(inputs)))

def layer_call_wrapper(call_collection, method, name):
    if False:
        print('Hello World!')
    'Ensures layer losses are kept the same, and runs method in call context.'

    def wrapper(*args, **kwargs):
        if False:
            while True:
                i = 10
        'Calls method within call context.'
        layer = call_collection.layer
        training = None
        inputs = _filtered_inputs([args, kwargs])
        if (args or kwargs) and call_collection.training_arg_was_passed(args, kwargs):
            training = call_collection.get_training_arg_value(args, kwargs)
        original_losses = _reset_layer_losses(layer)
        with base_layer_utils.call_context().enter(layer, inputs=inputs, build_graph=False, training=training, saving=True):
            with autocast_variable.enable_auto_cast_variables(layer._compute_dtype_object):
                ret = method(*args, **kwargs)
        _restore_layer_losses(original_losses)
        return ret
    fn = tf_decorator.make_decorator(target=method, decorator_func=wrapper)
    fn.__name__ = name
    return fn

class LayerCall(object):
    """Function that triggers traces of other functions in the same collection."""

    def __init__(self, call_collection, call_fn, name, input_signature):
        if False:
            while True:
                i = 10
        'Initializes a LayerCall object.\n\n    Args:\n      call_collection: a LayerCallCollection, which contains the other layer\n        call functions (e.g. call_with_conditional_losses, call). These\n        functions should be traced with the same arguments.\n      call_fn: A call function.\n      name: Name of the call function.\n      input_signature: Input signature of call_fn (can be None).\n    '
        self.call_collection = call_collection
        self.input_signature = input_signature
        self.wrapped_call = def_function.function(layer_call_wrapper(call_collection, call_fn, name), input_signature=input_signature)
        self.original_layer_call = call_collection.layer_call_method

    def _maybe_trace(self, args, kwargs):
        if False:
            i = 10
            return i + 15
        if tracing_enabled():
            self.call_collection.add_trace(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        self._maybe_trace(args, kwargs)
        return self.wrapped_call(*args, **kwargs)

    def get_concrete_function(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self._maybe_trace(args, kwargs)
        return self.wrapped_call.get_concrete_function(*args, **kwargs)

def _wrap_call_and_conditional_losses(layer):
    if False:
        for i in range(10):
            print('nop')
    'Wraps call function that returns a tuple of (outputs, losses).\n\n  The losses returned are conditional on the inputs passed to the call function.\n  Unconditional losses (e.g. weight regularizeration) are wrapped separately.\n\n  Args:\n    layer: a Keras layer object\n\n  Returns:\n    python call function that returns outputs and conditional losses -- excludes\n    activity regularizer\n  '
    layer_call = _get_layer_call_method(layer)

    def call_and_return_conditional_losses(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Returns layer (call_output, conditional losses) tuple.'
        call_output = layer_call(*args, **kwargs)
        if version_utils.is_v1_layer_or_model(layer):
            conditional_losses = layer.get_losses_for(_filtered_inputs([args, kwargs]))
        else:
            conditional_losses = [l for l in layer.losses if not hasattr(l, '_unconditional_loss')]
        return (call_output, conditional_losses)
    return _create_call_fn_decorator(layer, call_and_return_conditional_losses)

def _extract_outputs_from_fn(layer, call_and_return_conditional_losses):
    if False:
        return 10
    'Returns a function that returns only call function outputs.'
    if isinstance(layer, keras_load.RevivedLayer):
        return layer.keras_api.__call__

    def call(inputs, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return call_and_return_conditional_losses(inputs, *args, **kwargs)[0]
    return _create_call_fn_decorator(layer, call)

def _append_activity_regularizer_loss(layer, call_fn_with_losses, activity_regularizer_fn):
    if False:
        return 10
    'Appends activity regularizer loss to losses returned by the wrapped fn.'

    def fn(inputs, *args, **kwargs):
        if False:
            while True:
                i = 10
        (outputs, losses) = call_fn_with_losses(inputs, *args, **kwargs)
        losses.append(activity_regularizer_fn(outputs))
        return (outputs, losses)
    return _create_call_fn_decorator(layer, fn)

def _create_call_fn_decorator(layer, wrapped_call):
    if False:
        print('Hello World!')
    call_fn = _get_layer_call_method(layer)
    (fn, arg_spec) = utils.maybe_add_training_arg(call_fn, wrapped_call, layer._expects_training_arg, default_training_value=False)
    return tf_decorator.make_decorator(target=call_fn, decorator_func=fn, decorator_argspec=arg_spec)

def _wrap_unconditional_loss(loss_fn, index):
    if False:
        for i in range(10):
            print('nop')
    'Wraps callable/unconditional loss, returning a serializable function.'
    fn = loss_fn.args[0] if isinstance(loss_fn, functools.partial) else loss_fn
    if isinstance(fn, def_function.Function):
        return fn
    else:
        return def_function.Function(fn, 'loss_fn_{}'.format(index), input_signature=[])

def _wrap_activity_regularizer(layer):
    if False:
        return 10
    'Wraps the activity regularizer.'
    if isinstance(layer._activity_regularizer, def_function.Function):
        return layer._activity_regularizer
    return def_function.Function(layer._activity_regularizer, '{}_activity_regularizer'.format(layer.name), input_signature=[tensor_spec.TensorSpec(None, layer._compute_dtype or K.floatx())])

def _get_layer_call_method(layer):
    if False:
        return 10
    if isinstance(layer.call, def_function.Function):
        return layer.call.python_function
    return layer.call