"""A Keras Layer for using TF Hub modules in TF2 format."""
import functools
import json
from absl import logging
import tensorflow as tf
from tensorflow_hub import module_v2
version_fn = getattr(tf.keras, 'version', None)
if version_fn and version_fn().startswith('3.'):
    import tf_keras as keras
else:
    keras = tf.keras
from tensorflow.python.framework import smart_cond
from tensorflow.python.util import tf_inspect
try:
    from tensorflow.python.trackable import data_structures
except ImportError:
    from tensorflow.python.training.tracking import data_structures

class KerasLayer(keras.layers.Layer):
    """Wraps a SavedModel (or a legacy TF1 Hub format) as a Keras Layer.

  This layer wraps a callable object for use as a Keras layer. The callable
  object can be passed directly, or be specified by a Python string with a
  handle that gets passed to `hub.load()`.

  This is the preferred API to load a TF2-style SavedModel from TF Hub
  into a Keras model. Calling this function requires TF 1.15 or newer.
  It can be called both in eager and graph mode.

  The callable object is expected to follow the conventions detailed below.
  (These are met by TF2-compatible modules loaded from TensorFlow Hub.)

  The callable is invoked with a single positional argument set to one tensor
  or a nest of tensors containing the inputs to the layer. If the callable
  accepts a `training` argument, a Python boolean is passed for it. It is True
  if this layer is marked trainable *and* called for training, analogous to
  keras.layers.BatchNormalization. (By contrast, keras.layers.Dropout
  ignores the trainable state and applies the training argument verbatim.)

  If present, the following attributes of callable are understood to have
  special meanings:
    variables: a list of all tf.Variable objects that the callable depends on.
    trainable_variables: those elements of `variables` that are reported
      as trainable variables of this Keras Layer when the layer is trainable.
    regularization_losses: a list of callables to be added as losses of this
      Keras Layer when the layer is trainable. Each one must accept zero
      arguments and return a scalar tensor.

  Note: to work-around missing shape inference functionalities from functions
  created from FunctionDefs, in rare cases one has to pass an 'output_shape'
  and potentially 'input_shape' and 'dtype'. E.g. the following is a typical
  work-around:
  ```
  hub.KerasLayer(
      "/tmp/text_embedding_model",
      output_shape=[20],  # Outputs a tensor with shape [batch_size, 20].
      input_shape=[],     # Expects a tensor of shape [batch_size] as input.
      dtype=tf.string)    # Expects a tf.string input tensor.
  ```

  Note: This layer can be used inside the model_fn of a TF2 Estimator. See the
  [migration guide]
  (https://www.tensorflow.org/beta/guide/migration_guide#using_a_custom_model_fn)
  for guidance on how to pick up trainable variables, losses and updates
  explicitly from Keras objects instead of relying on graph collections.
  This layer class does not support graph collections.
  Distributed training of the Estimator requires setting the option
  `session_config.share_cluster_devices_in_session` within the
  `tf.estimator.RunConfig`. (This option was experimental from TF1.14 to TF2.1.)

  Note: The data types used by a saved model have been fixed at saving time.
  Using keras.mixed_precision etc. has no effect on the saved model
  that gets loaded by a hub.KerasLayer.

  Attributes:
    handle: A callable object (subject to the conventions above), or a Python
      string to load a saved model via hub.load(). A string is required to save
      the Keras config of this Layer.
    trainable: Optional. A boolean controlling whether this layer is trainable.
      Must not be set to True when using a signature (raises ValueError),
      including the use of legacy TF1 Hub format.
    arguments: Optional. A dict with additional keyword arguments passed to the
      callable. These must be JSON-serializable to save the Keras config of this
      layer, and are not tracked as checkpointing dependencies of this layer.
    _sentinel: Used to prevent further positional arguments.
    tags: Optional. If set indicates which graph variant to use. For legacy
      models in TF1 Hub format leaving unset means to use the empty tags set.
    signature: Optional. If set, KerasLayer will use the requested signature.
      For legacy models in TF1 Hub format leaving unset means to use the
      `default` signature. When using a signature, either
      signature_outputs_as_dict or output_key have to set.
    signature_outputs_as_dict: If set to True, the call to this layer returns a
      dict of all the signature outputs. Can only be used if a signature is
      specified (or default signature is used for legacy models in TF1 Hub
      format).
    output_key: Name of the output item to return if the layer returns a dict.
      For legacy models in TF1 Hub format leaving unset means to return the
      `default` output.
    output_shape: A tuple or a nest of tuples with the (possibly partial) output
      shapes of the callable *without* leading batch size. This must have the
      same nesting structure as the output of the callable object and cover all
      output tensors.
    load_options: Optional, `tf.saved_model.LoadOptions` object that specifies
      options for loading when a Python string is provided as `handle`. This
      argument can only be used from TensorFlow 2.3 onwards.
    **kwargs: Forwarded to Keras' base Layer constructor.
  """

    def __init__(self, handle, trainable=False, arguments=None, _sentinel=None, tags=None, signature=None, signature_outputs_as_dict=None, output_key=None, output_shape=None, load_options=None, **kwargs):
        if False:
            while True:
                i = 10
        self._handle = handle
        self._arguments = data_structures.NoDependency(arguments or {})
        self._signature = signature
        self._signature_outputs_as_dict = signature_outputs_as_dict
        self._output_key = output_key
        if output_shape:
            self._output_shape = data_structures.NoDependency(_convert_nest_to_shapes(output_shape))
        self._load_options = load_options
        self._func = load_module(handle, tags, self._load_options)
        self._is_hub_module_v1 = getattr(self._func, '_is_hub_module_v1', False)
        if self._is_hub_module_v1:
            self._signature = self._signature or 'default'
            if not self._signature_outputs_as_dict:
                self._output_key = self._output_key or 'default'
        if self._signature and bool(self._output_key is not None) == bool(self._signature_outputs_as_dict):
            raise ValueError('When using a signature, either output_key or signature_outputs_as_dict=True should be set.')
        if not self._signature and self._signature_outputs_as_dict:
            raise ValueError('signature_outputs_as_dict is only valid if specifying a signature (or using a legacy TF1 Hub format).')
        self._callable = self._get_callable()
        self._has_training_argument = func_has_training_argument(self._callable)
        self._setup_layer(trainable, **kwargs)

    def _setup_layer(self, trainable=False, **kwargs):
        if False:
            print('Hello World!')
        'Constructs keras layer with relevant weights and losses.'
        super().__init__(trainable=trainable, **kwargs)
        if hasattr(self._func, 'trainable_variables'):
            for v in self._func.trainable_variables:
                self._add_existing_weight(v, trainable=True)
            trainable_variables = {id(v) for v in self._func.trainable_variables}
        else:
            trainable_variables = set()
        if hasattr(self._func, 'variables'):
            for v in self._func.variables:
                if id(v) not in trainable_variables:
                    self._add_existing_weight(v, trainable=False)
        if hasattr(self._func, 'regularization_losses'):
            for l in self._func.regularization_losses:
                if not callable(l):
                    raise ValueError('hub.KerasLayer(obj) expects obj.regularization_losses to be an iterable of callables, each returning a scalar loss term.')
                self.add_loss(self._call_loss_if_trainable(l))

    def _add_existing_weight(self, weight, trainable=None):
        if False:
            print('Hello World!')
        'Calls add_weight() to register but not create an existing weight.'
        if trainable is None:
            trainable = weight.trainable
        self.add_weight(name=weight.name, shape=weight.shape, dtype=weight.dtype, trainable=trainable, experimental_autocast=False, getter=lambda *_, **__: weight)

    def _call_loss_if_trainable(self, loss):
        if False:
            print('Hello World!')
        'Returns `loss` conditioned on whether this layer is trainable.'
        return lambda : loss() if self.trainable else 0.0

    def call(self, inputs, training=None):
        if False:
            for i in range(10):
                print('nop')
        self._check_trainability()
        args = []
        kwargs = self._arguments.copy()
        if self._signature and isinstance(inputs, dict):
            kwargs.update(inputs)
        else:
            args.append(inputs)
        f = functools.partial(self._callable, *args, **kwargs)
        if not self._has_training_argument:
            result = f()
        else:
            if self.trainable:
                if training is None:
                    training = keras.backend.learning_phase()
            else:
                training = False
            result = smart_cond.smart_cond(training, lambda : f(training=True), lambda : f(training=False))
        if self._output_key:
            if not isinstance(result, dict):
                raise ValueError('Specifying `output_key` is forbidden if output type %s is not a dict.' % type(result))
            if self._output_key not in result:
                raise ValueError('KerasLayer output does not contain the output key %s (available: %s).' % (self._output_key, result.keys()))
            result = result[self._output_key]
        result = self._apply_output_shape_if_set(inputs, result)
        return result

    def _check_trainability(self):
        if False:
            while True:
                i = 10
        'Raises or logs errors for unuspported uses of trainable=True.'
        if not self.trainable:
            return
        if self._is_hub_module_v1:
            raise ValueError('Setting hub.KerasLayer.trainable = True is unsupported when loading from the TF1 Hub format.')
        elif self._signature:
            raise ValueError('Setting hub.KerasLayer.trainable = True is unsupported when calling a SavedModel signature.')
        if not self.trainable_weights:
            if not hasattr(self, '_already_logged_trainable_with_zero_weights'):
                logging.error('hub.KerasLayer is trainable but has zero trainable weights.')
                setattr(self, '_already_logged_trainable_with_zero_weights', True)

    def _get_callable(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a callable object.'
        if callable(self._func) and (not self._signature):
            return self._func
        if not hasattr(self._func, 'signatures'):
            if self._signature:
                raise ValueError('Loaded object has no signatures.')
            else:
                raise ValueError('Loaded object is not callable and has no signatures.')
        if self._signature is None:
            raise ValueError('Signature name has to be specified for non-callable saved models (if not legacy TF1 Hub format).')
        if self._signature not in self._func.signatures:
            raise ValueError('Unknown signature %s in %s (available signatures: %s).' % (self._signature, self._handle, self._func.signatures))
        f = self._func.signatures[self._signature]
        if not callable(f):
            raise ValueError('Internal error: signature %s is not callable in %s' % (self._signature, self._handle))
        return f

    def _apply_output_shape_if_set(self, inputs, outputs):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, '_output_shape'):
            return outputs
        output_shape = getattr(self, '_output_shape')
        batch_size = tf.nest.flatten(inputs)[0].shape[0]

        def _inplace_set_shape(tensor, shape):
            if False:
                print('Hello World!')
            tensor.set_shape(tf.TensorShape(batch_size).concatenate(shape))
        tf.nest.map_structure(_inplace_set_shape, outputs, output_shape)
        return outputs

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a serializable dict of keras layer configuration parameters.'
        config = super().get_config()
        if not isinstance(self._handle, str):
            raise NotImplementedError('Can only generate a valid config for `hub.KerasLayer(handle, ...)`that uses a string `handle`.\n\nGot `type(handle)`: {}'.format(type(self._handle)))
        config['handle'] = self._handle
        if hasattr(self, '_output_shape'):
            output_shape = _convert_nest_from_shapes(self._output_shape)
            try:
                json.dumps(output_shape)
            except TypeError:
                raise ValueError('hub.KerasLayer(..., output_shape=) is not json-serializable.\nGot value: {}'.format(output_shape))
            config['output_shape'] = output_shape
        if self._arguments:
            for (key, value) in self._arguments.items():
                try:
                    json.dumps(value)
                except TypeError:
                    raise ValueError('`hub.KerasLayer(..., arguments)` contains non json-serializablevalues in key: {}'.format(key))
            config['arguments'] = self._arguments
        if self._signature:
            config['signature'] = self._signature
        if self._output_key:
            config['output_key'] = self._output_key
        if self._signature_outputs_as_dict:
            config['signature_outputs_as_dict'] = self._signature_outputs_as_dict
        return config

    @property
    def resolved_object(self):
        if False:
            i = 10
            return i + 15
        'Returns the callable object to which `handle` resolved in `__init__`.'
        return self._func

    def compute_output_shape(self, input_shape):
        if False:
            return 10
        'Computes the output shape of the layer.\n\n    This relies on the `output_shape` provided during initialization, if any,\n    else falls back to the default behavior from `keras.layers.Layer`.\n\n    Args:\n      input_shape: Shape tuple (tuple of integers) or list of shape tuples (one\n        per output tensor of the layer). Shape tuples can include None for free\n        dimensions, instead of an integer.\n\n    Returns:\n      An input shape tuple.\n    '
        if hasattr(self, '_output_shape'):
            output_shape = getattr(self, '_output_shape')
            batch_size = tf.nest.flatten(input_shape)[0]
            return tf.TensorShape((batch_size,)).concatenate(output_shape)
        return super(KerasLayer, self).compute_output_shape(input_shape)

def _convert_nest_to_shapes(x):
    if False:
        for i in range(10):
            print('nop')
    'In a nest, converts raw tuples/lists of int or None to tf.TensorShape.'
    if isinstance(x, dict):
        return type(x)([(k, _convert_nest_to_shapes(v)) for (k, v) in x.items()])
    try:
        return tf.TensorShape(x)
    except TypeError:
        pass
    if isinstance(x, (list, tuple)):
        return type(x)([_convert_nest_to_shapes(v) for v in x])
    else:
        raise TypeError('Cannot convert to nest of TensorShapes, found none of TensorShape, dict, list, tuple: %r' % x)

def _convert_nest_from_shapes(x):
    if False:
        return 10
    'Converts a nest of tf.TensorShape to raw tuples of int or None.'

    def _shape_as_tuple(x):
        if False:
            return 10
        assert isinstance(x, tf.TensorShape)
        return tuple(x.as_list())
    return tf.nest.map_structure(_shape_as_tuple, x)

def load_module(handle, tags=None, load_options=None):
    if False:
        while True:
            i = 10
    if callable(handle):
        if tags is not None:
            raise ValueError('Passing a callable handle is mutually exclusive with setting tags.')
        if load_options is not None:
            raise ValueError('Passing a callable handle is mutually exclusive with setting load_options.')
        return handle
    else:
        try:
            from keras.saving.legacy.saved_model import load_context
            set_load_options = load_options or load_context.get_load_options()
        except ImportError:
            try:
                from tensorflow.keras.saving.saved_model import load_context
                set_load_options = load_options or load_context.get_load_options()
            except ImportError:
                try:
                    from tensorflow.python.saved_model import load_context
                    set_load_options = load_options or load_context.get_load_options()
                except ImportError:
                    set_load_options = load_options
        return module_v2.load(handle, tags=tags, options=set_load_options)

def func_has_training_argument(func):
    if False:
        for i in range(10):
            print('nop')
    'Checks whether saved model has a `training` argument.'
    if not callable(func):
        return False
    fullargspec = tf_inspect.getfullargspec(func.__call__)
    return 'training' in fullargspec.args or 'training' in fullargspec.kwonlyargs