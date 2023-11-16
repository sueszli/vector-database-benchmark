"""Contains the base ProcessingLayer and a subclass that uses Combiners."""
import abc
import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as trackable

class PreprocessingLayer(Layer, metaclass=abc.ABCMeta):
    """Base class for Preprocessing Layers.

  **Don't use this class directly: it's an abstract base class!** You may
  be looking for one of the many built-in
  [preprocessing layers](https://keras.io/guides/preprocessing_layers/)
  instead.

  Preprocessing layers are layers whose state gets computed before model
  training starts. They do not get updated during training.
  Most preprocessing layers implement an `adapt()` method for state computation.

  The `PreprocessingLayer` class is the base class you would subclass to
  implement your own preprocessing layers.

  Attributes:
    streaming: Whether a layer can be adapted multiple times without resetting
      the state of the layer.
  """
    _must_restore_from_config = True

    def __init__(self, streaming=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(PreprocessingLayer, self).__init__(**kwargs)
        self._streaming = streaming
        self._is_compiled = False
        self._is_adapted = False
        self._reset_state_impl = self.reset_state
        self.reset_state = self._reset_state_wrapper
        self._adapt_function = None

    @property
    def streaming(self):
        if False:
            return 10
        'Whether `adapt` can be called twice without resetting the state.'
        return self._streaming

    @property
    def is_adapted(self):
        if False:
            i = 10
            return i + 15
        'Whether the layer has been fit to data already.'
        return self._is_adapted

    def update_state(self, data):
        if False:
            i = 10
            return i + 15
        'Accumulates statistics for the preprocessing layer.\n\n    Arguments:\n      data: A mini-batch of inputs to the layer.\n    '
        raise NotImplementedError

    def reset_state(self):
        if False:
            while True:
                i = 10
        'Resets the statistics of the preprocessing layer.'
        raise NotImplementedError

    def merge_state(self, layers):
        if False:
            print('Hello World!')
        'Merge the statistics of multiple preprocessing layers.\n\n    This layer will contain the merged state.\n\n    Arguments:\n      layers: Layers whose statistics should be merge with the statistics of\n        this layer.\n    '
        raise NotImplementedError

    def finalize_state(self):
        if False:
            while True:
                i = 10
        "Finalize the statistics for the preprocessing layer.\n\n    This method is called at the end of `adapt` or after restoring a serialized\n    preprocessing layer's state. This method handles any one-time operations\n    that should occur on the layer's state before `Layer.__call__`.\n    "
        pass

    def make_adapt_function(self):
        if False:
            for i in range(10):
                print('nop')
        'Creates a function to execute one step of `adapt`.\n\n    This method can be overridden to support custom adapt logic.\n    This method is called by `PreprocessingLayer.adapt`.\n\n    Typically, this method directly controls `tf.function` settings,\n    and delegates the actual state update logic to\n    `PreprocessingLayer.update_state`.\n\n    This function is cached the first time `PreprocessingLayer.adapt`\n    is called. The cache is cleared whenever `PreprocessingLayer.compile`\n    is called.\n\n    Returns:\n      Function. The function created by this method should accept a\n      `tf.data.Iterator`, retrieve a batch, and update the state of the\n      layer.\n    '
        if self._adapt_function is not None:
            return self._adapt_function

        def adapt_step(iterator):
            if False:
                for i in range(10):
                    print('nop')
            data = next(iterator)
            self._adapt_maybe_build(data)
            self.update_state(data)
        if self._steps_per_execution.numpy().item() == 1:
            adapt_fn = adapt_step
        else:

            def adapt_fn(iterator):
                if False:
                    while True:
                        i = 10
                for _ in math_ops.range(self._steps_per_execution):
                    adapt_step(iterator)
        if not self._run_eagerly:
            adapt_fn = def_function.function(adapt_fn)
        self._adapt_function = adapt_fn
        return self._adapt_function

    def compile(self, run_eagerly=None, steps_per_execution=None):
        if False:
            print('Hello World!')
        "Configures the layer for `adapt`.\n\n    Arguments:\n      run_eagerly: Bool. Defaults to `False`. If `True`, this `Model`'s logic\n        will not be wrapped in a `tf.function`. Recommended to leave this as\n        `None` unless your `Model` cannot be run inside a `tf.function`.\n        steps_per_execution: Int. Defaults to 1. The number of batches to run\n          during each `tf.function` call. Running multiple batches inside a\n          single `tf.function` call can greatly improve performance on TPUs or\n          small models with a large Python overhead.\n    "
        if steps_per_execution is None:
            steps_per_execution = 1
        self._configure_steps_per_execution(steps_per_execution)
        if run_eagerly is None:
            run_eagerly = self.dynamic
        self._run_eagerly = run_eagerly
        self._is_compiled = True

    def adapt(self, data, batch_size=None, steps=None, reset_state=True):
        if False:
            for i in range(10):
                print('nop')
        "Fits the state of the preprocessing layer to the data being passed.\n\n    After calling `adapt` on a layer, a preprocessing layer's state will not\n    update during training. In order to make preprocessing layers efficient in\n    any distribution context, they are kept constant with respect to any\n    compiled `tf.Graph`s that call the layer. This does not affect the layer use\n    when adapting each layer only once, but if you adapt a layer multiple times\n    you will need to take care to re-compile any compiled functions as follows:\n\n     * If you are adding a preprocessing layer to a `keras.Model`, you need to\n       call `model.compile` after each subsequent call to `adapt`.\n     * If you are calling a preprocessing layer inside `tf.data.Dataset.map`,\n       you should call `map` again on the input `tf.data.Dataset` after each\n       `adapt`.\n     * If you are using a `tf.function` directly which calls a preprocessing\n       layer, you need to call `tf.function` again on your callable after\n       each subsequent call to `adapt`.\n\n    `tf.keras.Model` example with multiple adapts:\n\n    >>> layer = tf.keras.layers.experimental.preprocessing.Normalization(\n    ...     axis=None)\n    >>> layer.adapt([0, 2])\n    >>> model = tf.keras.Sequential(layer)\n    >>> model.predict([0, 1, 2])\n    array([-1.,  0.,  1.], dtype=float32)\n    >>> layer.adapt([-1, 1])\n    >>> model.compile() # This is needed to re-compile model.predict!\n    >>> model.predict([0, 1, 2])\n    array([0., 1., 2.], dtype=float32)\n\n    `tf.data.Dataset` example with multiple adapts:\n\n    >>> layer = tf.keras.layers.experimental.preprocessing.Normalization(\n    ...     axis=None)\n    >>> layer.adapt([0, 2])\n    >>> input_ds = tf.data.Dataset.range(3)\n    >>> normalized_ds = input_ds.map(layer)\n    >>> list(normalized_ds.as_numpy_iterator())\n    [array([-1.], dtype=float32),\n     array([0.], dtype=float32),\n     array([1.], dtype=float32)]\n    >>> layer.adapt([-1, 1])\n    >>> normalized_ds = input_ds.map(layer) # Re-map over the input dataset.\n    >>> list(normalized_ds.as_numpy_iterator())\n    [array([0.], dtype=float32),\n     array([1.], dtype=float32),\n     array([2.], dtype=float32)]\n\n    Arguments:\n        data: The data to train on. It can be passed either as a tf.data\n          Dataset, or as a numpy array.\n        batch_size: Integer or `None`.\n            Number of samples per state update.\n            If unspecified, `batch_size` will default to 32.\n            Do not specify the `batch_size` if your data is in the\n            form of datasets, generators, or `keras.utils.Sequence` instances\n            (since they generate batches).\n        steps: Integer or `None`.\n            Total number of steps (batches of samples)\n            When training with input tensors such as\n            TensorFlow data tensors, the default `None` is equal to\n            the number of samples in your dataset divided by\n            the batch size, or 1 if that cannot be determined. If x is a\n            `tf.data` dataset, and 'steps' is None, the epoch will run until\n            the input dataset is exhausted. When passing an infinitely\n            repeating dataset, you must specify the `steps` argument. This\n            argument is not supported with array inputs.\n        reset_state: Optional argument specifying whether to clear the state of\n          the layer at the start of the call to `adapt`, or whether to start\n          from the existing state. This argument may not be relevant to all\n          preprocessing layers: a subclass of PreprocessingLayer may choose to\n          throw if 'reset_state' is set to False.\n    "
        _disallow_inside_tf_function('adapt')
        if not version_utils.should_use_v2():
            raise RuntimeError('`adapt` is only supported in tensorflow v2.')
        if not self.streaming and self._is_adapted and (not reset_state):
            raise ValueError('{} does not supporting calling `adapt` twice without resetting the state.'.format(self.__class__.__name__))
        if not self._is_compiled:
            self.compile()
        if self.built and reset_state:
            self.reset_state()
        data_handler = data_adapter.DataHandler(data, batch_size=batch_size, steps_per_epoch=steps, epochs=1, steps_per_execution=self._steps_per_execution, distribute=False)
        self._adapt_function = self.make_adapt_function()
        for (_, iterator) in data_handler.enumerate_epochs():
            with data_handler.catch_stop_iteration():
                for _ in data_handler.steps():
                    self._adapt_function(iterator)
                    if data_handler.should_sync:
                        context.async_wait()
        self.finalize_state()
        self._is_adapted = True

    def _reset_state_wrapper(self):
        if False:
            while True:
                i = 10
        'Calls `reset_state` and sets `adapted` to `False`.'
        self._reset_state_impl()
        self._is_adapted = False

    @trackable.no_automatic_dependency_tracking
    def _configure_steps_per_execution(self, steps_per_execution):
        if False:
            return 10
        self._steps_per_execution = variables.Variable(steps_per_execution, dtype='int64', aggregation=variables.VariableAggregationV2.ONLY_FIRST_REPLICA)

    def _adapt_maybe_build(self, data):
        if False:
            i = 10
            return i + 15
        if not self.built:
            try:
                data_shape = data.shape
                data_shape_nones = tuple([None] * len(data.shape))
            except AttributeError:
                data_shape = None
                data_shape_nones = None
            batch_input_shape = getattr(self, '_batch_input_shape', None)
            if batch_input_shape is None:
                self._batch_input_shape = data_shape_nones
            self.build(data_shape)
            self.built = True

class CombinerPreprocessingLayer(PreprocessingLayer):
    """Base class for PreprocessingLayers that do computation using a Combiner.

  This class provides several helper methods to make creating a
  PreprocessingLayer easier. It assumes that the core of your computation will
  be done via a Combiner object. Subclassing this class to create a
  PreprocessingLayer allows your layer to be compatible with distributed
  computation.

  This class is compatible with Tensorflow 2.0+.
  """

    def __init__(self, combiner, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(CombinerPreprocessingLayer, self).__init__(**kwargs)
        self.state_variables = collections.OrderedDict()
        self._combiner = combiner
        self._adapt_accumulator = None

    def reset_state(self):
        if False:
            return 10
        self._adapt_accumulator = None

    @trackable.no_automatic_dependency_tracking
    def update_state(self, data):
        if False:
            while True:
                i = 10
        if self._adapt_accumulator is None:
            self._adapt_accumulator = self._get_accumulator()
        self._adapt_accumulator = self._combiner.compute(data, self._adapt_accumulator)

    def merge_state(self, layers):
        if False:
            return 10
        accumulators = [self._get_accumulator()] + [l._get_accumulator() for l in layers]
        merged_accumulator = self._combiner.merge(accumulators)
        self._set_accumulator(merged_accumulator)

    def finalize_state(self):
        if False:
            return 10
        if self._adapt_accumulator is not None:
            self._set_accumulator(self._adapt_accumulator)

    def compile(self, run_eagerly=None, steps_per_execution=None):
        if False:
            while True:
                i = 10
        if run_eagerly is None:
            run_eagerly = True
        super(CombinerPreprocessingLayer, self).compile(run_eagerly=run_eagerly, steps_per_execution=steps_per_execution)

    def adapt(self, data, batch_size=None, steps=None, reset_state=True):
        if False:
            while True:
                i = 10
        if not reset_state:
            self._adapt_accumulator = self._combiner.restore(self._restore_updates())
        super(CombinerPreprocessingLayer, self).adapt(data, batch_size=batch_size, steps=steps, reset_state=reset_state)

    def _add_state_variable(self, name, shape, dtype, initializer=None, partitioner=None, use_resource=None, **kwargs):
        if False:
            return 10
        'Add a variable that can hold state which is updated during adapt().\n\n    Args:\n      name: Variable name.\n      shape: Variable shape. Defaults to scalar if unspecified.\n      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.\n      initializer: initializer instance (callable).\n      partitioner: Partitioner to be passed to the `Trackable` API.\n      use_resource: Whether to use `ResourceVariable`\n      **kwargs: Additional keyword arguments. Accepted values are `getter` and\n        `collections`.\n\n    Returns:\n      The created variable.\n    '
        weight = self.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, regularizer=None, trainable=False, constraint=None, partitioner=partitioner, use_resource=use_resource, **kwargs)
        self.state_variables[name] = weight
        return weight

    def _restore_updates(self):
        if False:
            while True:
                i = 10
        "Recreates a dict of updates from the layer's weights."
        data_dict = {}
        for (name, var) in self.state_variables.items():
            data_dict[name] = var.numpy()
        return data_dict

    def _get_accumulator(self):
        if False:
            for i in range(10):
                print('nop')
        if self._is_adapted:
            return self._combiner.restore(self._restore_updates())
        else:
            return None

    def _set_accumulator(self, accumulator):
        if False:
            i = 10
            return i + 15
        updates = self._combiner.extract(accumulator)
        self._set_state_variables(updates)
        self._adapt_accumulator = None

    def _set_state_variables(self, updates):
        if False:
            for i in range(10):
                print('nop')
        "Directly update the internal state of this Layer.\n\n    This method expects a string-keyed dict of {state_variable_name: state}. The\n    precise nature of the state, and the names associated, are describe by\n    the subclasses of CombinerPreprocessingLayer.\n\n    Args:\n      updates: A string keyed dict of weights to update.\n\n    Raises:\n      RuntimeError: if 'build()' was not called before 'set_processing_state'.\n    "
        if not self.built:
            raise RuntimeError('_set_state_variables() must be called after build().')
        with ops.init_scope():
            for (var_name, value) in updates.items():
                self.state_variables[var_name].assign(value)

def convert_to_list(values, sparse_default_value=None):
    if False:
        return 10
    'Convert a TensorLike, CompositeTensor, or ndarray into a Python list.'
    if tf_utils.is_ragged(values):
        if isinstance(values, ragged_tensor.RaggedTensor) and (not context.executing_eagerly()):
            values = backend.get_session(values).run(values)
        values = values.to_list()
    if isinstance(values, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
        if sparse_default_value is None:
            if dtypes.as_dtype(values.values.dtype) == dtypes.string:
                sparse_default_value = ''
            else:
                sparse_default_value = -1
        dense_tensor = sparse_ops.sparse_tensor_to_dense(values, default_value=sparse_default_value)
        values = backend.get_value(dense_tensor)
    if isinstance(values, tensor.Tensor):
        values = backend.get_value(values)
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return values

class Combiner(object):
    """Functional object that defines a shardable computation.

  This object defines functions required to create and manipulate data objects.
  These data objects, referred to below as 'accumulators', are computation-
  specific and may be implemented alongside concrete subclasses of Combiner
  (if necessary - some computations may be simple enough that standard Python
  types can be used as accumulators).

  The intent for this class is that by describing computations in this way, we
  can arbitrarily shard a dataset, perform computations on a subset, and then
  merge the computation into a final result. This enables distributed
  computation.

  The combiner itself does not own any state - all computational state is owned
  by the accumulator objects. This is so that we can have an arbitrary number of
  Combiners (thus sharding the computation N ways) without risking any change
  to the underlying computation. These accumulator objects are uniquely
  associated with each Combiner; a Combiner defines what the accumulator object
  should be and will only work with accumulators of that type.
  """
    __metaclass__ = abc.ABCMeta

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<{}>'.format(self.__class__.__name__)

    @abc.abstractmethod
    def compute(self, batch_values, accumulator=None):
        if False:
            print('Hello World!')
        'Compute a step in this computation, returning a new accumulator.\n\n    This method computes a step of the computation described by this Combiner.\n    If an accumulator is passed, the data in that accumulator is also used; so\n    compute(batch_values) results in f(batch_values), while\n    compute(batch_values, accumulator) results in\n    merge(f(batch_values), accumulator).\n\n    Args:\n      batch_values: A list of ndarrays representing the values of the inputs for\n        this step of the computation.\n      accumulator: the current accumulator. Can be None.\n\n    Returns:\n      An accumulator that includes the passed batch of inputs.\n    '
        pass

    @abc.abstractmethod
    def merge(self, accumulators):
        if False:
            while True:
                i = 10
        'Merge several accumulators to a single accumulator.\n\n    This method takes the partial values in several accumulators and combines\n    them into a single accumulator. This computation must not be order-specific\n    (that is, merge([a, b]) must return the same result as merge([b, a]).\n\n    Args:\n      accumulators: the accumulators to merge, as a list.\n\n    Returns:\n      A merged accumulator.\n    '
        pass

    @abc.abstractmethod
    def extract(self, accumulator):
        if False:
            i = 10
            return i + 15
        'Convert an accumulator into a dict of output values.\n\n    Args:\n      accumulator: The accumulator to convert.\n\n    Returns:\n      A dict of ndarrays representing the data in this accumulator.\n    '
        pass

    @abc.abstractmethod
    def restore(self, output):
        if False:
            print('Hello World!')
        "Create an accumulator based on 'output'.\n\n    This method creates a new accumulator with identical internal state to the\n    one used to create the data in 'output'. This means that if you do\n\n    output_data = combiner.extract(accumulator_1)\n    accumulator_2 = combiner.restore(output_data)\n\n    then accumulator_1 and accumulator_2 will have identical internal state, and\n    computations using either of them will be equivalent.\n\n    Args:\n      output: The data output from a previous computation. Should be in the same\n        form as provided by 'extract_output'.\n\n    Returns:\n      A new accumulator.\n    "
        pass

    @abc.abstractmethod
    def serialize(self, accumulator):
        if False:
            for i in range(10):
                print('nop')
        'Serialize an accumulator for a remote call.\n\n    This function serializes an accumulator to be sent to a remote process.\n\n    Args:\n      accumulator: The accumulator to serialize.\n\n    Returns:\n      A byte string representing the passed accumulator.\n    '
        pass

    @abc.abstractmethod
    def deserialize(self, encoded_accumulator):
        if False:
            print('Hello World!')
        "Deserialize an accumulator received from 'serialize()'.\n\n    This function deserializes an accumulator serialized by 'serialize()'.\n\n    Args:\n      encoded_accumulator: A byte string representing an accumulator.\n\n    Returns:\n      The accumulator represented by the passed byte_string.\n    "
        pass

def _disallow_inside_tf_function(method_name):
    if False:
        return 10
    'Disallow calling a method inside a `tf.function`.'
    if ops.inside_function():
        error_msg = 'Detected a call to `PreprocessingLayer.{method_name}` inside a `tf.function`. `PreprocessingLayer.{method_name} is a high-level endpoint that manages its own `tf.function`. Please move the call to `PreprocessingLayer.{method_name}` outside of all enclosing `tf.function`s. Note that you can call a `PreprocessingLayer` directly on `Tensor`s inside a `tf.function` like: `layer(x)`, or update its state like: `layer.update_state(x)`.'.format(method_name=method_name)
        raise RuntimeError(error_msg)