"""Input layer code (`Input` and `InputLayer`)."""
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import tf_utils

def _assert_other_arg_none(arg_name, arg):
    if False:
        return 10
    if arg is not None:
        raise ValueError('When `type_spec` is not None, all other args except `name` must be None, but %s is not None.' % arg_name)

class InputLayer(base_layer.Layer):
    """Layer to be used as an entry point into a Network (a graph of layers).

  It can either wrap an existing tensor (pass an `input_tensor` argument)
  or create a placeholder tensor (pass arguments `input_shape`, and
  optionally, `dtype`).

  It is generally recommend to use the functional layer API via `Input`,
  (which creates an `InputLayer`) without directly using `InputLayer`.

  When using InputLayer with Keras Sequential model, it can be skipped by
  moving the input_shape parameter to the first layer after the InputLayer.

  This class can create placeholders for tf.Tensors, tf.SparseTensors, and
  tf.RaggedTensors by choosing 'sparse=True' or 'ragged=True'. Note that
  'sparse' and 'ragged' can't be configured to True at same time.
  Usage:

  ```python
  # With explicit InputLayer.
  model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(4,)),
    tf.keras.layers.Dense(8)])
  model.compile(tf.optimizers.RMSprop(0.001), loss='mse')
  model.fit(np.zeros((10, 4)),
            np.ones((10, 8)))

  # Without InputLayer and let the first layer to have the input_shape.
  # Keras will add a input for the model behind the scene.
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(4,))])
  model.compile(tf.optimizers.RMSprop(0.001), loss='mse')
  model.fit(np.zeros((10, 4)),
            np.ones((10, 8)))
  ```

  Args:
      input_shape: Shape tuple (not including the batch axis), or `TensorShape`
        instance (not including the batch axis).
      batch_size: Optional input batch size (integer or None).
      dtype: Optional datatype of the input. When not provided, the Keras
          default float type will be used.
      input_tensor: Optional tensor to use as layer input. If set, the layer
          will use the `tf.TypeSpec` of this tensor rather
          than creating a new placeholder tensor.
      sparse: Boolean, whether the placeholder created is meant to be sparse.
          Default to False.
      ragged: Boolean, whether the placeholder created is meant to be ragged.
          In this case, values of 'None' in the 'shape' argument represent
          ragged dimensions. For more information about RaggedTensors, see
          [this guide](https://www.tensorflow.org/guide/ragged_tensors).
          Default to False.
      type_spec: A `tf.TypeSpec` object to create Input from. This `tf.TypeSpec`
          represents the entire batch. When provided, all other args except
          name must be None.
      name: Optional name of the layer (string).
  """

    def __init__(self, input_shape=None, batch_size=None, dtype=None, input_tensor=None, sparse=None, name=None, ragged=None, type_spec=None, **kwargs):
        if False:
            i = 10
            return i + 15
        self._init_input_shape = input_shape
        self._init_batch_size = batch_size
        self._init_dtype = dtype
        self._init_sparse = sparse
        self._init_ragged = ragged
        self._init_type_spec = type_spec
        strategy = distribute_lib.get_strategy()
        if strategy and batch_size is not None and distributed_training_utils.global_batch_size_supported(strategy):
            if batch_size % strategy.num_replicas_in_sync != 0:
                raise ValueError('The `batch_size` argument ({}) must be divisible by the number of replicas ({})'.format(batch_size, strategy.num_replicas_in_sync))
            batch_size = batch_size // strategy.num_replicas_in_sync
        if 'batch_input_shape' in kwargs:
            batch_input_shape = kwargs.pop('batch_input_shape')
            if input_shape and batch_input_shape:
                raise ValueError('Only provide the input_shape OR batch_input_shape argument to InputLayer, not both at the same time.')
            if batch_input_shape:
                batch_size = batch_input_shape[0]
                input_shape = batch_input_shape[1:]
        if kwargs:
            raise ValueError('Unrecognized keyword arguments:', kwargs.keys())
        if sparse and ragged:
            raise ValueError('Cannot set both sparse and ragged to True in a Keras input.')
        if not name:
            prefix = 'input'
            name = prefix + '_' + str(backend.get_uid(prefix))
        if not dtype:
            if input_tensor is None:
                dtype = backend.floatx()
            else:
                dtype = backend.dtype(input_tensor)
        elif input_tensor is not None and input_tensor.dtype != dtype:
            raise ValueError('`input_tensor.dtype` differs from `dtype`: %s vs. %s' % (input_tensor.dtype, dtype))
        super(InputLayer, self).__init__(dtype=dtype, name=name)
        self.built = True
        self.sparse = True if sparse else False
        self.ragged = True if ragged else False
        self.batch_size = batch_size
        self.supports_masking = True
        if isinstance(input_shape, tensor_shape.TensorShape):
            input_shape = tuple(input_shape.as_list())
        elif isinstance(input_shape, int):
            input_shape = (input_shape,)
        if type_spec is not None:
            args_that_must_be_none = [('(input_)shape', self._init_input_shape), ('batch_size', self._init_batch_size), ('dtype', self._init_dtype), ('input_tensor', input_tensor), ('sparse', self._init_sparse), ('ragged', self._init_ragged)]
            for (arg_name, arg) in args_that_must_be_none:
                _assert_other_arg_none(arg_name, arg)
            if not ops.executing_eagerly_outside_functions():
                raise ValueError('Creating Keras inputs from a type_spec is only supported when eager execution is enabled.')
            input_tensor = keras_tensor.keras_tensor_from_type_spec(type_spec)
            if isinstance(input_tensor, keras_tensor.SparseKerasTensor):
                self.sparse = True
            if isinstance(input_tensor, keras_tensor.RaggedKerasTensor):
                self.ragged = True
            self.is_placeholder = True
            try:
                self._batch_input_shape = tuple(input_tensor.shape.as_list())
            except ValueError:
                self._batch_input_shape = None
        elif input_tensor is None:
            if input_shape is not None:
                batch_input_shape = (batch_size,) + tuple(input_shape)
            else:
                batch_input_shape = None
            graph = backend.get_graph()
            with graph.as_default():
                input_tensor = backend.placeholder(shape=batch_input_shape, dtype=dtype, name=self.name, sparse=sparse, ragged=ragged)
            self.is_placeholder = True
            self._batch_input_shape = batch_input_shape
        else:
            if ops.executing_eagerly_outside_functions():
                if not isinstance(input_tensor, keras_tensor.KerasTensor):
                    input_tensor = keras_tensor.keras_tensor_from_tensor(input_tensor)
            elif not tf_utils.is_symbolic_tensor(input_tensor):
                raise ValueError('You should not pass an EagerTensor to `Input`. For example, instead of creating an InputLayer, you should instantiate your model and directly call it on your input.')
            self.is_placeholder = False
            try:
                self._batch_input_shape = tuple(input_tensor.shape.as_list())
            except ValueError:
                self._batch_input_shape = None
        input_tensor._keras_mask = None
        node_module.Node(layer=self, outputs=input_tensor)
        if isinstance(input_tensor, keras_tensor.KerasTensor) or tf_utils.is_extension_type(input_tensor):
            self._type_spec = input_tensor._type_spec
        else:
            self._type_spec = tensor_spec.TensorSpec(shape=input_tensor.shape, dtype=input_tensor.dtype, name=self.name)

    def get_config(self):
        if False:
            i = 10
            return i + 15
        if self._init_type_spec is not None:
            config = {'name': self.name, 'type_spec': self._init_type_spec}
        else:
            config = {'batch_input_shape': self._batch_input_shape, 'dtype': self.dtype, 'sparse': self.sparse, 'ragged': self.ragged, 'name': self.name}
        return config

    @property
    def _trackable_saved_model_saver(self):
        if False:
            for i in range(10):
                print('nop')
        return layer_serialization.InputLayerSavedModelSaver(self)

def Input(shape=None, batch_size=None, name=None, dtype=None, sparse=None, tensor=None, ragged=None, type_spec=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "`Input()` is used to instantiate a Keras tensor.\n\n  A Keras tensor is a symbolic tensor-like object,\n  which we augment with certain attributes that allow us to build a Keras model\n  just by knowing the inputs and outputs of the model.\n\n  For instance, if `a`, `b` and `c` are Keras tensors,\n  it becomes possible to do:\n  `model = Model(input=[a, b], output=c)`\n\n  Args:\n      shape: A shape tuple (integers), not including the batch size.\n          For instance, `shape=(32,)` indicates that the expected input\n          will be batches of 32-dimensional vectors. Elements of this tuple\n          can be None; 'None' elements represent dimensions where the shape is\n          not known.\n      batch_size: optional static batch size (integer).\n      name: An optional name string for the layer.\n          Should be unique in a model (do not reuse the same name twice).\n          It will be autogenerated if it isn't provided.\n      dtype: The data type expected by the input, as a string\n          (`float32`, `float64`, `int32`...)\n      sparse: A boolean specifying whether the placeholder to be created is\n          sparse. Only one of 'ragged' and 'sparse' can be True. Note that,\n          if `sparse` is False, sparse tensors can still be passed into the\n          input - they will be densified with a default value of 0.\n      tensor: Optional existing tensor to wrap into the `Input` layer.\n          If set, the layer will use the `tf.TypeSpec` of this tensor rather\n          than creating a new placeholder tensor.\n      ragged: A boolean specifying whether the placeholder to be created is\n          ragged. Only one of 'ragged' and 'sparse' can be True. In this case,\n          values of 'None' in the 'shape' argument represent ragged dimensions.\n          For more information about RaggedTensors, see\n          [this guide](https://www.tensorflow.org/guide/ragged_tensors).\n      type_spec: A `tf.TypeSpec` object to create the input placeholder from.\n          When provided, all other args except name must be None.\n      **kwargs: deprecated arguments support. Supports `batch_shape` and\n          `batch_input_shape`.\n\n  Returns:\n    A `tensor`.\n\n  Example:\n\n  ```python\n  # this is a logistic regression in Keras\n  x = Input(shape=(32,))\n  y = Dense(16, activation='softmax')(x)\n  model = Model(x, y)\n  ```\n\n  Note that even if eager execution is enabled,\n  `Input` produces a symbolic tensor-like object (i.e. a placeholder).\n  This symbolic tensor-like object can be used with lower-level\n  TensorFlow ops that take tensors as inputs, as such:\n\n  ```python\n  x = Input(shape=(32,))\n  y = tf.square(x)  # This op will be treated like a layer\n  model = Model(x, y)\n  ```\n\n  (This behavior does not work for higher-order TensorFlow APIs such as\n  control flow and being directly watched by a `tf.GradientTape`).\n\n  However, the resulting model will not track any variables that were\n  used as inputs to TensorFlow ops. All variable usages must happen within\n  Keras layers to make sure they will be tracked by the model's weights.\n\n  The Keras Input can also create a placeholder from an arbitrary `tf.TypeSpec`,\n  e.g:\n\n  ```python\n  x = Input(type_spec=tf.RaggedTensorSpec(shape=[None, None],\n                                          dtype=tf.float32, ragged_rank=1))\n  y = x.values\n  model = Model(x, y)\n  ```\n  When passing an arbitrary `tf.TypeSpec`, it must represent the signature of an\n  entire batch instead of just one example.\n\n  Raises:\n    ValueError: If both `sparse` and `ragged` are provided.\n    ValueError: If both `shape` and (`batch_input_shape` or `batch_shape`) are\n      provided.\n    ValueError: If `shape`, `tensor` and `type_spec` are None.\n    ValueError: If arguments besides `type_spec` are non-None while `type_spec`\n                is passed.\n    ValueError: if any unrecognized parameters are provided.\n  "
    if sparse and ragged:
        raise ValueError('Cannot set both sparse and ragged to True in a Keras input.')
    input_layer_config = {'name': name, 'dtype': dtype, 'sparse': sparse, 'ragged': ragged, 'input_tensor': tensor, 'type_spec': type_spec}
    batch_input_shape = kwargs.pop('batch_input_shape', kwargs.pop('batch_shape', None))
    if shape is not None and batch_input_shape is not None:
        raise ValueError('Only provide the `shape` OR `batch_input_shape` argument to Input, not both at the same time.')
    if batch_input_shape is None and shape is None and (tensor is None) and (type_spec is None):
        raise ValueError('Please provide to Input a `shape` or a `tensor` or a `type_spec` argument. Note that `shape` does not include the batch dimension.')
    if kwargs:
        raise ValueError('Unrecognized keyword arguments:', kwargs.keys())
    if batch_input_shape:
        shape = batch_input_shape[1:]
        input_layer_config.update({'batch_input_shape': batch_input_shape})
    else:
        input_layer_config.update({'batch_size': batch_size, 'input_shape': shape})
    input_layer = InputLayer(**input_layer_config)
    outputs = input_layer._inbound_nodes[0].outputs
    if isinstance(outputs, list) and len(outputs) == 1:
        return outputs[0]
    else:
        return outputs