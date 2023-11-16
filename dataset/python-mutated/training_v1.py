"""V1 Training-related part of the Keras engine."""
import collections
import warnings
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.distribute import distributed_training_utils_v1
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_arrays_v1
from tensorflow.python.keras.engine import training_distributed_v1
from tensorflow.python.keras.engine import training_eager_v1
from tensorflow.python.keras.engine import training_generator_v1
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
try:
    from scipy.sparse import issparse
except ImportError:
    issparse = None

class Model(training_lib.Model):
    """`Model` groups layers into an object with training and inference features.

  There are two ways to instantiate a `Model`:

  1 - With the "functional API", where you start from `Input`,
  you chain layer calls to specify the model's forward pass,
  and finally you create your model from inputs and outputs:

  ```python
  import tensorflow as tf

  inputs = tf.keras.Input(shape=(3,))
  x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
  outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  ```

  2 - By subclassing the `Model` class: in that case, you should define your
  layers in `__init__` and you should implement the model's forward pass
  in `call`.

  ```python
  import tensorflow as tf

  class MyModel(tf.keras.Model):

    def __init__(self):
      super(MyModel, self).__init__()
      self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
      self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    def call(self, inputs):
      x = self.dense1(inputs)
      return self.dense2(x)

  model = MyModel()
  ```

  If you subclass `Model`, you can optionally have
  a `training` argument (boolean) in `call`, which you can use to specify
  a different behavior in training and inference:

  ```python
  import tensorflow as tf

  class MyModel(tf.keras.Model):

    def __init__(self):
      super(MyModel, self).__init__()
      self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
      self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
      self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
      x = self.dense1(inputs)
      if training:
        x = self.dropout(x, training=training)
      return self.dense2(x)

  model = MyModel()
  ```
  """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(Model, self).__init__(*args, **kwargs)
        self._distribution_strategy = None
        self._compile_time_distribution_strategy = None
        if ops.executing_eagerly_outside_functions() and distribute_lib.has_strategy():
            self._set_strategy(distribute_lib.get_strategy())
        self._compile_distribution = False
        self._run_eagerly = None
        self._experimental_run_tf_function = ops.executing_eagerly_outside_functions()
        self._v1_compile_was_called = False

    def _init_batch_counters(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @trackable.no_automatic_dependency_tracking
    def _set_strategy(self, strategy):
        if False:
            for i in range(10):
                print('nop')
        self._compile_time_distribution_strategy = strategy

    def get_weights(self):
        if False:
            for i in range(10):
                print('nop')
        'Retrieves the weights of the model.\n\n    Returns:\n        A flat list of Numpy arrays.\n    '
        strategy = self._distribution_strategy or self._compile_time_distribution_strategy
        if strategy:
            with strategy.scope():
                return base_layer.Layer.get_weights(self)
        return base_layer.Layer.get_weights(self)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False):
        if False:
            print('Hello World!')
        "Loads all layer weights, either from a TensorFlow or an HDF5 weight file.\n\n    If `by_name` is False weights are loaded based on the network's\n    topology. This means the architecture should be the same as when the weights\n    were saved.  Note that layers that don't have weights are not taken into\n    account in the topological ordering, so adding or removing layers is fine as\n    long as they don't have weights.\n\n    If `by_name` is True, weights are loaded into layers only if they share the\n    same name. This is useful for fine-tuning or transfer-learning models where\n    some of the layers have changed.\n\n    Only topological loading (`by_name=False`) is supported when loading weights\n    from the TensorFlow format. Note that topological loading differs slightly\n    between TensorFlow and HDF5 formats for user-defined classes inheriting from\n    `tf.keras.Model`: HDF5 loads based on a flattened list of weights, while the\n    TensorFlow format loads based on the object-local names of attributes to\n    which layers are assigned in the `Model`'s constructor.\n\n    Args:\n        filepath: String, path to the weights file to load. For weight files in\n            TensorFlow format, this is the file prefix (the same as was passed\n            to `save_weights`).\n        by_name: Boolean, whether to load weights by name or by topological\n            order. Only topological loading is supported for weight files in\n            TensorFlow format.\n        skip_mismatch: Boolean, whether to skip loading of layers where there is\n            a mismatch in the number of weights, or a mismatch in the shape of\n            the weight (only valid when `by_name=True`).\n\n    Returns:\n        When loading a weight file in TensorFlow format, returns the same status\n        object as `tf.train.Checkpoint.restore`. When graph building, restore\n        ops are run automatically as soon as the network is built (on first call\n        for user-defined classes inheriting from `Model`, immediately if it is\n        already built).\n\n        When loading weights in HDF5 format, returns `None`.\n\n    Raises:\n        ImportError: If h5py is not available and the weight file is in HDF5\n            format.\n        ValueError: If `skip_mismatch` is set to `True` when `by_name` is\n          `False`.\n    "
        if backend.is_tpu_strategy(self._distribution_strategy):
            if self._distribution_strategy.extended.steps_per_run > 1 and (not saving_utils.is_hdf5_filepath(filepath)):
                raise ValueError('Load weights is not yet supported with TPUStrategy with steps_per_run greater than 1.')
        return super(Model, self).load_weights(filepath, by_name, skip_mismatch)

    @trackable.no_automatic_dependency_tracking
    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None, distribute=None, **kwargs):
        if False:
            return 10
        'Configures the model for training.\n\n    Args:\n        optimizer: String (name of optimizer) or optimizer instance.\n            See `tf.keras.optimizers`.\n        loss: String (name of objective function), objective function or\n            `tf.keras.losses.Loss` instance. See `tf.keras.losses`. An objective\n            function is any callable with the signature\n            `scalar_loss = fn(y_true, y_pred)`. If the model has multiple\n            outputs, you can use a different loss on each output by passing a\n            dictionary or a list of losses. The loss value that will be\n            minimized by the model will then be the sum of all individual\n            losses.\n        metrics: List of metrics to be evaluated by the model during training\n            and testing. Typically you will use `metrics=[\'accuracy\']`.\n            To specify different metrics for different outputs of a\n            multi-output model, you could also pass a dictionary, such as\n            `metrics={\'output_a\': \'accuracy\', \'output_b\': [\'accuracy\', \'mse\']}`.\n            You can also pass a list (len = len(outputs)) of lists of metrics\n            such as `metrics=[[\'accuracy\'], [\'accuracy\', \'mse\']]` or\n            `metrics=[\'accuracy\', [\'accuracy\', \'mse\']]`.\n        loss_weights: Optional list or dictionary specifying scalar\n            coefficients (Python floats) to weight the loss contributions\n            of different model outputs.\n            The loss value that will be minimized by the model\n            will then be the *weighted sum* of all individual losses,\n            weighted by the `loss_weights` coefficients.\n            If a list, it is expected to have a 1:1 mapping\n            to the model\'s outputs. If a tensor, it is expected to map\n            output names (strings) to scalar coefficients.\n        sample_weight_mode: If you need to do timestep-wise\n            sample weighting (2D weights), set this to `"temporal"`.\n            `None` defaults to sample-wise weights (1D).\n            If the model has multiple outputs, you can use a different\n            `sample_weight_mode` on each output by passing a\n            dictionary or a list of modes.\n        weighted_metrics: List of metrics to be evaluated and weighted\n            by sample_weight or class_weight during training and testing.\n        target_tensors: By default, Keras will create placeholders for the\n            model\'s target, which will be fed with the target data during\n            training. If instead you would like to use your own\n            target tensors (in turn, Keras will not expect external\n            Numpy data for these targets at training time), you\n            can specify them via the `target_tensors` argument. It can be\n            a single tensor (for a single-output model), a list of tensors,\n            or a dict mapping output names to target tensors.\n        distribute: NOT SUPPORTED IN TF 2.0, please create and compile the\n            model under distribution strategy scope instead of passing it to\n            compile.\n        **kwargs: Any additional arguments.\n\n    Raises:\n        ValueError: In case of invalid arguments for\n            `optimizer`, `loss`, `metrics` or `sample_weight_mode`.\n    '
        self._assert_built_as_v1()
        self._run_eagerly = kwargs.pop('run_eagerly', None)
        self._experimental_run_tf_function = kwargs.pop('experimental_run_tf_function', True)
        self._v1_compile_was_called = True
        kwargs.pop('cloning', None)
        self._from_serialized = kwargs.pop('from_serialized', False)
        allowed_kwargs = {'feed_dict', 'fetches', 'options', 'run_metadata'}
        unknown_kwargs = set(kwargs.keys()) - allowed_kwargs
        if unknown_kwargs:
            raise TypeError('Invalid keyword argument(s) in `compile`: %s' % (unknown_kwargs,))
        self._function_kwargs = kwargs
        if self._function_kwargs:
            self._experimental_run_tf_function = False
            if self.run_eagerly:
                raise ValueError('Session keyword arguments are not supported when `run_eagerly=True`. You passed the following Session arguments: %s' % (self._function_kwargs,))
        self._set_optimizer(optimizer)
        is_any_keras_optimizer_v1 = any((isinstance(opt, optimizer_v1.Optimizer) and (not isinstance(opt, optimizer_v1.TFOptimizer)) for opt in nest.flatten(self.optimizer)))
        if is_any_keras_optimizer_v1 and ops.executing_eagerly_outside_functions():
            raise ValueError('`tf.compat.v1.keras` Optimizer (', optimizer, ') is not supported when eager execution is enabled. Use a `tf.keras` Optimizer instead, or disable eager execution.')
        if target_tensors is not None or not ops.executing_eagerly_outside_functions():
            self._experimental_run_tf_function = False
        if distribute is not None:
            if tf2.enabled() or self._experimental_run_tf_function:
                raise ValueError('Distribute argument in compile is not available in TF 2.0 please create the model under the distribution strategy scope.')
            logging.warning('Distribute argument in compile is deprecated please create the model under the distribution strategy scope.')
            self._distribution_strategy = distribute
            self._compile_distribution = True
        elif distribute_lib.has_strategy():
            if distribute_lib.in_cross_replica_context():
                self._distribution_strategy = distribute_lib.get_strategy()
        if isinstance(self._distribution_strategy, parameter_server_strategy.ParameterServerStrategyV1):
            raise NotImplementedError('`tf.compat.v1.distribute.experimental.ParameterServerStrategy` currently only works with the tf.Estimator API')
        if isinstance(self._distribution_strategy, parameter_server_strategy_v2.ParameterServerStrategyV2):
            raise NotImplementedError('`tf.distribute.experimental.ParameterServerStrategy` is only supported in TF2.')
        if not self._experimental_run_tf_function:
            self._validate_compile_param_for_distribution_strategy(self.run_eagerly, sample_weight_mode, target_tensors, weighted_metrics)
        if isinstance(self.optimizer, trackable.Trackable):
            self._track_trackable(self.optimizer, name='optimizer', overwrite=True)
        self.loss = loss or {}
        self.loss_weights = loss_weights
        self.sample_weight_mode = sample_weight_mode
        self._compile_metrics = metrics or []
        self._compile_weighted_metrics = weighted_metrics
        if self.run_eagerly and target_tensors is not None:
            raise ValueError('target_tensors argument is not supported when running a model eagerly.')
        self._training_endpoints = []
        self._compiled_trainable_state = self._get_trainable_state()
        self._distributed_model_cache = {}
        self._distributed_function_cache = {}
        self._clear_losses()
        if not context.executing_eagerly() and self._distribution_strategy is not None:
            backend.configure_and_create_distributed_session(self._distribution_strategy)
        self._init_metric_attributes()
        if not self.built or not self.inputs or (not self.outputs):
            return
        self._is_compiled = True
        self.loss_functions = training_utils_v1.prepare_loss_functions(self.loss, self.output_names)
        target_tensors = self._process_target_tensor_for_compile(target_tensors)
        for (o, n, l, t) in zip(self.outputs, self.output_names, self.loss_functions, target_tensors):
            endpoint = _TrainingEndpoint(o, n, l)
            endpoint.create_training_target(t, run_eagerly=self.run_eagerly)
            self._training_endpoints.append(endpoint)
        training_utils_v1.prepare_loss_weights(self._training_endpoints, loss_weights)
        if self.run_eagerly:
            self._compile_eagerly(metrics, weighted_metrics, sample_weight_mode)
            return
        with backend.get_graph().as_default():
            self._cache_output_metric_attributes(metrics, weighted_metrics)
            self._set_metric_attributes()
            self._handle_metrics(self.outputs, targets=self._targets, skip_target_masks=self._prepare_skip_target_masks(), masks=self._prepare_output_masks())
            training_utils_v1.prepare_sample_weight_modes(self._training_endpoints, sample_weight_mode)
            self._compile_weights_loss_and_weighted_metrics()
            self.train_function = None
            self.test_function = None
            self.predict_function = None
            self._collected_trainable_weights = self.trainable_weights
            if self._distribution_strategy and (not self._compile_distribution):
                for v in self.variables:
                    strategy = self._distribution_strategy
                    if not strategy.extended.variable_created_in_scope(v):
                        raise ValueError('Variable (%s) was not created in the distribution strategy scope of (%s). It is most likely due to not all layers or the model or optimizer being created outside the distribution strategy scope. Try to make sure your code looks similar to the following.\nwith strategy.scope():\n  model=_create_model()\n  model.compile(...)' % (v, strategy))

    @trackable.no_automatic_dependency_tracking
    def _init_distributed_function_cache_if_not_compiled(self):
        if False:
            return 10
        if not hasattr(self, '_distributed_function_cache'):
            self._distributed_function_cache = {}

    @property
    def metrics(self):
        if False:
            print('Hello World!')
        "Returns the model's metrics added using `compile`, `add_metric` APIs."
        metrics = []
        if self._is_compiled:
            if not hasattr(self, '_v1_compile_was_called'):
                return super(Model, self).metrics
            metrics += self._compile_metric_functions
        metrics.extend(self._metrics)
        metrics.extend(_get_metrics_from_layers(list(self._flatten_layers(include_self=False, recursive=False))))
        return metrics

    @property
    def metrics_names(self):
        if False:
            print('Hello World!')
        "Returns the model's display labels for all outputs."
        metrics_names = ['loss']
        if self._is_compiled:
            if not hasattr(self, '_v1_compile_was_called'):
                return super(Model, self).metrics_names
            if len(self._training_endpoints) > 1:
                metrics_names.extend([e.loss_name() for e in self._training_endpoints if not e.should_skip_target()])
        metrics_names += [m.name for m in self.metrics]
        return metrics_names

    @property
    def run_eagerly(self):
        if False:
            return 10
        'Settable attribute indicating whether the model should run eagerly.\n\n    Running eagerly means that your model will be run step by step,\n    like Python code. Your model might run slower, but it should become easier\n    for you to debug it by stepping into individual layer calls.\n\n    By default, we will attempt to compile your model to a static graph to\n    deliver the best execution performance.\n\n    Returns:\n      Boolean, whether the model should run eagerly.\n    '
        if self._run_eagerly is True and (not context.executing_eagerly()):
            raise ValueError('You can only set `run_eagerly=True` if eager execution is enabled.')
        if not self.dynamic:
            if self._run_eagerly is None:
                return def_function.functions_run_eagerly()
            else:
                return self._run_eagerly
        else:
            if not context.executing_eagerly():
                raise ValueError('Your model contains layers that can only be successfully run in eager execution (layers constructed with `dynamic=True`). You must enable eager execution with `tf.enable_eager_execution()`.')
            if self._run_eagerly is False:
                raise ValueError('Your model contains layers that can only be successfully run in eager execution (layers constructed with `dynamic=True`). You cannot set `run_eagerly=False`.')
            return context.executing_eagerly()

    @run_eagerly.setter
    def run_eagerly(self, value):
        if False:
            return 10
        self._run_eagerly = value

    def _select_training_loop(self, inputs):
        if False:
            i = 10
            return i + 15
        'Select training loop for fit/eval/predict based on the inputs.'
        if isinstance(inputs, (iterator_ops.Iterator, iterator_ops.IteratorBase)):
            raise ValueError('For performance reasons Keras `fit`, `evaluate` and`predict` accept tf.data `Datasets` as input but not iterators that have been manually generated from Datasets by users. Please directly pass in the original `Dataset` object instead of passing in `iter(dataset)`.')
        if self._distribution_strategy:
            if self._in_multi_worker_mode():
                return training_distributed_v1.DistributionMultiWorkerTrainingLoop(training_distributed_v1.DistributionSingleWorkerTrainingLoop())
            else:
                return training_distributed_v1.DistributionSingleWorkerTrainingLoop()
        if data_utils.is_generator_or_sequence(inputs):
            return training_generator_v1.GeneratorOrSequenceTrainingLoop()
        if training_utils_v1.is_eager_dataset_or_iterator(inputs):
            return training_generator_v1.EagerDatasetOrIteratorTrainingLoop()
        if self.run_eagerly:
            return training_generator_v1.GeneratorLikeTrainingLoop()
        else:
            return training_arrays_v1.ArrayLikeTrainingLoop()

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False, **kwargs):
        if False:
            while True:
                i = 10
        'Trains the model for a fixed number of epochs (iterations on a dataset).\n\n    Args:\n        x: Input data. It could be:\n          - A Numpy array (or array-like), or a list of arrays\n            (in case the model has multiple inputs).\n          - A TensorFlow tensor, or a list of tensors\n            (in case the model has multiple inputs).\n          - A dict mapping input names to the corresponding array/tensors,\n            if the model has named inputs.\n          - A `tf.data` dataset. Should return a tuple\n            of either `(inputs, targets)` or\n            `(inputs, targets, sample_weights)`.\n          - A generator or `keras.utils.Sequence` returning `(inputs, targets)`\n            or `(inputs, targets, sample weights)`.\n        y: Target data. Like the input data `x`,\n          it could be either Numpy array(s) or TensorFlow tensor(s).\n          It should be consistent with `x` (you cannot have Numpy inputs and\n          tensor targets, or inversely). If `x` is a dataset, generator,\n          or `keras.utils.Sequence` instance, `y` should\n          not be specified (since targets will be obtained from `x`).\n        batch_size: Integer or `None`.\n            Number of samples per gradient update.\n            If unspecified, `batch_size` will default to 32.\n            Do not specify the `batch_size` if your data is in the\n            form of symbolic tensors, datasets,\n            generators, or `keras.utils.Sequence` instances (since they generate\n            batches).\n        epochs: Integer. Number of epochs to train the model.\n            An epoch is an iteration over the entire `x` and `y`\n            data provided.\n            Note that in conjunction with `initial_epoch`,\n            `epochs` is to be understood as "final epoch".\n            The model is not trained for a number of iterations\n            given by `epochs`, but merely until the epoch\n            of index `epochs` is reached.\n        verbose: 0, 1, or 2. Verbosity mode.\n            0 = silent, 1 = progress bar, 2 = one line per epoch.\n            Note that the progress bar is not particularly useful when\n            logged to a file, so verbose=2 is recommended when not running\n            interactively (eg, in a production environment).\n        callbacks: List of `keras.callbacks.Callback` instances.\n            List of callbacks to apply during training.\n            See `tf.keras.callbacks`.\n        validation_split: Float between 0 and 1.\n            Fraction of the training data to be used as validation data.\n            The model will set apart this fraction of the training data,\n            will not train on it, and will evaluate\n            the loss and any model metrics\n            on this data at the end of each epoch.\n            The validation data is selected from the last samples\n            in the `x` and `y` data provided, before shuffling. This argument is\n            not supported when `x` is a dataset, generator or\n           `keras.utils.Sequence` instance.\n        validation_data: Data on which to evaluate\n            the loss and any model metrics at the end of each epoch.\n            The model will not be trained on this data.\n            `validation_data` will override `validation_split`.\n            `validation_data` could be:\n              - tuple `(x_val, y_val)` of Numpy arrays or tensors\n              - tuple `(x_val, y_val, val_sample_weights)` of Numpy arrays\n              - dataset\n            For the first two cases, `batch_size` must be provided.\n            For the last case, `validation_steps` could be provided.\n        shuffle: Boolean (whether to shuffle the training data\n            before each epoch) or str (for \'batch\').\n            \'batch\' is a special option for dealing with the\n            limitations of HDF5 data; it shuffles in batch-sized chunks.\n            Has no effect when `steps_per_epoch` is not `None`.\n        class_weight: Optional dictionary mapping class indices (integers)\n            to a weight (float) value, used for weighting the loss function\n            (during training only).\n            This can be useful to tell the model to\n            "pay more attention" to samples from\n            an under-represented class.\n        sample_weight: Optional Numpy array of weights for\n            the training samples, used for weighting the loss function\n            (during training only). You can either pass a flat (1D)\n            Numpy array with the same length as the input samples\n            (1:1 mapping between weights and samples),\n            or in the case of temporal data,\n            you can pass a 2D array with shape\n            `(samples, sequence_length)`,\n            to apply a different weight to every timestep of every sample.\n            In this case you should make sure to specify\n            `sample_weight_mode="temporal"` in `compile()`. This argument is not\n            supported when `x` is a dataset, generator, or\n           `keras.utils.Sequence` instance, instead provide the sample_weights\n            as the third element of `x`.\n        initial_epoch: Integer.\n            Epoch at which to start training\n            (useful for resuming a previous training run).\n        steps_per_epoch: Integer or `None`.\n            Total number of steps (batches of samples)\n            before declaring one epoch finished and starting the\n            next epoch. When training with input tensors such as\n            TensorFlow data tensors, the default `None` is equal to\n            the number of samples in your dataset divided by\n            the batch size, or 1 if that cannot be determined. If x is a\n            `tf.data` dataset, and \'steps_per_epoch\'\n            is None, the epoch will run until the input dataset is exhausted.\n            This argument is not supported with array inputs.\n        validation_steps: Only relevant if `validation_data` is provided and\n            is a `tf.data` dataset. Total number of steps (batches of\n            samples) to draw before stopping when performing validation\n            at the end of every epoch. If \'validation_steps\' is None, validation\n            will run until the `validation_data` dataset is exhausted. In the\n            case of a infinite dataset, it will run into a infinite loop.\n            If \'validation_steps\' is specified and only part of the dataset\n            will be consumed, the evaluation will start from the beginning of\n            the dataset at each epoch. This ensures that the same validation\n            samples are used every time.\n        validation_freq: Only relevant if validation data is provided. Integer\n            or `collections.abc.Container` instance (e.g. list, tuple, etc.).\n            If an integer, specifies how many training epochs to run before a\n            new validation run is performed, e.g. `validation_freq=2` runs\n            validation every 2 epochs. If a Container, specifies the epochs on\n            which to run validation, e.g. `validation_freq=[1, 2, 10]` runs\n            validation at the end of the 1st, 2nd, and 10th epochs.\n        max_queue_size: Integer. Used for generator or `keras.utils.Sequence`\n            input only. Maximum size for the generator queue.\n            If unspecified, `max_queue_size` will default to 10.\n        workers: Integer. Used for generator or `keras.utils.Sequence` input\n            only. Maximum number of processes to spin up\n            when using process-based threading. If unspecified, `workers`\n            will default to 1. If 0, will execute the generator on the main\n            thread.\n        use_multiprocessing: Boolean. Used for generator or\n            `keras.utils.Sequence` input only. If `True`, use process-based\n            threading. If unspecified, `use_multiprocessing` will default to\n            `False`. Note that because this implementation relies on\n            multiprocessing, you should not pass non-picklable arguments to\n            the generator as they can\'t be passed easily to children processes.\n        **kwargs: Used for backwards compatibility.\n\n    Returns:\n        A `History` object. Its `History.history` attribute is\n        a record of training loss values and metrics values\n        at successive epochs, as well as validation loss values\n        and validation metrics values (if applicable).\n\n    Raises:\n        RuntimeError: If the model was never compiled.\n        ValueError: In case of mismatch between the provided input data\n            and what the model expects.\n    '
        self._assert_built_as_v1()
        if 'nb_epoch' in kwargs:
            logging.warning('The `nb_epoch` argument in `fit` has been renamed `epochs`.')
            epochs = kwargs.pop('nb_epoch')
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))
        self._assert_compile_was_called()
        self._check_call_args('fit')
        func = self._select_training_loop(x)
        return func.fit(self, x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_split=validation_split, validation_data=validation_data, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        if False:
            print('Hello World!')
        'Returns the loss value & metrics values for the model in test mode.\n\n    Computation is done in batches (see the `batch_size` arg.)\n\n    Args:\n        x: Input data. It could be:\n          - A Numpy array (or array-like), or a list of arrays\n            (in case the model has multiple inputs).\n          - A TensorFlow tensor, or a list of tensors\n            (in case the model has multiple inputs).\n          - A dict mapping input names to the corresponding array/tensors,\n            if the model has named inputs.\n          - A `tf.data` dataset.\n          - A generator or `keras.utils.Sequence` instance.\n        y: Target data. Like the input data `x`,\n          it could be either Numpy array(s) or TensorFlow tensor(s).\n          It should be consistent with `x` (you cannot have Numpy inputs and\n          tensor targets, or inversely).\n          If `x` is a dataset, generator or\n          `keras.utils.Sequence` instance, `y` should not be specified (since\n          targets will be obtained from the iterator/dataset).\n        batch_size: Integer or `None`.\n            Number of samples per batch of computation.\n            If unspecified, `batch_size` will default to 32.\n            Do not specify the `batch_size` if your data is in the\n            form of symbolic tensors, dataset,\n            generators, or `keras.utils.Sequence` instances (since they generate\n            batches).\n        verbose: 0 or 1. Verbosity mode.\n            0 = silent, 1 = progress bar.\n        sample_weight: Optional Numpy array of weights for\n            the test samples, used for weighting the loss function.\n            You can either pass a flat (1D)\n            Numpy array with the same length as the input samples\n            (1:1 mapping between weights and samples),\n            or in the case of temporal data,\n            you can pass a 2D array with shape\n            `(samples, sequence_length)`,\n            to apply a different weight to every timestep of every sample.\n            In this case you should make sure to specify\n            `sample_weight_mode="temporal"` in `compile()`. This argument is not\n            supported when `x` is a dataset, instead pass\n            sample weights as the third element of `x`.\n        steps: Integer or `None`.\n            Total number of steps (batches of samples)\n            before declaring the evaluation round finished.\n            Ignored with the default value of `None`.\n            If x is a `tf.data` dataset and `steps` is\n            None, \'evaluate\' will run until the dataset is exhausted.\n            This argument is not supported with array inputs.\n        callbacks: List of `keras.callbacks.Callback` instances.\n            List of callbacks to apply during evaluation.\n            See [callbacks](/api_docs/python/tf/keras/callbacks).\n        max_queue_size: Integer. Used for generator or `keras.utils.Sequence`\n            input only. Maximum size for the generator queue.\n            If unspecified, `max_queue_size` will default to 10.\n        workers: Integer. Used for generator or `keras.utils.Sequence` input\n            only. Maximum number of processes to spin up when using\n            process-based threading. If unspecified, `workers` will default\n            to 1. If 0, will execute the generator on the main thread.\n        use_multiprocessing: Boolean. Used for generator or\n            `keras.utils.Sequence` input only. If `True`, use process-based\n            threading. If unspecified, `use_multiprocessing` will default to\n            `False`. Note that because this implementation relies on\n            multiprocessing, you should not pass non-picklable arguments to\n            the generator as they can\'t be passed easily to children processes.\n\n    Returns:\n        Scalar test loss (if the model has a single output and no metrics)\n        or list of scalars (if the model has multiple outputs\n        and/or metrics). The attribute `model.metrics_names` will give you\n        the display labels for the scalar outputs.\n\n    Raises:\n        ValueError: in case of invalid arguments.\n    '
        self._assert_built_as_v1()
        self._assert_compile_was_called()
        self._check_call_args('evaluate')
        func = self._select_training_loop(x)
        return func.evaluate(self, x=x, y=y, batch_size=batch_size, verbose=verbose, sample_weight=sample_weight, steps=steps, callbacks=callbacks, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)

    def predict(self, x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        if False:
            print('Hello World!')
        "Generates output predictions for the input samples.\n\n    Computation is done in batches (see the `batch_size` arg.)\n\n    Args:\n        x: Input samples. It could be:\n          - A Numpy array (or array-like), or a list of arrays\n            (in case the model has multiple inputs).\n          - A TensorFlow tensor, or a list of tensors\n            (in case the model has multiple inputs).\n          - A `tf.data` dataset.\n          - A generator or `keras.utils.Sequence` instance.\n        batch_size: Integer or `None`.\n            Number of samples per batch of computation.\n            If unspecified, `batch_size` will default to 32.\n            Do not specify the `batch_size` if your data is in the\n            form of symbolic tensors, dataset,\n            generators, or `keras.utils.Sequence` instances (since they generate\n            batches).\n        verbose: Verbosity mode, 0 or 1.\n        steps: Total number of steps (batches of samples)\n            before declaring the prediction round finished.\n            Ignored with the default value of `None`. If x is a `tf.data`\n            dataset and `steps` is None, `predict` will\n            run until the input dataset is exhausted.\n        callbacks: List of `keras.callbacks.Callback` instances.\n            List of callbacks to apply during prediction.\n            See [callbacks](/api_docs/python/tf/keras/callbacks).\n        max_queue_size: Integer. Used for generator or `keras.utils.Sequence`\n            input only. Maximum size for the generator queue.\n            If unspecified, `max_queue_size` will default to 10.\n        workers: Integer. Used for generator or `keras.utils.Sequence` input\n            only. Maximum number of processes to spin up when using\n            process-based threading. If unspecified, `workers` will default\n            to 1. If 0, will execute the generator on the main thread.\n        use_multiprocessing: Boolean. Used for generator or\n            `keras.utils.Sequence` input only. If `True`, use process-based\n            threading. If unspecified, `use_multiprocessing` will default to\n            `False`. Note that because this implementation relies on\n            multiprocessing, you should not pass non-picklable arguments to\n            the generator as they can't be passed easily to children processes.\n\n\n    Returns:\n        Numpy array(s) of predictions.\n\n    Raises:\n        ValueError: In case of mismatch between the provided\n            input data and the model's expectations,\n            or in case a stateful model receives a number of samples\n            that is not a multiple of the batch size.\n    "
        self._assert_built_as_v1()
        self._check_call_args('predict')
        func = self._select_training_loop(x)
        return func.predict(self, x=x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)

    def reset_metrics(self):
        if False:
            for i in range(10):
                print('nop')
        'Resets the state of metrics.'
        metrics = self._get_training_eval_metrics()
        for m in metrics:
            m.reset_state()
        if self._distribution_strategy:
            distributed_training_utils_v1._reset_metrics(self)

    def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None, reset_metrics=True):
        if False:
            while True:
                i = 10
        'Runs a single gradient update on a single batch of data.\n\n    Args:\n        x: Input data. It could be:\n          - A Numpy array (or array-like), or a list of arrays\n              (in case the model has multiple inputs).\n          - A TensorFlow tensor, or a list of tensors\n              (in case the model has multiple inputs).\n          - A dict mapping input names to the corresponding array/tensors,\n              if the model has named inputs.\n          - A `tf.data` dataset.\n        y: Target data. Like the input data `x`, it could be either Numpy\n          array(s) or TensorFlow tensor(s). It should be consistent with `x`\n          (you cannot have Numpy inputs and tensor targets, or inversely). If\n          `x` is a dataset, `y` should not be specified\n          (since targets will be obtained from the iterator).\n        sample_weight: Optional array of the same length as x, containing\n          weights to apply to the model\'s loss for each sample. In the case of\n          temporal data, you can pass a 2D array with shape (samples,\n          sequence_length), to apply a different weight to every timestep of\n          every sample. In this case you should make sure to specify\n          sample_weight_mode="temporal" in compile(). This argument is not\n          supported when `x` is a dataset.\n        class_weight: Optional dictionary mapping class indices (integers) to a\n          weight (float) to apply to the model\'s loss for the samples from this\n          class during training. This can be useful to tell the model to "pay\n          more attention" to samples from an under-represented class.\n        reset_metrics: If `True`, the metrics returned will be only for this\n          batch. If `False`, the metrics will be statefully accumulated across\n          batches.\n\n    Returns:\n        Scalar training loss\n        (if the model has a single output and no metrics)\n        or list of scalars (if the model has multiple outputs\n        and/or metrics). The attribute `model.metrics_names` will give you\n        the display labels for the scalar outputs.\n\n    Raises:\n      ValueError: In case of invalid user-provided arguments.\n    '
        self._assert_compile_was_called()
        self._check_call_args('train_on_batch')
        if self._distribution_strategy and distribute_lib.in_cross_replica_context():
            raise NotImplementedError('`train_on_batch` is not supported for models distributed with tf.distribute.Strategy.')
        (x, y, sample_weights) = self._standardize_user_data(x, y, sample_weight=sample_weight, class_weight=class_weight, extract_tensors_from_dataset=True)
        if self.run_eagerly or self._distribution_strategy:
            output_dict = training_eager_v1.train_on_batch(self, x, y, sample_weights=sample_weights, output_loss_metrics=self._output_loss_metrics)
            outputs = output_dict['total_loss'] + output_dict['output_losses'] + output_dict['metrics']
            outputs = [_non_none_constant_value(v) for v in outputs]
        else:
            x = training_utils_v1.ModelInputs(x).as_list()
            ins = x + list(y or []) + list(sample_weights or [])
            if not isinstance(backend.symbolic_learning_phase(), int):
                ins += [True]
            self._update_sample_weight_modes(sample_weights=sample_weights)
            self._make_train_function()
            outputs = self.train_function(ins)
        if reset_metrics:
            self.reset_metrics()
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def test_on_batch(self, x, y=None, sample_weight=None, reset_metrics=True):
        if False:
            return 10
        'Test the model on a single batch of samples.\n\n    Args:\n        x: Input data. It could be:\n          - A Numpy array (or array-like), or a list of arrays\n            (in case the model has multiple inputs).\n          - A TensorFlow tensor, or a list of tensors\n            (in case the model has multiple inputs).\n          - A dict mapping input names to the corresponding array/tensors,\n            if the model has named inputs.\n          - A `tf.data` dataset.\n        y: Target data. Like the input data `x`,\n          it could be either Numpy array(s) or TensorFlow tensor(s).\n          It should be consistent with `x` (you cannot have Numpy inputs and\n          tensor targets, or inversely). If `x` is a dataset `y` should\n          not be specified (since targets will be obtained from the iterator).\n        sample_weight: Optional array of the same length as x, containing\n            weights to apply to the model\'s loss for each sample.\n            In the case of temporal data, you can pass a 2D array\n            with shape (samples, sequence_length),\n            to apply a different weight to every timestep of every sample.\n            In this case you should make sure to specify\n            sample_weight_mode="temporal" in compile(). This argument is not\n            supported when `x` is a dataset.\n        reset_metrics: If `True`, the metrics returned will be only for this\n          batch. If `False`, the metrics will be statefully accumulated across\n          batches.\n\n    Returns:\n        Scalar test loss (if the model has a single output and no metrics)\n        or list of scalars (if the model has multiple outputs\n        and/or metrics). The attribute `model.metrics_names` will give you\n        the display labels for the scalar outputs.\n\n    Raises:\n        ValueError: In case of invalid user-provided arguments.\n    '
        self._assert_compile_was_called()
        self._check_call_args('test_on_batch')
        if self._distribution_strategy and distribute_lib.in_cross_replica_context():
            raise NotImplementedError('`test_on_batch` is not supported for models distributed with tf.distribute.Strategy.')
        (x, y, sample_weights) = self._standardize_user_data(x, y, sample_weight=sample_weight, extract_tensors_from_dataset=True)
        if self.run_eagerly or self._distribution_strategy:
            output_dict = training_eager_v1.test_on_batch(self, x, y, sample_weights=sample_weights, output_loss_metrics=self._output_loss_metrics)
            outputs = output_dict['total_loss'] + output_dict['output_losses'] + output_dict['metrics']
            outputs = [_non_none_constant_value(v) for v in outputs]
        else:
            x = training_utils_v1.ModelInputs(x).as_list()
            inputs = x + list(y or []) + list(sample_weights or [])
            self._update_sample_weight_modes(sample_weights=sample_weights)
            self._make_test_function()
            outputs = self.test_function(inputs)
        if reset_metrics:
            self.reset_metrics()
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def predict_on_batch(self, x):
        if False:
            for i in range(10):
                print('nop')
        'Returns predictions for a single batch of samples.\n\n    Args:\n        x: Input data. It could be:\n          - A Numpy array (or array-like), or a list of arrays\n            (in case the model has multiple inputs).\n          - A TensorFlow tensor, or a list of tensors\n            (in case the model has multiple inputs).\n          - A `tf.data` dataset.\n\n    Returns:\n        Numpy array(s) of predictions.\n\n    Raises:\n        ValueError: In case of mismatch between given number of inputs and\n          expectations of the model.\n    '
        self._check_call_args('predict_on_batch')
        if self._distribution_strategy and distribute_lib.in_cross_replica_context():
            raise NotImplementedError('`predict_on_batch` is not supported for models distributed with tf.distribute.Strategy.')
        (inputs, _, _) = self._standardize_user_data(x, extract_tensors_from_dataset=True)
        if self.run_eagerly or self._distribution_strategy:
            inputs = training_utils_v1.cast_if_floating_dtype(inputs)
            if isinstance(inputs, collections.abc.Sequence):
                if len(inputs) == 1:
                    inputs = inputs[0]
            return self(inputs)
        self._make_predict_function()
        outputs = self.predict_function(inputs)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, validation_freq=1, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0):
        if False:
            for i in range(10):
                print('nop')
        'Fits the model on data yielded batch-by-batch by a Python generator.\n\n    DEPRECATED:\n      `Model.fit` now supports generators, so there is no longer any need to use\n      this endpoint.\n    '
        warnings.warn('`model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.')
        return self.fit(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_data=validation_data, validation_steps=validation_steps, validation_freq=validation_freq, class_weight=class_weight, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing, shuffle=shuffle, initial_epoch=initial_epoch)

    def evaluate_generator(self, generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0):
        if False:
            return 10
        'Evaluates the model on a data generator.\n\n    DEPRECATED:\n      `Model.evaluate` now supports generators, so there is no longer any need\n      to use this endpoint.\n    '
        warnings.warn('`Model.evaluate_generator` is deprecated and will be removed in a future version. Please use `Model.evaluate`, which supports generators.')
        self._check_call_args('evaluate_generator')
        return self.evaluate(generator, steps=steps, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing, verbose=verbose, callbacks=callbacks)

    def predict_generator(self, generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0):
        if False:
            return 10
        'Generates predictions for the input samples from a data generator.\n\n    DEPRECATED:\n      `Model.predict` now supports generators, so there is no longer any need\n      to use this endpoint.\n    '
        warnings.warn('`Model.predict_generator` is deprecated and will be removed in a future version. Please use `Model.predict`, which supports generators.')
        return self.predict(generator, steps=steps, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing, verbose=verbose, callbacks=callbacks)

    def _check_call_args(self, method_name):
        if False:
            return 10
        'Check that `call` has only one positional arg.'
        fullargspec = self._call_full_argspec
        if fullargspec.defaults:
            positional_args = fullargspec.args[:-len(fullargspec.defaults)]
        else:
            positional_args = fullargspec.args
        if 'training' in positional_args:
            positional_args.remove('training')
        if len(positional_args) > 2:
            extra_args = positional_args[2:]
            raise ValueError('Models passed to `' + method_name + '` can only have `training` and the first argument in `call` as positional arguments, found: ' + str(extra_args) + '.')

    def _set_optimizer(self, optimizer):
        if False:
            while True:
                i = 10
        'Sets self.optimizer.\n\n    Sets self.optimizer to `optimizer`, potentially wrapping it with a\n    LossScaleOptimizer.\n\n    Args:\n      optimizer: The optimizer(s) to assign to self.optimizer.\n    '
        if isinstance(optimizer, (list, tuple)):
            self.optimizer = [optimizers.get(opt) for opt in optimizer]
        else:
            self.optimizer = optimizers.get(optimizer)
        if isinstance(self._dtype_policy, policy.PolicyV1):
            loss_scale = self._dtype_policy.loss_scale
        elif self._dtype_policy.name == 'mixed_float16':
            loss_scale = 'dynamic'
        else:
            loss_scale = None
        if loss_scale is not None and (not isinstance(self.optimizer, loss_scale_optimizer.LossScaleOptimizer)):
            if isinstance(self.optimizer, list):
                raise ValueError('When a dtype policy with a loss scale is used, you can only pass a single optimizer. Using policy %s and got optimizers: %s' % self._dtype_policy, self.optimizer)
            if not isinstance(self.optimizer, optimizer_v2.OptimizerV2):
                raise ValueError('"optimizer" must be an instance of tf.keras.optimizers.Optimizer when a dype policy with a loss scale  used, but got: %s. Using policy: %s' % (self.optimizer, self._dtype_policy))
            if loss_scale == 'dynamic':
                self.optimizer = loss_scale_optimizer.LossScaleOptimizer(self.optimizer)
            else:
                self.optimizer = loss_scale_optimizer.LossScaleOptimizerV1(self.optimizer, loss_scale)

    def _prepare_validation_data(self, validation_data, batch_size, validation_steps):
        if False:
            for i in range(10):
                print('nop')
        'Unpack and check the validation data.'
        (val_x, val_y, val_sample_weights) = training_utils_v1.unpack_validation_data(validation_data)
        return self._standardize_user_data(val_x, val_y, sample_weight=val_sample_weights, batch_size=batch_size, steps=validation_steps, steps_name='validation_steps')

    def _validate_compile_param_for_distribution_strategy(self, run_eagerly, sample_weight_mode, target_tensors, weighted_metrics):
        if False:
            i = 10
            return i + 15
        if self._distribution_strategy:
            if sample_weight_mode:
                raise NotImplementedError('sample_weight_mode is not supported with tf.distribute.Strategy.')
            if weighted_metrics:
                raise NotImplementedError('weighted_metrics is not supported with tf.distribute.Strategy.')
            if target_tensors:
                raise ValueError('target_tensors is not supported with tf.distribute.Strategy.')
            if run_eagerly:
                raise ValueError('We currently do not support enabling `run_eagerly` with distribution strategy.')
            if distributed_training_utils_v1.is_distributing_by_cloning(self) and (not self.built or not self.inputs or (not self.outputs)):
                raise ValueError('We currently do not support distribution strategy with a `Sequential` model that is created without `input_shape`/`input_dim` set in its first layer or a subclassed model.')

    def _process_target_tensor_for_compile(self, target_tensors):
        if False:
            i = 10
            return i + 15
        if self.run_eagerly:
            return [None for _ in self.output_names]
        if target_tensors is not None and (not (isinstance(target_tensors, list) and target_tensors == [])):
            if isinstance(target_tensors, list):
                if len(target_tensors) != len(self.outputs):
                    raise ValueError('When passing a list as `target_tensors`, it should have one entry per model output. The model has %s outputs, but you passed target_tensors=%s' % (len(self.outputs), target_tensors))
            elif isinstance(target_tensors, dict):
                unexpected_target_tensor_names = set(target_tensors.keys()).difference(self.output_names)
                if unexpected_target_tensor_names:
                    raise ValueError('Unknown entry in `target_tensors` dictionary: "{name}". Only expected the following keys: {keys}'.format(name=unexpected_target_tensor_names, keys=str(self.output_names)))
                tmp_target_tensors = []
                for name in self.output_names:
                    tmp_target_tensors.append(target_tensors.get(name, None))
                target_tensors = tmp_target_tensors
            elif tensor_util.is_tf_type(target_tensors):
                target_tensors = [target_tensors]
            else:
                raise TypeError('Expected `target_tensors` to be a list or tuple or dict or a single tensor, but got:', target_tensors)
        else:
            target_tensors = [None for _ in self.output_names]
        return target_tensors

    def _compile_eagerly(self, metrics, weighted_metrics, sample_weight_mode):
        if False:
            i = 10
            return i + 15
        training_utils_v1.prepare_sample_weight_modes(self._training_endpoints, sample_weight_mode)
        self._prepare_sample_weights()
        self._cache_output_metric_attributes(metrics, weighted_metrics)
        self.total_loss = None
        self._set_metric_attributes()
        self._collected_trainable_weights = self.trainable_weights

    def _update_sample_weight_modes(self, sample_weights=None):
        if False:
            for i in range(10):
                print('nop')
        "Updates sample weight modes based on training/eval inputs.\n\n    Sample weight placeholders will be created for all or no outputs\n    based on whether sample_weight is provided for any output.\n\n    If model contains `_sample_weight_modes` we check if the input\n    `sample_weights` corresponds to the sample weight modes.\n      1. Set sample weight mode to be 'temporal' for output i, if `compile`\n        sample_weight_mode was set to `temporal` and sample weight inputs\n        are given for one or more outputs.\n      2. Set sample weight mode to be 'samplewise' for output i, if `compile`\n        sample_weight_mode was not set and sample weight inputs are given for\n        one or more outputs.\n      3. Reset sample weight mode to None for output i if sample weight mode\n        was set but there is no sample weight input.\n\n    Args:\n      sample_weights: List of sample weights of the same length as model outputs\n        or None.\n    "
        if not self._is_compiled:
            return
        if sample_weights and any((s is not None for s in sample_weights)):
            for endpoint in self._training_endpoints:
                endpoint.sample_weight_mode = endpoint.sample_weight_mode or 'samplewise'
        else:
            for endpoint in self._training_endpoints:
                endpoint.sample_weight_mode = None

    def _recompile_weights_loss_and_weighted_metrics(self):
        if False:
            while True:
                i = 10
        if not self._is_compiled:
            return False
        recompile = any((e.sample_weights_mismatch() for e in self._training_endpoints))
        if recompile:
            self._compile_weights_loss_and_weighted_metrics()
        return recompile

    @trackable.no_automatic_dependency_tracking
    def _compile_weights_loss_and_weighted_metrics(self, sample_weights=None):
        if False:
            print('Hello World!')
        'Compiles the model loss and weighted metric sub-graphs.\n\n    This may be used to set graph tensors as sample weights (instead of creating\n    placeholders). This functionality is necessary for\n    `tf.keras.estimator.model_to_estimator`, which calls Keras models in a v1\n    graph, and creates iterator tensors for inputs, targets, and sample weights.\n\n    Args:\n      sample_weights: List of tensors to use as the sample weights. Must be the\n        same length as the number of outputs. If left as `None`, placeholders\n        are used instead.\n    '
        with backend.get_graph().as_default():
            if sample_weights is not None:
                self._update_sample_weight_modes(sample_weights)
            self._prepare_sample_weights(sample_weights)
            masks = self._prepare_output_masks()
            self._handle_metrics(self.outputs, targets=self._targets, skip_target_masks=self._prepare_skip_target_masks(), sample_weights=self.sample_weights, masks=masks, return_weighted_metrics=True)
            self.total_loss = self._prepare_total_loss(masks)

    def _prepare_skip_target_masks(self):
        if False:
            i = 10
            return i + 15
        'Boolean mask for whether the target in the output list should be skipped.\n\n    If the loss function corresponding to a model output is None, then this\n    output will be skipped during total loss calculation and feed targets\n    preparation.\n\n    Returns:\n      A boolean list for whether the corresponding target in the output list\n      should be skipped during loss calculation.\n    '
        return [l is None for l in self.loss_functions]

    def _prepare_output_masks(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns masks corresponding to model outputs.'
        return [getattr(x, '_keras_mask', None) for x in self.outputs]

    def _prepare_total_loss(self, masks):
        if False:
            return 10
        'Computes total loss from loss functions.\n\n    Args:\n        masks: List of mask values corresponding to each model output.\n\n    Returns:\n        A list of loss weights of python floats.\n\n    Raises:\n        TypeError: If model run_eagerly is True.\n    '
        if self.run_eagerly:
            raise TypeError('total loss can not be computed when compiled with run_eagerly = True.')
        loss_list = []
        with backend.name_scope('loss'):
            for (endpoint, mask) in zip(self._training_endpoints, masks):
                if endpoint.should_skip_target():
                    continue
                y_true = endpoint.training_target.target
                y_pred = endpoint.output
                loss_fn = endpoint.loss_fn
                loss_weight = endpoint.loss_weight
                loss_name = endpoint.loss_name()
                sample_weight = endpoint.sample_weight
                with backend.name_scope(loss_name):
                    if mask is not None:
                        mask = math_ops.cast(mask, y_pred.dtype)
                        if sample_weight is None:
                            sample_weight = mask
                        else:
                            (mask, _, sample_weight) = losses_utils.squeeze_or_expand_dimensions(mask, sample_weight=sample_weight)
                            sample_weight *= mask
                    if hasattr(loss_fn, 'reduction'):
                        per_sample_losses = loss_fn.call(y_true, y_pred)
                        weighted_losses = losses_utils.compute_weighted_loss(per_sample_losses, sample_weight=sample_weight, reduction=losses_utils.ReductionV2.NONE)
                        loss_reduction = loss_fn.reduction
                        if loss_reduction == losses_utils.ReductionV2.AUTO:
                            loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
                        output_loss = losses_utils.reduce_weighted_loss(weighted_losses, reduction=loss_reduction)
                    else:
                        output_loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
                        loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
                if len(self.outputs) > 1:
                    endpoint.output_loss_metric(output_loss)
                if loss_reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE:
                    output_loss = losses_utils.scale_loss_for_distribution(output_loss)
                loss_list.append(loss_weight * output_loss)
            if not loss_list and (not self.losses):
                raise ValueError('The model cannot be compiled because it has no loss to optimize.')
            custom_losses = self.get_losses_for(None) + self.get_losses_for(self.inputs)
            if custom_losses:
                total_custom_loss = math_ops.add_n(losses_utils.cast_losses_to_common_dtype(custom_losses))
                loss_list.append(losses_utils.scale_loss_for_distribution(total_custom_loss))
            loss_list = losses_utils.cast_losses_to_common_dtype(loss_list)
            if loss_list:
                total_loss = math_ops.add_n(loss_list)
            else:
                total_loss = 0.0
        return total_loss

    def _get_callback_model(self):
        if False:
            i = 10
            return i + 15
        'Returns the Callback Model for this Model.'
        if hasattr(self, '_replicated_model') and self._replicated_model:
            return self._replicated_model
        if hasattr(self, 'callback_model') and self.callback_model:
            return self.callback_model
        return self

    @trackable.no_automatic_dependency_tracking
    def _make_callback_model(self, grouped_model):
        if False:
            print('Hello World!')
        first_replicated_model = self._distribution_strategy.unwrap(grouped_model)[0]
        self._replicated_model = DistributedCallbackModel(first_replicated_model)
        self._replicated_model.set_original_model(self)

    def _validate_or_infer_batch_size(self, batch_size, steps, x):
        if False:
            while True:
                i = 10
        "Validates that the `batch_size` provided is consistent with InputLayer.\n\n    It's possible that the user specified a static batch size in their\n    InputLayer. If so, this method checks the provided `batch_size` and `x`\n    arguments are consistent with this static batch size. Also, if\n    `batch_size` is `None`, this method will attempt to infer the batch size\n    from the static batch size of the InputLayer. Lastly, ValueError will be\n    raised if `x` is a tf.data.Dataset and `batch_size` is specified as we\n    expect users to provide batched datasets.\n\n    Args:\n      batch_size: The batch_size provided as an argument to\n        fit/evaluate/predict.\n      steps: The steps provided as an argument to fit/evaluate/predict.\n      x: The data passed as `x` to fit/evaluate/predict.\n\n    Returns:\n      The validated batch_size, auto-inferred from the first layer if not\n      provided.\n    "
        if isinstance(x, (data_types.DatasetV1, data_types.DatasetV2, data_utils.Sequence)) or tf_inspect.isgenerator(x):
            if batch_size is not None:
                raise ValueError('The `batch_size` argument must not be specified for the given input type. Received input: {}, batch_size: {}'.format(x, batch_size))
            return
        layers = self._flatten_layers(include_self=False, recursive=False)
        first_layer = next(layers, None)
        if first_layer:
            static_batch_size = training_utils.get_static_batch_size(first_layer)
            if static_batch_size is not None:
                if self._distribution_strategy and distributed_training_utils.global_batch_size_supported(self._distribution_strategy):
                    num_splits_for_ds = self._distribution_strategy.num_replicas_in_sync
                else:
                    num_splits_for_ds = 1
                if batch_size is not None:
                    if batch_size % num_splits_for_ds != 0:
                        raise ValueError('The `batch_size` argument ({}) must be divisible the by number of replicas ({})'.format(batch_size, num_splits_for_ds))
                    per_replica_batch_size = batch_size // num_splits_for_ds
                    if per_replica_batch_size != static_batch_size:
                        raise ValueError('The `batch_size` argument value {} is incompatible with the specified batch size of your Input Layer: {}'.format(per_replica_batch_size, static_batch_size))
                if isinstance(x, (data_types.DatasetV2, iterator_ops.Iterator, iterator_ops.IteratorBase)):
                    ds_batch_size = tensor_shape.Dimension(nest.flatten(dataset_ops.get_legacy_output_shapes(x))[0][0]).value
                    if ds_batch_size is not None:
                        if ds_batch_size % num_splits_for_ds != 0:
                            raise ValueError('The batch output shape of your `Dataset` {} cannot be divisible by number of replicas {}'.format(ds_batch_size, num_splits_for_ds))
                        ds_per_replica_batch_size = ds_batch_size // num_splits_for_ds
                        if ds_per_replica_batch_size != static_batch_size:
                            raise ValueError('The batch output shape of your `Dataset` is {}, which is incompatible with the specified batch size of your Input Layer: {}'.format(ds_per_replica_batch_size, static_batch_size))
                if steps is None:
                    batch_size = static_batch_size * num_splits_for_ds
        if batch_size is None and steps is None:
            batch_size = 32
        return batch_size

    def _prepare_sample_weights(self, sample_weights=None):
        if False:
            for i in range(10):
                print('nop')
        'Sets sample weight attribute on the model.'
        if sample_weights is not None:
            if len(sample_weights) != len(self._training_endpoints):
                raise ValueError('Provided sample weights must have same length as the number of outputs. Expected: {}, got: {}.'.format(len(self._training_endpoints), len(sample_weights)))
        else:
            sample_weights = [None] * len(self._training_endpoints)
        for (endpoint, weight) in zip(self._training_endpoints, sample_weights):
            endpoint.populate_sample_weight(weight, endpoint.sample_weight_mode)

    def _cache_output_metric_attributes(self, metrics, weighted_metrics):
        if False:
            for i in range(10):
                print('nop')
        'Caches metric name and function attributes for every model output.'
        output_shapes = []
        for output in self.outputs:
            if output is None or output.shape.rank is None:
                output_shapes.append(None)
            else:
                output_shapes.append(output.shape.as_list())
        self._per_output_metrics = training_utils_v1.collect_per_output_metric_info(metrics, self.output_names, output_shapes, self.loss_functions, from_serialized=self._from_serialized)
        self._per_output_weighted_metrics = training_utils_v1.collect_per_output_metric_info(weighted_metrics, self.output_names, output_shapes, self.loss_functions, from_serialized=self._from_serialized, is_weighted=True)

    def _add_unique_metric_name(self, metric_name, metric_fn, output_index):
        if False:
            for i in range(10):
                print('nop')
        "Makes the metric name unique.\n\n      If there are multiple outputs for which the metrics are calculated, the\n      metric names have to be made unique by appending an integer.\n\n    Args:\n      metric_name: Metric name that corresponds to the metric specified by the\n          user. For example: 'acc'.\n      metric_fn: The Metric object.\n      output_index: The index of the model output for which the metric name is\n        being added.\n\n    Returns:\n      string, name of the model's unique metric name\n    "
        if len(self.output_names) > 1:
            if not getattr(metric_fn, '_from_serialized', False):
                metric_name = '%s_%s' % (self.output_names[output_index], metric_name)
        j = 1
        base_metric_name = metric_name
        while metric_name in self.metrics_names:
            metric_name = '%s_%d' % (base_metric_name, j)
            j += 1
        return metric_name

    def _init_metric_attributes(self):
        if False:
            return 10
        'Initialized model metric attributes.'
        self._compile_metric_functions = []

    def _set_per_output_metric_attributes(self, metrics_dict, output_index):
        if False:
            for i in range(10):
                print('nop')
        'Sets the metric attributes on the model for the given output.\n\n    Args:\n      metrics_dict: A dict with metric names as keys and metric fns as values.\n      output_index: The index of the model output for which the metric\n        attributes are added.\n\n    Returns:\n      Metrics dict updated with unique metric names as keys.\n    '
        updated_metrics_dict = collections.OrderedDict()
        for (metric_name, metric_fn) in metrics_dict.items():
            metric_name = self._add_unique_metric_name(metric_name, metric_fn, output_index)
            metric_fn._name = metric_name
            updated_metrics_dict[metric_name] = metric_fn
            self._compile_metric_functions.append(metric_fn)
        return updated_metrics_dict

    def _set_metric_attributes(self):
        if False:
            return 10
        'Sets the metric attributes on the model for all the model outputs.'
        updated_per_output_metrics = []
        updated_per_output_weighted_metrics = []
        for (i, endpoint) in enumerate(self._training_endpoints):
            if endpoint.should_skip_target():
                updated_per_output_metrics.append(self._per_output_metrics[i])
                updated_per_output_weighted_metrics.append(self._per_output_weighted_metrics[i])
                continue
            updated_per_output_metrics.append(self._set_per_output_metric_attributes(self._per_output_metrics[i], i))
            updated_per_output_weighted_metrics.append(self._set_per_output_metric_attributes(self._per_output_weighted_metrics[i], i))
        if len(self._training_endpoints) > 1:
            for endpoint in self._training_endpoints:
                if not endpoint.should_skip_target():
                    endpoint.output_loss_metric = metrics_module.Mean(name=endpoint.loss_name())
        self._per_output_metrics = updated_per_output_metrics
        self._per_output_weighted_metrics = updated_per_output_weighted_metrics

    def _handle_per_output_metrics(self, metrics_dict, y_true, y_pred, mask, weights=None):
        if False:
            i = 10
            return i + 15
        'Calls metric functions for a single output.\n\n    Args:\n      metrics_dict: A dict with metric names as keys and metric fns as values.\n      y_true: Target output.\n      y_pred: Predicted output.\n      mask: Computed mask value for the current output.\n      weights: Weights to be applied on the current output.\n\n    Returns:\n      A list of metric result tensors.\n    '
        metric_results = []
        for (metric_name, metric_fn) in metrics_dict.items():
            with backend.name_scope(metric_name):
                metric_result = training_utils_v1.call_metric_function(metric_fn, y_true, y_pred, weights=weights, mask=mask)
                metric_results.append(metric_result)
        return metric_results

    def _handle_metrics(self, outputs, targets=None, skip_target_masks=None, sample_weights=None, masks=None, return_weighted_metrics=False, return_weighted_and_unweighted_metrics=False):
        if False:
            print('Hello World!')
        'Handles calling metric functions.\n\n    Args:\n      outputs: List of outputs (predictions).\n      targets: List of targets.\n      skip_target_masks: Optional. List of boolean for whether the corresponding\n        target should be ignored or not.\n      sample_weights: Optional list of sample weight arrays.\n      masks: List of computed output mask values.\n      return_weighted_metrics: Flag that indicates whether weighted metrics\n        should be computed instead of unweighted metrics. This flag is ignored\n        when `return_weighted_and_unweighted_metrics` is enabled.\n      return_weighted_and_unweighted_metrics: Flag that is used to indicate\n        whether both weighted and unweighted metrics should be computed. When\n        this is not enabled, we use `return_weighted_metrics` param to indicate\n        whether weighted or unweighted metrics should be returned.\n\n    Returns:\n      A list of metric result tensors.\n    '
        skip_target_masks = skip_target_masks or [False] * len(outputs)
        metric_results = []
        with backend.name_scope('metrics'):
            for i in range(len(outputs)):
                if skip_target_masks[i]:
                    continue
                output = outputs[i] if outputs else None
                target = targets[i] if targets else None
                output_mask = masks[i] if masks else None
                if return_weighted_and_unweighted_metrics or not return_weighted_metrics:
                    metric_results.extend(self._handle_per_output_metrics(self._per_output_metrics[i], target, output, output_mask))
                if return_weighted_and_unweighted_metrics or return_weighted_metrics:
                    metric_results.extend(self._handle_per_output_metrics(self._per_output_weighted_metrics[i], target, output, output_mask, weights=sample_weights[i] if sample_weights else None))
        return metric_results

    def _check_trainable_weights_consistency(self):
        if False:
            for i in range(10):
                print('nop')
        'Check trainable weights count consistency.\n\n    This will raise a warning if `trainable_weights` and\n    `_collected_trainable_weights` are inconsistent (i.e. have different\n    number of parameters).\n    Inconsistency will typically arise when one modifies `model.trainable`\n    without calling `model.compile` again.\n    '
        if not hasattr(self, '_collected_trainable_weights'):
            return
        if len(self.trainable_weights) != len(self._collected_trainable_weights):
            logging.log_first_n(logging.WARN, 'Discrepancy between trainable weights and collected trainable weights, did you set `model.trainable` without calling `model.compile` after ?', 1)

    def _make_train_function(self):
        if False:
            for i in range(10):
                print('nop')
        has_recompiled = self._recompile_weights_loss_and_weighted_metrics()
        self._check_trainable_weights_consistency()
        if isinstance(self.optimizer, list):
            raise ValueError('The `optimizer` in `compile` should be a single optimizer.')
        if getattr(self, 'train_function', None) is None or has_recompiled:
            current_trainable_state = self._get_trainable_state()
            self._set_trainable_state(self._compiled_trainable_state)
            inputs = self._feed_inputs + self._feed_targets + self._feed_sample_weights
            if not isinstance(backend.symbolic_learning_phase(), int):
                inputs += [backend.symbolic_learning_phase()]
            with backend.get_graph().as_default():
                with backend.name_scope('training'):
                    updates = self.optimizer.get_updates(params=self._collected_trainable_weights, loss=self.total_loss)
                    updates += self.get_updates_for(None)
                    updates += self.get_updates_for(self.inputs)
                metrics = self._get_training_eval_metrics()
                metrics_tensors = [m._call_result for m in metrics if hasattr(m, '_call_result')]
            with backend.name_scope('training'):
                fn = backend.function(inputs, [self.total_loss] + metrics_tensors, updates=updates, name='train_function', **self._function_kwargs)
                setattr(self, 'train_function', fn)
            self._set_trainable_state(current_trainable_state)

    def _make_test_function(self):
        if False:
            print('Hello World!')
        has_recompiled = self._recompile_weights_loss_and_weighted_metrics()
        if getattr(self, 'test_function', None) is None or has_recompiled:
            inputs = self._feed_inputs + self._feed_targets + self._feed_sample_weights
            with backend.get_graph().as_default():
                metrics = self._get_training_eval_metrics()
                metrics_tensors = [m._call_result for m in metrics if hasattr(m, '_call_result')]
            with backend.name_scope('evaluation'):
                updates = self.state_updates
                fn = backend.function(inputs, [self.total_loss] + metrics_tensors, updates=updates, name='test_function', **self._function_kwargs)
                setattr(self, 'test_function', fn)

    def _make_predict_function(self):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, 'predict_function'):
            self.predict_function = None
        if self.predict_function is None:
            inputs = self._feed_inputs
            kwargs = getattr(self, '_function_kwargs', {})
            with backend.name_scope(ModeKeys.PREDICT):
                self.predict_function = backend.function(inputs, self.outputs, updates=self.state_updates, name='predict_function', **kwargs)

    def _make_execution_function(self, mode):
        if False:
            print('Hello World!')
        if mode == ModeKeys.TRAIN:
            self._make_train_function()
            return self.train_function
        if mode == ModeKeys.TEST:
            self._make_test_function()
            return self.test_function
        if mode == ModeKeys.PREDICT:
            self._make_predict_function()
            return self.predict_function

    def _distribution_standardize_user_data(self, x, y=None, sample_weight=None, class_weight=None, batch_size=None, validation_split=0, shuffle=False, epochs=1, allow_partial_batch=False):
        if False:
            return 10
        'Runs validation checks on input and target data passed by the user.\n\n    This is called when using tf.distribute.Strategy to train, evaluate or serve\n    the model.\n\n    Args:\n      x: Input data. A numpy array or `tf.data` dataset.\n      y: Target data. A numpy array or None if x is a `tf.data` dataset.\n      sample_weight: An optional sample-weight array passed by the user to\n        weight the importance of each sample in `x`.\n      class_weight: An optional class-weight array by the user to\n        weight the importance of samples in `x` based on the class they belong\n        to, as conveyed by `y`.\n      batch_size: Integer batch size. If provided, it is used to run additional\n        validation checks on stateful models.\n      validation_split: Float between 0 and 1.\n        Fraction of the training data to be used as validation data.\n      shuffle: Boolean whether to shuffle the training data before each epoch.\n      epochs: Integer epochs. If > 1, repeat the numpy training data epochs\n        times when converting to training dataset.\n      allow_partial_batch: Boolean whether to enforce that all batches have the\n        same size.\n\n    Returns:\n      Dataset instance.\n\n    Raises:\n      ValueError: In case of invalid user-provided data.\n      RuntimeError: If the model was never compiled.\n    '
        if class_weight:
            raise NotImplementedError('`class_weight` is currently not supported when using tf.distribute.Strategy.')
        if sample_weight is not None and sample_weight.all() and backend.is_tpu_strategy(self._distribution_strategy):
            raise NotImplementedError('`sample_weight` is currently not supported when using TPUStrategy.')
        if isinstance(x, data_types.DatasetV2):
            if shuffle:
                training_utils_v1.verify_dataset_shuffled(x)
        strategy = self._distribution_strategy
        with strategy.scope():
            if ops.executing_eagerly_outside_functions():
                session = None
            else:
                session = backend.get_session()
            first_x_value = nest.flatten(x)[0]
            if isinstance(first_x_value, np.ndarray):
                x = training_utils.list_to_tuple(x)
                if y is not None:
                    y = training_utils.list_to_tuple(y)
                    if sample_weight is not None:
                        sample_weight = training_utils.list_to_tuple(sample_weight)
                        in_tuple = (x, y, sample_weight)
                    else:
                        in_tuple = (x, y)
                else:
                    in_tuple = x
                ds = strategy.extended.experimental_make_numpy_dataset(in_tuple, session=session)
                if shuffle:
                    ds = ds.shuffle(max(1024, batch_size * 8))
                if epochs > 1:
                    ds = ds.repeat(epochs)
                drop_remainder = not allow_partial_batch and strategy.extended.experimental_require_static_shapes
                if backend.is_tpu_strategy(strategy) and (not drop_remainder):
                    dataset_size = first_x_value.shape[0]
                    if dataset_size % batch_size == 0:
                        drop_remainder = True
                x = ds.batch(batch_size, drop_remainder=drop_remainder)
            else:
                assert isinstance(x, data_types.DatasetV2)
                training_utils_v1.validate_dataset_input(x, y, sample_weight, validation_split)
        return x

    def _standardize_user_data(self, x, y=None, sample_weight=None, class_weight=None, batch_size=None, check_steps=False, steps_name='steps', steps=None, validation_split=0, shuffle=False, extract_tensors_from_dataset=False):
        if False:
            i = 10
            return i + 15
        "Runs validation checks on input and target data passed by the user.\n\n    Also standardizes the data to lists of arrays, in order.\n\n    Also builds and compiles the model on the fly if it is a subclassed model\n    that has never been called before (and thus has no inputs/outputs).\n\n    This is a purely internal method, subject to refactoring at any time.\n\n    Args:\n      x: Input data. It could be:\n        - A Numpy array (or array-like), or a list of arrays\n          (in case the model has multiple inputs).\n        - A TensorFlow tensor, or a list of tensors\n          (in case the model has multiple inputs).\n        - A dict mapping input names to the corresponding array/tensors,\n          if the model has named inputs.\n        - A `tf.data` dataset.\n      y: Target data. Like the input data `x`,\n        it could be either Numpy array(s) or TensorFlow tensor(s).\n        It should be consistent with `x` (you cannot have Numpy inputs and\n        tensor targets, or inversely). If `x` is a dataset, `y` should not be\n        specified (since targets will be obtained from the iterator).\n      sample_weight: An optional sample-weight array passed by the user to\n        weight the importance of each sample in `x`.\n      class_weight: An optional class-weight array by the user to\n        weight the importance of samples in `x` based on the class they belong\n        to, as conveyed by `y`. If both `sample_weight` and `class_weight` are\n        provided, the weights are multiplied.\n      batch_size: Integer batch size. If provided, it is used to run additional\n        validation checks on stateful models.\n      check_steps: boolean, True if we want to check for validity of `steps` and\n        False, otherwise. For example, when we are standardizing one batch of\n        data for train_on_batch/predict_on_batch/test_on_batch APIs, `steps`\n        value is not required and we should not check for its validity in these\n        cases.\n      steps_name: The public API's parameter name for `steps`.\n      steps: Integer or `None`. Total number of steps (batches of samples) to\n        execute.\n      validation_split: Float between 0 and 1.\n        Fraction of the training data to be used as validation data.\n      shuffle: Boolean whether to shuffle the training data before each epoch.\n      extract_tensors_from_dataset: Boolean. When `x` is a dataset instance,\n        this indicates whether to extract actual tensors from the dataset or\n        instead output the dataset instance itself.\n        Set to True when calling from `train_on_batch`/etc.\n\n    Returns:\n      A tuple of 3: inputs (arrays or dicts, depending on whether `x` was a dict\n      or not), target arrays, sample-weight arrays.\n      If the model's input and targets are symbolic, these lists are empty\n      (since the model takes no user-provided data, instead the data comes\n      from the symbolic inputs/targets).\n\n    Raises:\n      ValueError: In case of invalid user-provided data.\n      RuntimeError: If the model was never compiled.\n    "
        if isinstance(x, (data_types.DatasetV1, data_types.DatasetV2)):
            training_utils_v1.validate_dataset_input(x, y, sample_weight, validation_split)
            if shuffle:
                training_utils_v1.verify_dataset_shuffled(x)
            is_dataset = True
            if extract_tensors_from_dataset:
                (x, y, sample_weight) = training_utils_v1.extract_tensors_from_dataset(x)
        elif isinstance(x, iterator_ops.Iterator):
            training_utils_v1.validate_dataset_input(x, y, sample_weight, validation_split)
            iterator = x
            (x, y, sample_weight) = training_utils_v1.unpack_iterator_input(iterator)
            is_dataset = True
        else:
            is_dataset = False
        if check_steps:
            training_utils_v1.check_steps_argument(x, steps, steps_name)
        if not self.inputs:
            (all_inputs, y_input, dict_inputs) = self._build_model_with_inputs(x, y)
            is_build_called = True
        else:
            all_inputs = []
            dict_inputs = isinstance(self.inputs, dict)
            is_build_called = False
            y_input = y
        is_compile_called = False
        if not self._is_compiled and self.optimizer:
            self._compile_from_inputs(all_inputs, y_input, x, y)
            is_compile_called = True
        run_eagerly = self.run_eagerly
        if not run_eagerly and is_build_called and is_compile_called and (not is_dataset) and any((_is_symbolic_tensor(v) for v in all_inputs)):
            return ([], [], None)
        return self._standardize_tensors(x, y, sample_weight, run_eagerly=run_eagerly, dict_inputs=dict_inputs, is_dataset=is_dataset, class_weight=class_weight, batch_size=batch_size)

    def _standardize_tensors(self, x, y, sample_weight, run_eagerly, dict_inputs, is_dataset, class_weight=None, batch_size=None):
        if False:
            i = 10
            return i + 15
        if run_eagerly:
            feed_input_names = self.input_names
            feed_input_shapes = None
        elif not self._is_graph_network:
            feed_input_names = self._feed_input_names
            feed_input_shapes = None
        else:
            feed_input_names = self._feed_input_names
            feed_input_shapes = self._feed_input_shapes
        if not isinstance(x, (data_types.DatasetV1, data_types.DatasetV2)):
            x = training_utils_v1.standardize_input_data(x, feed_input_names, feed_input_shapes, check_batch_axis=False, exception_prefix='input')
        if isinstance(x, data_types.DatasetV2):
            x_shapes = dataset_ops.get_structure(x)
            if isinstance(x_shapes, tuple):
                x_shapes = x_shapes[0]
        else:
            flat_inputs = nest.flatten(x, expand_composites=False)
            flat_expected_inputs = nest.flatten(self.inputs, expand_composites=False)
            converted_x = []
            for (a, b) in zip(flat_inputs, flat_expected_inputs):
                converted_x.append(_convert_scipy_sparse_tensor(a, b))
            x = nest.pack_sequence_as(x, converted_x, expand_composites=False)

            def _type_spec_from_value(value):
                if False:
                    for i in range(10):
                        print('nop')
                'Grab type_spec without converting array-likes to tensors.'
                if tf_utils.is_extension_type(value):
                    return value._type_spec
                if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                    return tensor_spec.TensorSpec(value.shape, value.dtype)
                else:
                    return type_spec.type_spec_from_value(value)
            x_shapes = nest.map_structure(_type_spec_from_value, x)
        flat_inputs = nest.flatten(x_shapes, expand_composites=False)
        flat_expected_inputs = nest.flatten(self.inputs, expand_composites=False)
        for (a, b) in zip(flat_inputs, flat_expected_inputs):
            nest.assert_same_structure(a, b, expand_composites=True)
        if y is not None:
            training_utils_v1.prepare_sample_weight_modes(self._training_endpoints, self.sample_weight_mode)
            feed_output_names = self._feed_output_names
            feed_sample_weight_modes = self._sample_weight_modes
            if not self._is_graph_network:
                feed_output_shapes = None
            else:
                feed_output_shapes = self._feed_output_shapes
            y = training_utils_v1.standardize_input_data(y, feed_output_names, shapes=None, check_batch_axis=False, exception_prefix='target')
            sample_weights = training_utils_v1.standardize_sample_weights(sample_weight, feed_output_names)
            class_weights = training_utils_v1.standardize_class_weights(class_weight, feed_output_names)
            sample_weights = [training_utils_v1.standardize_weights(ref, sw, cw, mode) for (ref, sw, cw, mode) in zip(y, sample_weights, class_weights, feed_sample_weight_modes)]
            if not self._distribution_strategy:
                training_utils_v1.check_array_lengths(x, y, sample_weights)
                if self._is_graph_network and (not run_eagerly):
                    training_utils_v1.check_loss_and_target_compatibility(y, self._feed_loss_fns, feed_output_shapes)
            (sample_weights, _, _) = training_utils.handle_partial_sample_weights(y, sample_weights, feed_sample_weight_modes, check_all_flat=True)
        else:
            y = []
            sample_weights = None
        if self.stateful and batch_size and (not is_dataset):
            if x[0].shape[0] % batch_size != 0:
                raise ValueError('In a stateful network, you should only pass inputs with a number of samples that can be divided by the batch size. Found: ' + str(x[0].shape[0]) + ' samples')
        if dict_inputs and (not isinstance(x, (data_types.DatasetV1, data_types.DatasetV2))):
            x = dict(zip(feed_input_names, x))
        return (x, y, sample_weights)

    def _build_model_with_inputs(self, inputs, targets):
        if False:
            print('Hello World!')
        'Build the model (set model inputs/outputs), mainly for subclass model.'
        processed_inputs = []
        is_dict_inputs = False
        orig_inputs = inputs
        if isinstance(inputs, (data_types.DatasetV1, data_types.DatasetV2)):
            (inputs, targets, _) = training_utils_v1.extract_tensors_from_dataset(inputs)
        training_utils_v1.validate_input_types(inputs, orig_inputs)
        if isinstance(inputs, (list, tuple)):
            processed_inputs += list(inputs)
        elif isinstance(inputs, dict):
            is_dict_inputs = True
            keys = sorted(inputs.keys())
            processed_inputs = [inputs[k] for k in keys]
        else:
            processed_inputs.append(inputs)
        for input_tensor in processed_inputs:
            if training_utils_v1.is_composite_or_composite_value(input_tensor):
                raise ValueError('All SparseTensor and RaggedTensor inputs must be explicitly declared using a keras.Input() with sparse=True or ragged=True. We found an undeclared input %s. For Sequential models, please add a keras.Input() as your first Layer. For subclassed models, please call self._set_inputs() on your input set, which you can create using keras.Input() for each input to your model.' % (input_tensor,))
        if isinstance(orig_inputs, (data_types.DatasetV1, data_types.DatasetV2, iterator_ops.Iterator)):
            if not self.inputs:
                inputs = training_utils_v1.cast_if_floating_dtype(inputs, self.dtype)

            def create_tensor_spec(t):
                if False:
                    print('Hello World!')
                return tensor_spec.TensorSpec(t.shape, t.dtype)
            cast_inputs = nest.map_structure(create_tensor_spec, inputs)
        elif training_utils_v1.has_tensors(inputs):
            cast_inputs = training_utils_v1.cast_if_floating_dtype(inputs)
        else:
            cast_inputs = inputs
        self._set_inputs(cast_inputs)
        return (processed_inputs, targets, is_dict_inputs)

    def _compile_from_inputs(self, all_inputs, target, orig_inputs, orig_target):
        if False:
            i = 10
            return i + 15
        if target is not None:
            if training_utils_v1.has_tensors(target):
                target = training_utils_v1.cast_if_floating_dtype_and_mismatch(target, self.outputs)
            training_utils_v1.validate_input_types(target, orig_target, allow_dict=False, field_name='target')
            if isinstance(target, (list, tuple)):
                all_inputs += list(target)
            else:
                all_inputs.append(target)
        if any((tensor_util.is_tf_type(v) for v in all_inputs)):
            if not all((tensor_util.is_tf_type(v) for v in all_inputs)):
                raise ValueError('Do not pass inputs that mix Numpy arrays and TensorFlow tensors. You passed: x=' + str(orig_inputs) + '; y=' + str(orig_target))
        is_dataset = isinstance(orig_inputs, (data_types.DatasetV1, data_types.DatasetV2, iterator_ops.Iterator))
        if is_dataset or context.executing_eagerly():
            target_tensors = None
        elif target is not None:
            if not isinstance(target, (list, tuple)):
                target = [target]
            target_tensors = [v for v in target if _is_symbolic_tensor(v)]
        else:
            target_tensors = None
        self.compile(optimizer=self.optimizer, loss=self.loss, metrics=self._compile_metrics, weighted_metrics=self._compile_weighted_metrics, loss_weights=self.loss_weights, target_tensors=target_tensors, sample_weight_mode=self.sample_weight_mode, run_eagerly=self.run_eagerly, experimental_run_tf_function=self._experimental_run_tf_function)

    def _set_inputs(self, inputs, outputs=None, training=None):
        if False:
            i = 10
            return i + 15
        "Set model's input and output specs based on the input data received.\n\n    This is to be used for Model subclasses, which do not know at instantiation\n    time what their inputs look like.\n\n    Args:\n      inputs: Single array, or list of arrays. The arrays could be placeholders,\n        Numpy arrays, data tensors, or TensorSpecs.\n        - if placeholders: the model is built on top of these placeholders,\n          and we expect Numpy data to be fed for them when calling `fit`/etc.\n        - if Numpy data or TensorShapes: we create placeholders matching the\n          TensorShapes or shapes of the Numpy arrays. We expect Numpy data to be\n          fed for these placeholders when calling `fit`/etc.\n        - if data tensors: the model is built on top of these tensors.\n          We do not expect any Numpy data to be provided when calling `fit`/etc.\n      outputs: None, a data tensor, or a list of tensors. If None, the\n        outputs will be determined by invoking `self.call()`, otherwise the\n        provided value will be used.\n      training: Boolean or None. Only relevant in symbolic mode. Specifies\n        whether to build the model's graph in inference mode (False), training\n        mode (True), or using the Keras learning phase (None).\n    Raises:\n      ValueError: If dict inputs are passed to a Sequential Model where the\n        first layer isn't FeatureLayer.\n    "
        self._set_save_spec(inputs)
        inputs = self._set_input_attrs(inputs)
        if outputs is None:
            kwargs = {}
            if self._expects_training_arg:
                if training is None and (not ops.executing_eagerly_outside_functions()):
                    training = backend.learning_phase()
                if training is not None:
                    kwargs['training'] = training
            try:
                outputs = self(inputs, **kwargs)
            except NotImplementedError:
                outputs = None
        self._set_output_attrs(outputs)

    @trackable.no_automatic_dependency_tracking
    def _set_input_attrs(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        'Sets attributes related to the inputs of the Model.'
        if self.inputs:
            raise ValueError('Model inputs are already set.')
        if self.__class__.__name__ == 'Sequential' and (not self.built):
            if tensor_util.is_tf_type(inputs):
                input_shape = (None,) + tuple(inputs.shape.as_list()[1:])
            elif isinstance(inputs, tensor_shape.TensorShape):
                input_shape = (None,) + tuple(inputs.as_list()[1:])
            elif isinstance(inputs, dict):
                if not training_utils_v1.is_feature_layer(self.layers[0]):
                    raise ValueError("Passing a dictionary input to a Sequential Model which doesn't have FeatureLayer as the first layer is an error.")
                input_shape = (None,)
            else:
                input_shape = (None,) + tuple(inputs.shape[1:])
            self._build_input_shape = input_shape
        inputs = self._maybe_cast_inputs(inputs)
        model_inputs = training_utils_v1.ModelInputs(inputs)
        inputs = model_inputs.get_symbolic_inputs()
        self.inputs = model_inputs.get_symbolic_inputs(return_single_as_list=True)
        self.input_names = model_inputs.get_input_names()
        self._feed_inputs = []
        self._feed_input_names = []
        self._feed_input_shapes = []
        for (k, v) in model_inputs.as_dict():
            if backend.is_placeholder(v):
                self._feed_input_names.append(k)
                self._feed_inputs.append(v)
                self._feed_input_shapes.append(backend.int_shape(v))
        return inputs

    @trackable.no_automatic_dependency_tracking
    def _set_output_attrs(self, outputs):
        if False:
            i = 10
            return i + 15
        'Sets attributes related to the outputs of the Model.'
        outputs = nest.flatten(outputs)
        self.outputs = outputs
        self.output_names = training_utils_v1.generic_output_names(outputs)
        self.built = True

    @property
    def _targets(self):
        if False:
            for i in range(10):
                print('nop')
        'The output target tensors for the model.'
        return [e.training_target.target for e in self._training_endpoints if e.has_training_target()]

    @property
    def _feed_targets(self):
        if False:
            for i in range(10):
                print('nop')
        return [e.training_target.target for e in self._training_endpoints if e.has_feedable_training_target()]

    @property
    def _feed_output_names(self):
        if False:
            while True:
                i = 10
        return [e.output_name for e in self._training_endpoints if e.has_feedable_training_target()]

    @property
    def _feed_output_shapes(self):
        if False:
            i = 10
            return i + 15
        return [e.feed_output_shape for e in self._training_endpoints if e.has_feedable_training_target()]

    @property
    def _feed_loss_fns(self):
        if False:
            while True:
                i = 10
        return [e.loss_fn for e in self._training_endpoints if e.has_feedable_training_target()]

    @property
    def _loss_weights_list(self):
        if False:
            for i in range(10):
                print('nop')
        return [e.loss_weight for e in self._training_endpoints]

    @property
    def _output_loss_metrics(self):
        if False:
            while True:
                i = 10
        if hasattr(self, '_training_endpoints'):
            return [e.output_loss_metric for e in self._training_endpoints if e.output_loss_metric is not None]
        return None

    @property
    def sample_weights(self):
        if False:
            i = 10
            return i + 15
        return [e.sample_weight for e in self._training_endpoints]

    @property
    def _sample_weight_modes(self):
        if False:
            print('Hello World!')
        return [e.sample_weight_mode for e in self._training_endpoints]

    @property
    def _feed_sample_weights(self):
        if False:
            i = 10
            return i + 15
        return [e.sample_weight for e in self._training_endpoints if e.sample_weight is not None]

    def _maybe_load_initial_epoch_from_ckpt(self, initial_epoch, mode):
        if False:
            print('Hello World!')
        'Maybe load initial epoch from ckpt considering possible worker recovery.\n\n    Refer to tensorflow/python/keras/distribute/worker_training_state.py\n    for more information.\n\n    Args:\n      initial_epoch: The original initial_epoch user passes in in `fit()`.\n      mode: The mode for running `model.fit()`.\n\n    Returns:\n      If the training is recovering from previous failure under multi-worker\n      training setting, return the epoch the training is supposed to continue\n      at. Otherwise, return the `initial_epoch` the user passes in.\n    '
        if self._training_state is not None:
            return self._training_state.maybe_load_initial_epoch_from_ckpt(initial_epoch, mode)
        return initial_epoch

    def _get_training_eval_metrics(self):
        if False:
            while True:
                i = 10
        'Returns all the metrics that are to be reported.\n\n    This includes the output loss metrics, compile metrics/weighted metrics,\n    add_metric metrics.\n    '
        metrics = []
        metrics.extend(getattr(self, '_output_loss_metrics', None) or [])
        metrics.extend(getattr(self, 'metrics', None) or [])
        return metrics

    def _assert_compile_was_called(self):
        if False:
            return 10
        if not self._compile_was_called:
            raise RuntimeError('You must compile your model before training/testing. Use `model.compile(optimizer, loss)`.')

    def _in_multi_worker_mode(self):
        if False:
            for i in range(10):
                print('nop')
        "Method to infer if this `Model` is working in multi-worker settings.\n\n    Multi-worker training refers to the setup where the training is\n    distributed across multiple workers, as opposed to the case where\n    only a local process performs the training. This function is\n    used to infer for example whether or not a distribute coordinator\n    should be run, and thus TensorFlow servers should be started for\n    communication with other servers in the cluster, or whether or not\n    saving/restoring checkpoints is relevant for preemption fault tolerance.\n\n    Experimental. Signature and implementation are subject to change.\n\n    Returns:\n      Whether this model indicates it's working in multi-worker settings.\n    "
        strategy = self._distribution_strategy
        if not strategy and distribute_lib.has_strategy():
            strategy = distribute_lib.get_strategy()
        return strategy and strategy.extended._in_multi_worker_mode()

    @property
    def _trackable_saved_model_saver(self):
        if False:
            while True:
                i = 10
        return model_serialization.ModelSavedModelSaver(self)

    def _get_compile_args(self, user_metrics=True):
        if False:
            return 10
        del user_metrics
        self._assert_compile_was_called()
        kwargs = {'loss': self.loss, 'metrics': self._compile_metrics, 'loss_weights': self.loss_weights, 'sample_weight_mode': self.sample_weight_mode, 'weighted_metrics': self._compile_weighted_metrics}
        return kwargs

    @property
    def _compile_was_called(self):
        if False:
            while True:
                i = 10
        return self._v1_compile_was_called

class DistributedCallbackModel(Model):
    """Model that is used for callbacks with tf.distribute.Strategy."""

    def __init__(self, model):
        if False:
            for i in range(10):
                print('nop')
        super(DistributedCallbackModel, self).__init__()
        self.optimizer = model.optimizer

    def set_original_model(self, orig_model):
        if False:
            while True:
                i = 10
        self._original_model = orig_model

    def save_weights(self, filepath, overwrite=True, save_format=None):
        if False:
            return 10
        self._replicated_model.save_weights(filepath, overwrite=overwrite, save_format=save_format)

    def save(self, filepath, overwrite=True, include_optimizer=True):
        if False:
            print('Hello World!')
        distributed_model_weights = self.get_weights()
        self._original_model.set_weights(distributed_model_weights)
        self._original_model.save(filepath, overwrite=True, include_optimizer=False)

    def load_weights(self, filepath, by_name=False):
        if False:
            for i in range(10):
                print('nop')
        self._original_model.load_weights(filepath, by_name=False)
        orig_model_weights = self._original_model.get_weights()
        distributed_training_utils_v1.set_weights(self._original_model._distribution_strategy, self, orig_model_weights)

    def __getattr__(self, item):
        if False:
            while True:
                i = 10
        if item not in ('_setattr_tracking', '_layers'):
            logging.warning('You are accessing attribute ' + item + ' of the DistributedCallbackModel that may not have been set correctly.')
        return super(DistributedCallbackModel, self).__getattr__(item)

class _TrainingEndpoint(object):
    """A container for the training output/target and related entities.

  In the case of model with multiple outputs, there is a one-to-one mapping
  between model output (y_pred), model target (y_true), loss, metrics etc.
  By unifying these entities into one class, different entity can access
  information between each other, rather than currently access different list of
  attributes of the model.
  """

    def __init__(self, output, output_name, loss_fn, loss_weight=None, training_target=None, output_loss_metric=None, sample_weight=None, sample_weight_mode=None):
        if False:
            while True:
                i = 10
        "Initialize the _TrainingEndpoint.\n\n    Note that the output and output_name should be stable as long as the model\n    structure doesn't change. The training_target suppose to be mutable since\n    the information is provided via `compile()`\n\n    Args:\n      output: the output tensor of the model.\n      output_name: the unique name of the output tensor.\n      loss_fn: the loss function for the output tensor.\n      loss_weight: float, the weights for the loss.\n      training_target: the _TrainingTarget for the model.\n      output_loss_metric: the metric object for the loss function.\n      sample_weight: the weights for how a sample is weighted during metric and\n        loss calculation. Could be None.\n      sample_weight_mode: string, 'temporal', 'samplewise' or None. The mode for\n        how the sample_weight is populated.\n    "
        self._output = output
        self._output_name = output_name
        self._loss_fn = loss_fn
        self._loss_weight = loss_weight
        self._training_target = training_target
        self._output_loss_metric = output_loss_metric
        self._sample_weight = sample_weight
        self._sample_weight_mode = sample_weight_mode

    @property
    def output(self):
        if False:
            print('Hello World!')
        return self._output

    @property
    def output_name(self):
        if False:
            return 10
        return self._output_name

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        return backend.int_shape(self.output)

    @property
    def loss_fn(self):
        if False:
            while True:
                i = 10
        return self._loss_fn

    @property
    def loss_weight(self):
        if False:
            return 10
        return self._loss_weight

    @loss_weight.setter
    def loss_weight(self, value):
        if False:
            while True:
                i = 10
        self._loss_weight = value

    @property
    def training_target(self):
        if False:
            for i in range(10):
                print('nop')
        return self._training_target

    @training_target.setter
    def training_target(self, value):
        if False:
            i = 10
            return i + 15
        self._training_target = value

    def create_training_target(self, target, run_eagerly=False):
        if False:
            i = 10
            return i + 15
        'Create training_target instance and update the self.training_target.\n\n    Note that the input target should just be a tensor or None, and\n    corresponding training target will be created based on the output and\n    loss_fn.\n\n    Args:\n      target: the target tensor for the current output. Could be None.\n      run_eagerly: boolean, whether the model is in run_eagerly mode.\n\n    Raises:\n      ValueError if the training_target field for the current instance has\n      already been populated.\n    '
        if self.has_training_target():
            raise ValueError('The training_target field for the _TrainingEndpoint instance has already been populated')
        if run_eagerly:
            self.training_target = _TrainingTarget(None, feedable=True, skip_target_weights=False)
            return
        if self.should_skip_target():
            self.training_target = _TrainingTarget(None)
        else:
            if target is not None and (not backend.is_placeholder(target)):
                feedable = False
                skip_target_weights = True
            else:
                feedable = True
                skip_target_weights = False
            if target is None:
                target_dtype = losses.LABEL_DTYPES_FOR_LOSSES.get(self.loss_fn, backend.dtype(self.output))
                target = backend.placeholder(ndim=len(self.shape), name=self.output_name + '_target', sparse=backend.is_sparse(self.output), dtype=target_dtype)
            self.training_target = _TrainingTarget(target, feedable=feedable, skip_target_weights=skip_target_weights)

    @property
    def output_loss_metric(self):
        if False:
            i = 10
            return i + 15
        return self._output_loss_metric

    @output_loss_metric.setter
    def output_loss_metric(self, value):
        if False:
            while True:
                i = 10
        self._output_loss_metric = value

    @property
    def sample_weight(self):
        if False:
            return 10
        return self._sample_weight

    @sample_weight.setter
    def sample_weight(self, value):
        if False:
            print('Hello World!')
        self._sample_weight = value

    @property
    def sample_weight_mode(self):
        if False:
            i = 10
            return i + 15
        return self._sample_weight_mode

    @sample_weight_mode.setter
    def sample_weight_mode(self, value):
        if False:
            return 10
        self._sample_weight_mode = value

    def should_skip_target(self):
        if False:
            return 10
        return self._loss_fn is None

    def should_skip_target_weights(self):
        if False:
            print('Hello World!')
        return self.should_skip_target() or self.training_target is None or self.training_target.skip_target_weights

    def has_training_target(self):
        if False:
            for i in range(10):
                print('nop')
        return self.training_target is not None

    def has_feedable_training_target(self):
        if False:
            while True:
                i = 10
        return not self.should_skip_target() and self.training_target is not None and self.training_target.feedable

    def loss_name(self):
        if False:
            print('Hello World!')
        if self._loss_fn is not None:
            return self._output_name + '_loss'
        return None

    @property
    def feed_output_shape(self):
        if False:
            while True:
                i = 10
        'The output shape for the feedable target.'
        if not self.has_feedable_training_target():
            return None
        if isinstance(self.loss_fn, losses.LossFunctionWrapper) and self.loss_fn.fn == losses.sparse_categorical_crossentropy or isinstance(self.loss_fn, losses.SparseCategoricalCrossentropy):
            if backend.image_data_format() == 'channels_first':
                return (self.shape[0], 1) + self.shape[2:]
            else:
                return self.shape[:-1] + (1,)
        elif not isinstance(self.loss_fn, losses.Loss) or (isinstance(self.loss_fn, losses.LossFunctionWrapper) and getattr(losses, self.loss_fn.fn.__name__, None) is None):
            return None
        else:
            return self.shape

    def sample_weights_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        'Check if the sample weight and the mode match or not.'
        return self.sample_weight_mode is not None and self.sample_weight is None or (self.sample_weight_mode is None and self.sample_weight is not None)

    def populate_sample_weight(self, sample_weight, sample_weight_mode):
        if False:
            print('Hello World!')
        'Populate the sample weight and based on the sample weight mode.'
        if sample_weight is None and (self.should_skip_target_weights() or sample_weight_mode is None or context.executing_eagerly()):
            self._sample_weight = None
            return
        assert sample_weight_mode in ['temporal', 'samplewise']
        if sample_weight_mode == 'temporal':
            default_value = [[1.0]]
            shape = [None, None]
        else:
            default_value = [1.0]
            shape = [None]
        if sample_weight is not None:
            if not sample_weight.shape.is_compatible_with(shape):
                raise ValueError('Received sample weight with shape {}. Expected shape {}.'.format(sample_weight.shape, shape))
            self._sample_weight = sample_weight
        else:
            self._sample_weight = array_ops.placeholder_with_default(constant_op.constant(default_value, dtype=backend.floatx()), shape=shape, name=self.output_name + '_sample_weights')

class _TrainingTarget(object):
    """Container for a target tensor (y_true) and its metadata (shape, loss...).

  Args:
    target: A target tensor for the model. It may be `None` if the
      output is excluded from loss computation. It is still kept as None
      since each output of the model should have a corresponding target. If
      the target is None, the rest of the attributes will be None as well.
    feedable: Boolean, whether the target is feedable (requires data to be
      passed in `fit` or `train_on_batch`), or not (model compiled with
      `target_tensors` argument).
    skip_target_weights: Boolean, whether the target should be skipped during
      weights calculation.
  """

    def __init__(self, target, feedable=False, skip_target_weights=True):
        if False:
            i = 10
            return i + 15
        self._target = target
        self._feedable = feedable
        self._skip_target_weights = skip_target_weights

    @property
    def target(self):
        if False:
            return 10
        return self._target

    @property
    def feedable(self):
        if False:
            return 10
        return self._feedable

    @property
    def skip_target_weights(self):
        if False:
            return 10
        return self._skip_target_weights

def _is_symbolic_tensor(x):
    if False:
        i = 10
        return i + 15
    return tensor_util.is_tf_type(x)

def _convert_scipy_sparse_tensor(value, expected_input):
    if False:
        while True:
            i = 10
    "Handle scipy sparse tensor conversions.\n\n  This method takes a value 'value' and returns the proper conversion. If\n  value is a scipy sparse tensor and the expected input is a dense tensor,\n  we densify 'value'. If value is a scipy sparse tensor and the expected input\n  is a TF SparseTensor, we convert 'value' to a SparseTensor. If 'value' is\n  not a scipy sparse tensor, or scipy is not imported, we pass it through\n  unchanged.\n\n  Args:\n    value: An object that may be a scipy sparse tensor\n    expected_input: The expected input placeholder.\n\n  Returns:\n    The possibly-converted 'value'.\n  "
    if issparse is not None and issparse(value):
        if backend.is_sparse(expected_input):
            sparse_coo = value.tocoo()
            (row, col) = (sparse_coo.row, sparse_coo.col)
            (data, shape) = (sparse_coo.data, sparse_coo.shape)
            indices = np.concatenate((np.expand_dims(row, 1), np.expand_dims(col, 1)), 1)
            return sparse_tensor.SparseTensor(indices, data, shape)
        else:
            if ops.executing_eagerly_outside_functions():
                raise ValueError('A SciPy sparse matrix was passed to a model that expects dense inputs. Please densify your inputs first, such as by calling `x.toarray().')
            return value.toarray()
    else:
        return value

def _get_metrics_from_layers(layers):
    if False:
        i = 10
        return i + 15
    'Returns list of metrics from the given layers.\n\n  This will not include the `compile` metrics of a model layer.\n\n  Args:\n    layers: List of layers.\n\n  Returns:\n    List of metrics.\n  '
    metrics = []
    layers = layer_utils.filter_empty_layer_containers(layers)
    for layer in layers:
        if isinstance(layer, Model):
            metrics.extend(layer._metrics)
            metrics.extend(_get_metrics_from_layers(layer.layers))
        else:
            metrics.extend(layer.metrics)
    return metrics

def _non_none_constant_value(v):
    if False:
        return 10
    constant_value = tensor_util.constant_value(v)
    return constant_value if constant_value is not None else v