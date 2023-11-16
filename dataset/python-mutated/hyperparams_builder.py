"""Builder function to construct tf-slim arg_scope for convolution, fc ops."""
import tensorflow as tf
from object_detection.core import freezable_batch_norm
from object_detection.protos import hyperparams_pb2
from object_detection.utils import context_manager
slim = tf.contrib.slim

class KerasLayerHyperparams(object):
    """
  A hyperparameter configuration object for Keras layers used in
  Object Detection models.
  """

    def __init__(self, hyperparams_config):
        if False:
            return 10
        'Builds keras hyperparameter config for layers based on the proto config.\n\n    It automatically converts from Slim layer hyperparameter configs to\n    Keras layer hyperparameters. Namely, it:\n    - Builds Keras initializers/regularizers instead of Slim ones\n    - sets weights_regularizer/initializer to kernel_regularizer/initializer\n    - converts batchnorm decay to momentum\n    - converts Slim l2 regularizer weights to the equivalent Keras l2 weights\n\n    Contains a hyperparameter configuration for ops that specifies kernel\n    initializer, kernel regularizer, activation. Also contains parameters for\n    batch norm operators based on the configuration.\n\n    Note that if the batch_norm parameters are not specified in the config\n    (i.e. left to default) then batch norm is excluded from the config.\n\n    Args:\n      hyperparams_config: hyperparams.proto object containing\n        hyperparameters.\n\n    Raises:\n      ValueError: if hyperparams_config is not of type hyperparams.Hyperparams.\n    '
        if not isinstance(hyperparams_config, hyperparams_pb2.Hyperparams):
            raise ValueError('hyperparams_config not of type hyperparams_pb.Hyperparams.')
        self._batch_norm_params = None
        if hyperparams_config.HasField('batch_norm'):
            self._batch_norm_params = _build_keras_batch_norm_params(hyperparams_config.batch_norm)
        self._activation_fn = _build_activation_fn(hyperparams_config.activation)
        self._op_params = {'kernel_regularizer': _build_keras_regularizer(hyperparams_config.regularizer), 'kernel_initializer': _build_initializer(hyperparams_config.initializer, build_for_keras=True), 'activation': _build_activation_fn(hyperparams_config.activation)}

    def use_batch_norm(self):
        if False:
            return 10
        return self._batch_norm_params is not None

    def batch_norm_params(self, **overrides):
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict containing batchnorm layer construction hyperparameters.\n\n    Optionally overrides values in the batchnorm hyperparam dict. Overrides\n    only apply to individual calls of this method, and do not affect\n    future calls.\n\n    Args:\n      **overrides: keyword arguments to override in the hyperparams dictionary\n\n    Returns: dict containing the layer construction keyword arguments, with\n      values overridden by the `overrides` keyword arguments.\n    '
        if self._batch_norm_params is None:
            new_batch_norm_params = dict()
        else:
            new_batch_norm_params = self._batch_norm_params.copy()
        new_batch_norm_params.update(overrides)
        return new_batch_norm_params

    def build_batch_norm(self, training=None, **overrides):
        if False:
            while True:
                i = 10
        'Returns a Batch Normalization layer with the appropriate hyperparams.\n\n    If the hyperparams are configured to not use batch normalization,\n    this will return a Keras Lambda layer that only applies tf.Identity,\n    without doing any normalization.\n\n    Optionally overrides values in the batch_norm hyperparam dict. Overrides\n    only apply to individual calls of this method, and do not affect\n    future calls.\n\n    Args:\n      training: if True, the normalization layer will normalize using the batch\n       statistics. If False, the normalization layer will be frozen and will\n       act as if it is being used for inference. If None, the layer\n       will look up the Keras learning phase at `call` time to decide what to\n       do.\n      **overrides: batch normalization construction args to override from the\n        batch_norm hyperparams dictionary.\n\n    Returns: Either a FreezableBatchNorm layer (if use_batch_norm() is True),\n      or a Keras Lambda layer that applies the identity (if use_batch_norm()\n      is False)\n    '
        if self.use_batch_norm():
            return freezable_batch_norm.FreezableBatchNorm(training=training, **self.batch_norm_params(**overrides))
        else:
            return tf.keras.layers.Lambda(tf.identity)

    def build_activation_layer(self, name='activation'):
        if False:
            for i in range(10):
                print('nop')
        'Returns a Keras layer that applies the desired activation function.\n\n    Args:\n      name: The name to assign the Keras layer.\n    Returns: A Keras lambda layer that applies the activation function\n      specified in the hyperparam config, or applies the identity if the\n      activation function is None.\n    '
        if self._activation_fn:
            return tf.keras.layers.Lambda(self._activation_fn, name=name)
        else:
            return tf.keras.layers.Lambda(tf.identity, name=name)

    def params(self, include_activation=False, **overrides):
        if False:
            while True:
                i = 10
        'Returns a dict containing the layer construction hyperparameters to use.\n\n    Optionally overrides values in the returned dict. Overrides\n    only apply to individual calls of this method, and do not affect\n    future calls.\n\n    Args:\n      include_activation: If False, activation in the returned dictionary will\n        be set to `None`, and the activation must be applied via a separate\n        layer created by `build_activation_layer`. If True, `activation` in the\n        output param dictionary will be set to the activation function\n        specified in the hyperparams config.\n      **overrides: keyword arguments to override in the hyperparams dictionary.\n\n    Returns: dict containing the layer construction keyword arguments, with\n      values overridden by the `overrides` keyword arguments.\n    '
        new_params = self._op_params.copy()
        new_params['activation'] = None
        if include_activation:
            new_params['activation'] = self._activation_fn
        if self.use_batch_norm() and self.batch_norm_params()['center']:
            new_params['use_bias'] = False
        else:
            new_params['use_bias'] = True
        new_params.update(**overrides)
        return new_params

def build(hyperparams_config, is_training):
    if False:
        i = 10
        return i + 15
    'Builds tf-slim arg_scope for convolution ops based on the config.\n\n  Returns an arg_scope to use for convolution ops containing weights\n  initializer, weights regularizer, activation function, batch norm function\n  and batch norm parameters based on the configuration.\n\n  Note that if no normalization parameters are specified in the config,\n  (i.e. left to default) then both batch norm and group norm are excluded\n  from the arg_scope.\n\n  The batch norm parameters are set for updates based on `is_training` argument\n  and conv_hyperparams_config.batch_norm.train parameter. During training, they\n  are updated only if batch_norm.train parameter is true. However, during eval,\n  no updates are made to the batch norm variables. In both cases, their current\n  values are used during forward pass.\n\n  Args:\n    hyperparams_config: hyperparams.proto object containing\n      hyperparameters.\n    is_training: Whether the network is in training mode.\n\n  Returns:\n    arg_scope_fn: A function to construct tf-slim arg_scope containing\n      hyperparameters for ops.\n\n  Raises:\n    ValueError: if hyperparams_config is not of type hyperparams.Hyperparams.\n  '
    if not isinstance(hyperparams_config, hyperparams_pb2.Hyperparams):
        raise ValueError('hyperparams_config not of type hyperparams_pb.Hyperparams.')
    normalizer_fn = None
    batch_norm_params = None
    if hyperparams_config.HasField('batch_norm'):
        normalizer_fn = slim.batch_norm
        batch_norm_params = _build_batch_norm_params(hyperparams_config.batch_norm, is_training)
    if hyperparams_config.HasField('group_norm'):
        normalizer_fn = tf.contrib.layers.group_norm
    affected_ops = [slim.conv2d, slim.separable_conv2d, slim.conv2d_transpose]
    if hyperparams_config.HasField('op') and hyperparams_config.op == hyperparams_pb2.Hyperparams.FC:
        affected_ops = [slim.fully_connected]

    def scope_fn():
        if False:
            for i in range(10):
                print('nop')
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) if batch_norm_params is not None else context_manager.IdentityContextManager():
            with slim.arg_scope(affected_ops, weights_regularizer=_build_slim_regularizer(hyperparams_config.regularizer), weights_initializer=_build_initializer(hyperparams_config.initializer), activation_fn=_build_activation_fn(hyperparams_config.activation), normalizer_fn=normalizer_fn) as sc:
                return sc
    return scope_fn

def _build_activation_fn(activation_fn):
    if False:
        print('Hello World!')
    'Builds a callable activation from config.\n\n  Args:\n    activation_fn: hyperparams_pb2.Hyperparams.activation\n\n  Returns:\n    Callable activation function.\n\n  Raises:\n    ValueError: On unknown activation function.\n  '
    if activation_fn == hyperparams_pb2.Hyperparams.NONE:
        return None
    if activation_fn == hyperparams_pb2.Hyperparams.RELU:
        return tf.nn.relu
    if activation_fn == hyperparams_pb2.Hyperparams.RELU_6:
        return tf.nn.relu6
    raise ValueError('Unknown activation function: {}'.format(activation_fn))

def _build_slim_regularizer(regularizer):
    if False:
        while True:
            i = 10
    'Builds a tf-slim regularizer from config.\n\n  Args:\n    regularizer: hyperparams_pb2.Hyperparams.regularizer proto.\n\n  Returns:\n    tf-slim regularizer.\n\n  Raises:\n    ValueError: On unknown regularizer.\n  '
    regularizer_oneof = regularizer.WhichOneof('regularizer_oneof')
    if regularizer_oneof == 'l1_regularizer':
        return slim.l1_regularizer(scale=float(regularizer.l1_regularizer.weight))
    if regularizer_oneof == 'l2_regularizer':
        return slim.l2_regularizer(scale=float(regularizer.l2_regularizer.weight))
    if regularizer_oneof is None:
        return None
    raise ValueError('Unknown regularizer function: {}'.format(regularizer_oneof))

def _build_keras_regularizer(regularizer):
    if False:
        for i in range(10):
            print('nop')
    'Builds a keras regularizer from config.\n\n  Args:\n    regularizer: hyperparams_pb2.Hyperparams.regularizer proto.\n\n  Returns:\n    Keras regularizer.\n\n  Raises:\n    ValueError: On unknown regularizer.\n  '
    regularizer_oneof = regularizer.WhichOneof('regularizer_oneof')
    if regularizer_oneof == 'l1_regularizer':
        return tf.keras.regularizers.l1(float(regularizer.l1_regularizer.weight))
    if regularizer_oneof == 'l2_regularizer':
        return tf.keras.regularizers.l2(float(regularizer.l2_regularizer.weight * 0.5))
    raise ValueError('Unknown regularizer function: {}'.format(regularizer_oneof))

def _build_initializer(initializer, build_for_keras=False):
    if False:
        for i in range(10):
            print('nop')
    'Build a tf initializer from config.\n\n  Args:\n    initializer: hyperparams_pb2.Hyperparams.regularizer proto.\n    build_for_keras: Whether the initializers should be built for Keras\n      operators. If false builds for Slim.\n\n  Returns:\n    tf initializer.\n\n  Raises:\n    ValueError: On unknown initializer.\n  '
    initializer_oneof = initializer.WhichOneof('initializer_oneof')
    if initializer_oneof == 'truncated_normal_initializer':
        return tf.truncated_normal_initializer(mean=initializer.truncated_normal_initializer.mean, stddev=initializer.truncated_normal_initializer.stddev)
    if initializer_oneof == 'random_normal_initializer':
        return tf.random_normal_initializer(mean=initializer.random_normal_initializer.mean, stddev=initializer.random_normal_initializer.stddev)
    if initializer_oneof == 'variance_scaling_initializer':
        enum_descriptor = hyperparams_pb2.VarianceScalingInitializer.DESCRIPTOR.enum_types_by_name['Mode']
        mode = enum_descriptor.values_by_number[initializer.variance_scaling_initializer.mode].name
        if build_for_keras:
            if initializer.variance_scaling_initializer.uniform:
                return tf.variance_scaling_initializer(scale=initializer.variance_scaling_initializer.factor, mode=mode.lower(), distribution='uniform')
            else:
                try:
                    return tf.variance_scaling_initializer(scale=initializer.variance_scaling_initializer.factor, mode=mode.lower(), distribution='truncated_normal')
                except ValueError:
                    truncate_constant = 0.8796256610342398
                    truncated_scale = initializer.variance_scaling_initializer.factor / (truncate_constant * truncate_constant)
                    return tf.variance_scaling_initializer(scale=truncated_scale, mode=mode.lower(), distribution='normal')
        else:
            return slim.variance_scaling_initializer(factor=initializer.variance_scaling_initializer.factor, mode=mode, uniform=initializer.variance_scaling_initializer.uniform)
    raise ValueError('Unknown initializer function: {}'.format(initializer_oneof))

def _build_batch_norm_params(batch_norm, is_training):
    if False:
        return 10
    'Build a dictionary of batch_norm params from config.\n\n  Args:\n    batch_norm: hyperparams_pb2.ConvHyperparams.batch_norm proto.\n    is_training: Whether the models is in training mode.\n\n  Returns:\n    A dictionary containing batch_norm parameters.\n  '
    batch_norm_params = {'decay': batch_norm.decay, 'center': batch_norm.center, 'scale': batch_norm.scale, 'epsilon': batch_norm.epsilon, 'is_training': is_training and batch_norm.train}
    return batch_norm_params

def _build_keras_batch_norm_params(batch_norm):
    if False:
        print('Hello World!')
    'Build a dictionary of Keras BatchNormalization params from config.\n\n  Args:\n    batch_norm: hyperparams_pb2.ConvHyperparams.batch_norm proto.\n\n  Returns:\n    A dictionary containing Keras BatchNormalization parameters.\n  '
    batch_norm_params = {'momentum': batch_norm.decay, 'center': batch_norm.center, 'scale': batch_norm.scale, 'epsilon': batch_norm.epsilon}
    return batch_norm_params