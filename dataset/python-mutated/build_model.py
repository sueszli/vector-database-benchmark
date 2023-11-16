"""Defines convolutional model graph for Seq2Species.

Builds TensorFlow computation graph for predicting the given taxonomic target
labels from short reads of DNA using convolutional filters, followed by
fully-connected layers and a softmax output layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import math
import tensorflow as tf
import input as seq2species_input
import seq2label_utils

class ConvolutionalNet(object):
    """Class to build and store the model's computational graph and operations.

  Attributes:
    read_length: int; the length in basepairs of the input reads of DNA.
    placeholders: dict; mapping from name to tf.Placeholder.
    global_step: tf.Variable tracking number of training iterations performed.
    train_op: operation to perform one training step by gradient descent.
    summary_op: operation to log model's performance metrics to TF event files.
    accuracy: tf.Variable giving the model's read-level accuracy for the
      current inputs.
    weighted_accuracy: tf.Variable giving the model's read-level weighted
      accuracy for the current inputs.
    loss: tf.Variable giving the model's current cross entropy loss.
    logits: tf.Variable containing the model's logits for the current inputs.
    predictions: tf.Variable containing the model's current predicted
      probability distributions for the current inputs.
    possible_labels: a dict of possible label values (list of strings), keyed by
      target name.  Labels in the lists are the order used for integer encoding.
    use_tpu: whether model is to be run on TPU.
  """

    def __init__(self, hparams, dataset_info, targets, use_tpu=False):
        if False:
            return 10
        "Initializes the ConvolutionalNet according to provided hyperparameters.\n\n    Does not build the graph---this is done by calling `build_graph` on the\n    constructed object or using `model_fn`.\n\n    Args:\n      hparams: tf.contrib.training.Hparams object containing the model's\n        hyperparamters; see configuration.py for hyperparameter definitions.\n      dataset_info: a `Seq2LabelDatasetInfo` message reflecting the dataset\n        metadata.\n      targets: list of strings: the names of the prediction targets.\n      use_tpu: whether we are running on TPU; if True, summaries will be\n        disabled.\n    "
        self._placeholders = {}
        self._targets = targets
        self._dataset_info = dataset_info
        self._hparams = hparams
        all_label_values = seq2label_utils.get_all_label_values(self.dataset_info)
        self._possible_labels = {target: all_label_values[target] for target in self.targets}
        self._use_tpu = use_tpu

    @property
    def hparams(self):
        if False:
            print('Hello World!')
        return self._hparams

    @property
    def dataset_info(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dataset_info

    @property
    def possible_labels(self):
        if False:
            return 10
        return self._possible_labels

    @property
    def bases(self):
        if False:
            print('Hello World!')
        return seq2species_input.DNA_BASES

    @property
    def n_bases(self):
        if False:
            print('Hello World!')
        return seq2species_input.NUM_DNA_BASES

    @property
    def targets(self):
        if False:
            i = 10
            return i + 15
        return self._targets

    @property
    def read_length(self):
        if False:
            return 10
        return self.dataset_info.read_length

    @property
    def placeholders(self):
        if False:
            for i in range(10):
                print('nop')
        return self._placeholders

    @property
    def global_step(self):
        if False:
            for i in range(10):
                print('nop')
        return self._global_step

    @property
    def train_op(self):
        if False:
            for i in range(10):
                print('nop')
        return self._train_op

    @property
    def summary_op(self):
        if False:
            for i in range(10):
                print('nop')
        return self._summary_op

    @property
    def accuracy(self):
        if False:
            print('Hello World!')
        return self._accuracy

    @property
    def weighted_accuracy(self):
        if False:
            return 10
        return self._weighted_accuracy

    @property
    def loss(self):
        if False:
            i = 10
            return i + 15
        return self._loss

    @property
    def total_loss(self):
        if False:
            while True:
                i = 10
        return self._total_loss

    @property
    def logits(self):
        if False:
            i = 10
            return i + 15
        return self._logits

    @property
    def predictions(self):
        if False:
            while True:
                i = 10
        return self._predictions

    @property
    def use_tpu(self):
        if False:
            for i in range(10):
                print('nop')
        return self._use_tpu

    def _summary_scalar(self, name, scalar):
        if False:
            print('Hello World!')
        'Adds a summary scalar, if the platform supports summaries.'
        if not self.use_tpu:
            return tf.summary.scalar(name, scalar)
        else:
            return None

    def _summary_histogram(self, name, values):
        if False:
            for i in range(10):
                print('nop')
        'Adds a summary histogram, if the platform supports summaries.'
        if not self.use_tpu:
            return tf.summary.histogram(name, values)
        else:
            return None

    def _init_weights(self, shape, scale=1.0, name='weights'):
        if False:
            return 10
        'Randomly initializes a weight Tensor of the given shape.\n\n    Args:\n      shape: list; desired Tensor dimensions.\n      scale: float; standard deviation scale with which to initialize weights.\n      name: string name for the variable.\n\n    Returns:\n      TF Variable contining truncated random Normal initialized weights.\n    '
        num_inputs = shape[0] if len(shape) < 3 else shape[0] * shape[1] * shape[2]
        stddev = scale / math.sqrt(num_inputs)
        return tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(0.0, stddev))

    def _init_bias(self, size):
        if False:
            for i in range(10):
                print('nop')
        'Initializes bias vector of given shape as zeros.\n\n    Args:\n      size: int; desired size of bias Tensor.\n\n    Returns:\n      TF Variable containing the initialized biases.\n    '
        return tf.get_variable(name='b_{}'.format(size), shape=[size], initializer=tf.zeros_initializer())

    def _add_summaries(self, mode, gradient_norm, parameter_norm):
        if False:
            print('Hello World!')
        'Defines TensorFlow operation for logging summaries to event files.\n\n    Args:\n      mode: the ModeKey string.\n      gradient_norm: Tensor; norm of gradients produced during the current\n        training operation.\n      parameter_norm: Tensor; norm of the model parameters produced during the\n        current training operation.\n    '
        if mode == tf.estimator.ModeKeys.TRAIN:
            self._summary_scalar('norm_of_gradients', gradient_norm)
            self._summary_scalar('norm_of_parameters', parameter_norm)
            self._summary_scalar('total_loss', self.total_loss)
            self._summary_scalar('learning_rate', self._learn_rate)
            for target in self.targets:
                self._summary_scalar('per_read_weighted_accuracy/{}'.format(target), self.weighted_accuracy[target])
                self._summary_scalar('per_read_accuracy/{}'.format(target), self.accuracy[target])
                self._summary_histogram('prediction_frequency/{}'.format(target), self._predictions[target])
                self._summary_scalar('cross_entropy_loss/{}'.format(target), self._loss[target])
            self._summary_op = tf.summary.merge_all()
        else:
            summaries = []
            for target in self.targets:
                accuracy_ph = tf.placeholder(tf.float32, shape=())
                weighted_accuracy_ph = tf.placeholder(tf.float32, shape=())
                cross_entropy_ph = tf.placeholder(tf.float32, shape=())
                self._placeholders.update({'accuracy/{}'.format(target): accuracy_ph, 'weighted_accuracy/{}'.format(target): weighted_accuracy_ph, 'cross_entropy/{}'.format(target): cross_entropy_ph})
                summaries += [self._summary_scalar('cross_entropy_loss/{}'.format(target), cross_entropy_ph), self._summary_scalar('per_read_accuracy/{}'.format(target), accuracy_ph), self._summary_scalar('per_read_weighted_accuracy/{}'.format(target), weighted_accuracy_ph)]
            self._summary_op = tf.summary.merge(summaries)

    def _convolution(self, inputs, filter_dim, pointwise_dim=None, scale=1.0, padding='SAME'):
        if False:
            for i in range(10):
                print('nop')
        'Applies convolutional filter of given dimensions to given input Tensor.\n\n    If a pointwise dimension is specified, a depthwise separable convolution is\n    performed.\n\n    Args:\n      inputs: 4D Tensor of shape (# reads, 1, # basepairs, # bases).\n      filter_dim: integer tuple of the form (width, depth).\n      pointwise_dim: int; output dimension for pointwise convolution.\n      scale: float; standard deviation scale with which to initialize weights.\n      padding: string; type of padding to use. One of "SAME" or "VALID".\n\n    Returns:\n      4D Tensor result of applying the convolutional filter to the inputs.\n    '
        in_channels = inputs.get_shape()[3].value
        (filter_width, filter_depth) = filter_dim
        filters = self._init_weights([1, filter_width, in_channels, filter_depth], scale)
        self._summary_histogram(filters.name.split(':')[0].split('/')[1], filters)
        if pointwise_dim is None:
            return tf.nn.conv2d(inputs, filters, strides=[1, 1, 1, 1], padding=padding, name='weights')
        pointwise_filters = self._init_weights([1, 1, filter_depth * in_channels, pointwise_dim], scale, name='pointwise_weights')
        self._summary_histogram(pointwise_filters.name.split(':')[0].split('/')[1], pointwise_filters)
        return tf.nn.separable_conv2d(inputs, filters, pointwise_filters, strides=[1, 1, 1, 1], padding=padding)

    def _pool(self, inputs, pooling_type):
        if False:
            for i in range(10):
                print('nop')
        'Performs pooling across width and height of the given inputs.\n\n    Args:\n      inputs: Tensor shaped (batch, height, width, channels) over which to pool.\n        In our case, height is a unitary dimension and width can be thought of\n        as the read dimension.\n      pooling_type: string; one of "avg" or "max".\n\n    Returns:\n      Tensor result of performing pooling of the given pooling_type over the\n      height and width dimensions of the given inputs.\n    '
        if pooling_type == 'max':
            return tf.reduce_max(inputs, axis=[1, 2])
        if pooling_type == 'avg':
            return tf.reduce_sum(inputs, axis=[1, 2]) / tf.to_float(tf.shape(inputs)[2])

    def _leaky_relu(self, lrelu_slope, inputs):
        if False:
            while True:
                i = 10
        'Applies leaky ReLu activation to the given inputs with the given slope.\n\n    Args:\n      lrelu_slope: float; slope value for the activation function.\n        A slope of 0.0 defines a standard ReLu activation, while a positive\n        slope defines a leaky ReLu.\n      inputs: Tensor upon which to apply the activation function.\n\n    Returns:\n      Tensor result of applying the activation function to the given inputs.\n    '
        with tf.variable_scope('leaky_relu_activation'):
            return tf.maximum(lrelu_slope * inputs, inputs)

    def _dropout(self, inputs, keep_prob):
        if False:
            return 10
        'Applies dropout to the given inputs.\n\n    Args:\n      inputs: Tensor upon which to apply dropout.\n      keep_prob: float; probability with which to randomly retain values in\n        the given input.\n\n    Returns:\n      Tensor result of applying dropout to the given inputs.\n    '
        with tf.variable_scope('dropout'):
            if keep_prob < 1.0:
                return tf.nn.dropout(inputs, keep_prob)
            return inputs

    def build_graph(self, features, labels, mode, batch_size):
        if False:
            while True:
                i = 10
        'Creates TensorFlow model graph.\n\n    Args:\n      features: a dict of input features Tensors.\n      labels: a dict (by target name) of prediction labels.\n      mode: the ModeKey string.\n      batch_size: the integer batch size.\n\n    Side Effect:\n      Adds the following key Tensors and operations as class attributes:\n        placeholders, global_step, train_op, summary_op, accuracy,\n        weighted_accuracy, loss, logits, and predictions.\n    '
        is_train = mode == tf.estimator.ModeKeys.TRAIN
        read = features['sequence']
        read = tf.expand_dims(read, 1)
        prev_out = read
        filters = zip(self.hparams.filter_widths, self.hparams.filter_depths)
        for (i, f) in enumerate(filters):
            with tf.variable_scope('convolution_' + str(i)):
                if self.hparams.use_depthwise_separable:
                    p = self.hparams.pointwise_depths[i]
                else:
                    p = None
                conv_out = self._convolution(prev_out, f, pointwise_dim=p, scale=self.hparams.weight_scale)
                conv_act_out = self._leaky_relu(self.hparams.lrelu_slope, conv_out)
                prev_out = self._dropout(conv_act_out, self.hparams.keep_prob) if is_train else conv_act_out
        for i in xrange(self.hparams.num_fc_layers):
            with tf.variable_scope('fully_connected_' + str(i)):
                biases = self._init_bias(self.hparams.num_fc_units)
                if i == 0:
                    filter_dimensions = (self.hparams.min_read_length, self.hparams.num_fc_units)
                else:
                    filter_dimensions = (1, self.hparams.num_fc_units)
                fc_out = biases + self._convolution(prev_out, filter_dimensions, scale=self.hparams.weight_scale, padding='VALID')
                self._summary_histogram(biases.name.split(':')[0].split('/')[1], biases)
                fc_act_out = self._leaky_relu(self.hparams.lrelu_slope, fc_out)
                prev_out = self._dropout(fc_act_out, self.hparams.keep_prob) if is_train else fc_act_out
        with tf.variable_scope('pool'):
            pool_out = self._pool(prev_out, self.hparams.pooling_type)
        with tf.variable_scope('output'):
            self._logits = {}
            self._predictions = {}
            self._weighted_accuracy = {}
            self._accuracy = {}
            self._loss = collections.OrderedDict()
            for target in self.targets:
                with tf.variable_scope(target):
                    label = labels[target]
                    possible_labels = self.possible_labels[target]
                    weights = self._init_weights([pool_out.get_shape()[1].value, len(possible_labels)], self.hparams.weight_scale, name='weights')
                    biases = self._init_bias(len(possible_labels))
                    self._summary_histogram(weights.name.split(':')[0].split('/')[1], weights)
                    self._summary_histogram(biases.name.split(':')[0].split('/')[1], biases)
                    logits = tf.matmul(pool_out, weights) + biases
                    predictions = tf.nn.softmax(logits)
                    gather_inds = tf.stack([tf.range(batch_size), label], axis=1)
                    self._weighted_accuracy[target] = tf.reduce_mean(tf.gather_nd(predictions, gather_inds))
                    argmax_prediction = tf.cast(tf.argmax(predictions, axis=1), tf.int32)
                    self._accuracy[target] = tf.reduce_mean(tf.to_float(tf.equal(label, argmax_prediction)))
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
                    self._loss[target] = tf.reduce_mean(losses)
                    self._logits[target] = logits
                    self._predictions[target] = predictions
        self._total_loss = tf.add_n(self._loss.values())
        self._global_step = tf.train.get_or_create_global_step()
        if self.hparams.lr_decay < 0:
            self._learn_rate = self.hparams.lr_init
        else:
            self._learn_rate = tf.train.exponential_decay(self.hparams.lr_init, self._global_step, int(self.hparams.train_steps), self.hparams.lr_decay, staircase=False)
        if self.hparams.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(self._learn_rate, self.hparams.optimizer_hp)
        elif self.hparams.optimizer == 'momentum':
            opt = tf.train.MomentumOptimizer(self._learn_rate, self.hparams.optimizer_hp)
        if self.use_tpu:
            opt = tf.contrib.tpu.CrossShardOptimizer(opt)
        (gradients, variables) = zip(*opt.compute_gradients(self._total_loss))
        (clipped_gradients, _) = tf.clip_by_global_norm(gradients, self.hparams.grad_clip_norm)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self._train_op = opt.apply_gradients(zip(clipped_gradients, variables), global_step=self._global_step)
        if not self.use_tpu:
            grad_norm = tf.global_norm(gradients) if is_train else None
            param_norm = tf.global_norm(variables) if is_train else None
            self._add_summaries(mode, grad_norm, param_norm)

    def model_fn(self, features, labels, mode, params):
        if False:
            print('Hello World!')
        'Function fulfilling the tf.estimator model_fn interface.\n\n    Args:\n      features: a dict containing the input features for prediction.\n      labels: a dict from target name to Tensor-value prediction.\n      mode: the ModeKey string.\n      params: a dictionary of parameters for building the model; current params\n        are params["batch_size"]: the integer batch size.\n\n    Returns:\n      A tf.estimator.EstimatorSpec object ready for use in training, inference.\n      or evaluation.\n    '
        self.build_graph(features, labels, mode, params['batch_size'])
        return tf.estimator.EstimatorSpec(mode, predictions=self.predictions, loss=self.total_loss, train_op=self.train_op, eval_metric_ops={})