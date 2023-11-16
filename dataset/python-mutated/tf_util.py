import os
import collections
import functools
import multiprocessing
from typing import Set
import numpy as np
import tensorflow as tf

def is_image(tensor):
    if False:
        return 10
    '\n    Check if a tensor has the shape of\n    a valid image for tensorboard logging.\n    Valid image: RGB, RGBD, GrayScale\n\n    :param tensor: (np.ndarray or tf.placeholder)\n    :return: (bool)\n    '
    return len(tensor.shape) == 3 and tensor.shape[-1] in [1, 3, 4]

def batch_to_seq(tensor_batch, n_batch, n_steps, flat=False):
    if False:
        return 10
    '\n    Transform a batch of Tensors, into a sequence of Tensors for recurrent policies\n\n    :param tensor_batch: (TensorFlow Tensor) The input tensor to unroll\n    :param n_batch: (int) The number of batch to run (n_envs * n_steps)\n    :param n_steps: (int) The number of steps to run for each environment\n    :param flat: (bool) If the input Tensor is flat\n    :return: (TensorFlow Tensor) sequence of Tensors for recurrent policies\n    '
    if flat:
        tensor_batch = tf.reshape(tensor_batch, [n_batch, n_steps])
    else:
        tensor_batch = tf.reshape(tensor_batch, [n_batch, n_steps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=n_steps, value=tensor_batch)]

def seq_to_batch(tensor_sequence, flat=False):
    if False:
        while True:
            i = 10
    '\n    Transform a sequence of Tensors, into a batch of Tensors for recurrent policies\n\n    :param tensor_sequence: (TensorFlow Tensor) The input tensor to batch\n    :param flat: (bool) If the input Tensor is flat\n    :return: (TensorFlow Tensor) batch of Tensors for recurrent policies\n    '
    shape = tensor_sequence[0].get_shape().as_list()
    if not flat:
        assert len(shape) > 1
        n_hidden = tensor_sequence[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=tensor_sequence), [-1, n_hidden])
    else:
        return tf.reshape(tf.stack(values=tensor_sequence, axis=1), [-1])

def check_shape(tensors, shapes):
    if False:
        for i in range(10):
            print('nop')
    '\n    Verifies the tensors match the given shape, will raise an error if the shapes do not match\n\n    :param tensors: ([TensorFlow Tensor]) The tensors that should be checked\n    :param shapes: ([list]) The list of shapes for each tensor\n    '
    i = 0
    for (tensor, shape) in zip(tensors, shapes):
        assert tensor.get_shape().as_list() == shape, 'id ' + str(i) + ' shape ' + str(tensor.get_shape()) + str(shape)
        i += 1

def huber_loss(tensor, delta=1.0):
    if False:
        return 10
    '\n    Reference: https://en.wikipedia.org/wiki/Huber_loss\n\n    :param tensor: (TensorFlow Tensor) the input value\n    :param delta: (float) Huber loss delta value\n    :return: (TensorFlow Tensor) Huber loss output\n    '
    return tf.where(tf.abs(tensor) < delta, tf.square(tensor) * 0.5, delta * (tf.abs(tensor) - 0.5 * delta))

def sample(logits):
    if False:
        while True:
            i = 10
    '\n    Creates a sampling Tensor for non deterministic policies\n    when using categorical distribution.\n    It uses the Gumbel-max trick: http://amid.fish/humble-gumbel\n\n    :param logits: (TensorFlow Tensor) The input probability for each action\n    :return: (TensorFlow Tensor) The sampled action\n    '
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)

def calc_entropy(logits):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculates the entropy of the output values of the network\n\n    :param logits: (TensorFlow Tensor) The input probability for each action\n    :return: (TensorFlow Tensor) The Entropy of the output values of the network\n    '
    a_0 = logits - tf.reduce_max(logits, 1, keepdims=True)
    exp_a_0 = tf.exp(a_0)
    z_0 = tf.reduce_sum(exp_a_0, 1, keepdims=True)
    p_0 = exp_a_0 / z_0
    return tf.reduce_sum(p_0 * (tf.log(z_0) - a_0), 1)

def mse(pred, target):
    if False:
        i = 10
        return i + 15
    '\n    Returns the Mean squared error between prediction and target\n\n    :param pred: (TensorFlow Tensor) The predicted value\n    :param target: (TensorFlow Tensor) The target value\n    :return: (TensorFlow Tensor) The Mean squared error between prediction and target\n    '
    return tf.reduce_mean(tf.square(pred - target))

def avg_norm(tensor):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return an average of the L2 normalization of the batch\n\n    :param tensor: (TensorFlow Tensor) The input tensor\n    :return: (TensorFlow Tensor) Average L2 normalization of the batch\n    '
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=-1)))

def gradient_add(grad_1, grad_2, param, verbose=0):
    if False:
        while True:
            i = 10
    '\n    Sum two gradients\n\n    :param grad_1: (TensorFlow Tensor) The first gradient\n    :param grad_2: (TensorFlow Tensor) The second gradient\n    :param param: (TensorFlow parameters) The trainable parameters\n    :param verbose: (int) verbosity level\n    :return: (TensorFlow Tensor) the sum of the gradients\n    '
    if verbose > 1:
        print([grad_1, grad_2, param.name])
    if grad_1 is None and grad_2 is None:
        return None
    elif grad_1 is None:
        return grad_2
    elif grad_2 is None:
        return grad_1
    else:
        return grad_1 + grad_2

def q_explained_variance(q_pred, q_true):
    if False:
        i = 10
        return i + 15
    '\n    Calculates the explained variance of the Q value\n\n    :param q_pred: (TensorFlow Tensor) The predicted Q value\n    :param q_true: (TensorFlow Tensor) The expected Q value\n    :return: (TensorFlow Tensor) the explained variance of the Q value\n    '
    (_, var_y) = tf.nn.moments(q_true, axes=[0, 1])
    (_, var_pred) = tf.nn.moments(q_true - q_pred, axes=[0, 1])
    check_shape([var_y, var_pred], [[]] * 2)
    return 1.0 - var_pred / var_y

def make_session(num_cpu=None, make_default=False, graph=None):
    if False:
        while True:
            i = 10
    "\n    Returns a session that will use <num_cpu> CPU's only\n\n    :param num_cpu: (int) number of CPUs to use for TensorFlow\n    :param make_default: (bool) if this should return an InteractiveSession or a normal Session\n    :param graph: (TensorFlow Graph) the graph of the session\n    :return: (TensorFlow session)\n    "
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    tf_config = tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=num_cpu, intra_op_parallelism_threads=num_cpu)
    tf_config.gpu_options.allow_growth = True
    if make_default:
        return tf.InteractiveSession(config=tf_config, graph=graph)
    else:
        return tf.Session(config=tf_config, graph=graph)

def single_threaded_session(make_default=False, graph=None):
    if False:
        return 10
    '\n    Returns a session which will only use a single CPU\n\n    :param make_default: (bool) if this should return an InteractiveSession or a normal Session\n    :param graph: (TensorFlow Graph) the graph of the session\n    :return: (TensorFlow session)\n    '
    return make_session(num_cpu=1, make_default=make_default, graph=graph)

def in_session(func):
    if False:
        for i in range(10):
            print('nop')
    '\n    Wraps a function so that it is in a TensorFlow Session\n\n    :param func: (function) the function to wrap\n    :return: (function)\n    '

    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        if False:
            while True:
                i = 10
        with tf.Session():
            func(*args, **kwargs)
    return newfunc
ALREADY_INITIALIZED = set()

def initialize(sess=None):
    if False:
        return 10
    '\n    Initialize all the uninitialized variables in the global scope.\n\n    :param sess: (TensorFlow Session)\n    '
    if sess is None:
        sess = tf.get_default_session()
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    sess.run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)

def function(inputs, outputs, updates=None, givens=None):
    if False:
        print('Hello World!')
    '\n    Take a bunch of tensorflow placeholders and expressions\n    computed based on those placeholders and produces f(inputs) -> outputs. Function f takes\n    values to be fed to the input\'s placeholders and produces the values of the expressions\n    in outputs. Just like a Theano function.\n\n    Input values can be passed in the same order as inputs or can be provided as kwargs based\n    on placeholder name (passed to constructor or accessible via placeholder.op.name).\n\n    Example:\n       >>> x = tf.placeholder(tf.int32, (), name="x")\n       >>> y = tf.placeholder(tf.int32, (), name="y")\n       >>> z = 3 * x + 2 * y\n       >>> lin = function([x, y], z, givens={y: 0})\n       >>> with single_threaded_session():\n       >>>     initialize()\n       >>>     assert lin(2) == 6\n       >>>     assert lin(x=3) == 9\n       >>>     assert lin(2, 2) == 10\n\n    :param inputs: (TensorFlow Tensor or Object with make_feed_dict) list of input arguments\n    :param outputs: (TensorFlow Tensor) list of outputs or a single output to be returned from function. Returned\n        value will also have the same shape.\n    :param updates: ([tf.Operation] or tf.Operation)\n        list of update functions or single update function that will be run whenever\n        the function is called. The return is ignored.\n    :param givens: (dict) the values known for the output\n    '
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        func = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), func(*args, **kwargs)))
    else:
        func = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: func(*args, **kwargs)[0]

class _Function(object):

    def __init__(self, inputs, outputs, updates, givens):
        if False:
            while True:
                i = 10
        '\n        Theano like function\n\n        :param inputs: (TensorFlow Tensor or Object with make_feed_dict) list of input arguments\n        :param outputs: (TensorFlow Tensor) list of outputs or a single output to be returned from function. Returned\n            value will also have the same shape.\n        :param updates: ([tf.Operation] or tf.Operation)\n        list of update functions or single update function that will be run whenever\n        the function is called. The return is ignored.\n        :param givens: (dict) the values known for the output\n        '
        for inpt in inputs:
            if not hasattr(inpt, 'make_feed_dict') and (not (isinstance(inpt, tf.Tensor) and len(inpt.op.inputs) == 0)):
                assert False, 'inputs should all be placeholders, constants, or have a make_feed_dict method'
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    @classmethod
    def _feed_input(cls, feed_dict, inpt, value):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = value

    def __call__(self, *args, sess=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert len(args) <= len(self.inputs), 'Too many arguments provided'
        if sess is None:
            sess = tf.get_default_session()
        feed_dict = {}
        for (inpt, value) in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = sess.run(self.outputs_update, feed_dict=feed_dict, **kwargs)[:-1]
        return results

def var_shape(tensor):
    if False:
        i = 10
        return i + 15
    '\n    get TensorFlow Tensor shape\n\n    :param tensor: (TensorFlow Tensor) the input tensor\n    :return: ([int]) the shape\n    '
    out = tensor.get_shape().as_list()
    assert all((isinstance(a, int) for a in out)), 'shape function assumes that shape is fully known'
    return out

def numel(tensor):
    if False:
        for i in range(10):
            print('nop')
    "\n    get TensorFlow Tensor's number of elements\n\n    :param tensor: (TensorFlow Tensor) the input tensor\n    :return: (int) the number of elements\n    "
    return intprod(var_shape(tensor))

def intprod(tensor):
    if False:
        return 10
    '\n    calculates the product of all the elements in a list\n\n    :param tensor: ([Number]) the list of elements\n    :return: (int) the product truncated\n    '
    return int(np.prod(tensor))

def flatgrad(loss, var_list, clip_norm=None):
    if False:
        while True:
            i = 10
    '\n    calculates the gradient and flattens it\n\n    :param loss: (float) the loss value\n    :param var_list: ([TensorFlow Tensor]) the variables\n    :param clip_norm: (float) clip the gradients (disabled if None)\n    :return: ([TensorFlow Tensor]) flattened gradient\n    '
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)]) for (v, grad) in zip(var_list, grads)])

class SetFromFlat(object):

    def __init__(self, var_list, dtype=tf.float32, sess=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the parameters from a flat vector\n\n        :param var_list: ([TensorFlow Tensor]) the variables\n        :param dtype: (type) the type for the placeholder\n        :param sess: (TensorFlow Session)\n        '
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])
        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, _var) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.assign(_var, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.operation = tf.group(*assigns)
        self.sess = sess

    def __call__(self, theta):
        if False:
            return 10
        if self.sess is None:
            return tf.get_default_session().run(self.operation, feed_dict={self.theta: theta})
        else:
            return self.sess.run(self.operation, feed_dict={self.theta: theta})

class GetFlat(object):

    def __init__(self, var_list, sess=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the parameters as a flat vector\n\n        :param var_list: ([TensorFlow Tensor]) the variables\n        :param sess: (TensorFlow Session)\n        '
        self.operation = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
        self.sess = sess

    def __call__(self):
        if False:
            i = 10
            return i + 15
        if self.sess is None:
            return tf.get_default_session().run(self.operation)
        else:
            return self.sess.run(self.operation)

def get_trainable_vars(name):
    if False:
        i = 10
        return i + 15
    '\n    returns the trainable variables\n\n    :param name: (str) the scope\n    :return: ([TensorFlow Variable])\n    '
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

def get_globals_vars(name):
    if False:
        return 10
    '\n    returns the trainable variables\n\n    :param name: (str) the scope\n    :return: ([TensorFlow Variable])\n    '
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

def outer_scope_getter(scope, new_scope=''):
    if False:
        for i in range(10):
            print('nop')
    '\n    remove a scope layer for the getter\n\n    :param scope: (str) the layer to remove\n    :param new_scope: (str) optional replacement name\n    :return: (function (function, str, ``*args``, ``**kwargs``): Tensorflow Tensor)\n    '

    def _getter(getter, name, *args, **kwargs):
        if False:
            while True:
                i = 10
        name = name.replace(scope + '/', new_scope, 1)
        val = getter(name, *args, **kwargs)
        return val
    return _getter

def total_episode_reward_logger(rew_acc, rewards, masks, writer, steps):
    if False:
        for i in range(10):
            print('nop')
    '\n    calculates the cumulated episode reward, and prints to tensorflow log the output\n\n    :param rew_acc: (np.array float) the total running reward\n    :param rewards: (np.array float) the rewards\n    :param masks: (np.array bool) the end of episodes\n    :param writer: (TensorFlow Session.writer) the writer to log to\n    :param steps: (int) the current timestep\n    :return: (np.array float) the updated total running reward\n    :return: (np.array float) the updated total running reward\n    '
    with tf.variable_scope('environment_info', reuse=True):
        for env_idx in range(rewards.shape[0]):
            dones_idx = np.sort(np.argwhere(masks[env_idx]))
            if len(dones_idx) == 0:
                rew_acc[env_idx] += sum(rewards[env_idx])
            else:
                rew_acc[env_idx] += sum(rewards[env_idx, :dones_idx[0, 0]])
                summary = tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=rew_acc[env_idx])])
                writer.add_summary(summary, steps + dones_idx[0, 0])
                for k in range(1, len(dones_idx[:, 0])):
                    rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[k - 1, 0]:dones_idx[k, 0]])
                    summary = tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=rew_acc[env_idx])])
                    writer.add_summary(summary, steps + dones_idx[k, 0])
                rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[-1, 0]:])
    return rew_acc