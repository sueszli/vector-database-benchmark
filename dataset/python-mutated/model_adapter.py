"""Implementation of the ModelAdapter class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mock
import tensorflow as tf
from learned_optimizer.problems import problem_generator as pg

class ModelAdapter(pg.Problem):
    """Adapts Tensorflow models/graphs into a form suitable for meta-training.

  This class adapts an existing TensorFlow graph into a form suitable for
  meta-training a learned optimizer.
  """

    def __init__(self, make_loss_and_init_fn):
        if False:
            while True:
                i = 10
        "Wraps a model in the Problem interface.\n\n    make_loss_and_init argument is a callable that returns a tuple of\n    two other callables as follows.\n\n    The first will construct most of the graph and return the problem loss. It\n    is essential that this graph contains the totality of the model's variables,\n    but none of its queues.\n\n    The second will return construct the model initialization graph given a list\n    of parameters and return a callable that is passed an instance of\n    tf.Session, and should initialize the models' parameters.\n\n    An argument value function would look like this:\n\n    ```python\n    def make_loss_and_init_fn():\n      inputs = queued_reader()\n\n      def make_loss():\n        return create_model_with_variables(inputs)\n\n      def make_init_fn(parameters):\n        saver = tf.Saver(parameters)\n        def init_fn(sess):\n          sess.restore(sess, ...)\n        return init_fn\n\n      return make_loss, make_init_fn\n    ```\n\n    Args:\n      make_loss_and_init_fn: a callable, as described aboce\n    "
        (make_loss_fn, make_init_fn) = make_loss_and_init_fn()
        self.make_loss_fn = make_loss_fn
        (self.parameters, self.constants) = _get_variables(make_loss_fn)
        if make_init_fn is not None:
            init_fn = make_init_fn(self.parameters + self.constants)
        else:
            init_op = tf.initialize_variables(self.parameters + self.constants)
            init_fn = lambda sess: sess.run(init_op)
        tf.logging.info('ModelAdapter parameters: %s', [op.name for op in self.parameters])
        tf.logging.info('ModelAdapter constants: %s', [op.name for op in self.constants])
        super(ModelAdapter, self).__init__([], random_seed=None, noise_stdev=0.0, init_fn=init_fn)

    def init_tensors(self, seed=None):
        if False:
            return 10
        'Returns a list of tensors with the given shape.'
        return self.parameters

    def init_variables(self, seed=None):
        if False:
            print('Hello World!')
        'Returns a list of variables with the given shape.'
        return self.parameters

    def objective(self, parameters, data=None, labels=None):
        if False:
            i = 10
            return i + 15
        'Computes the objective given a list of parameters.\n\n    Args:\n      parameters: The parameters to optimize (as a list of tensors)\n      data: An optional batch of data for calculating objectives\n      labels: An optional batch of corresponding labels\n\n    Returns:\n      A scalar tensor representing the objective value\n    '
        parameter_mapping = {old_p.name: p for (old_p, p) in zip(self.parameters, parameters)}
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            return _make_with_custom_variables(self.make_loss_fn, parameter_mapping)

def _get_variables(func):
    if False:
        while True:
            i = 10
    'Calls func, returning any variables created.\n\n  The created variables are modified to not be trainable, and are placed into\n  the LOCAL_VARIABLES collection.\n\n  Args:\n    func: Function to be called.\n\n  Returns:\n    A tuple (variables, constants) where the first element is a list of\n    trainable variables and the second is the non-trainable variables.\n  '
    variables = []
    constants = []
    original_init = tf.Variable.__init__

    def custom_init(self, *args, **kwargs):
        if False:
            print('Hello World!')
        trainable = kwargs['trainable']
        kwargs['trainable'] = False
        kwargs['collections'] = [tf.GraphKeys.LOCAL_VARIABLES]
        original_init(self, *args, **kwargs)
        if trainable:
            variables.append(self)
        else:
            constants.append(self)
    with tf.name_scope('unused_graph'):
        with mock.patch.object(tf.Variable, '__init__', custom_init):
            func()
    return (variables, constants)

def _make_with_custom_variables(func, variable_mapping):
    if False:
        print('Hello World!')
    'Calls func and replaces the value of some variables created in it.\n\n  Args:\n    func: Function to be called.\n    variable_mapping: A mapping of variable name to the replacement tensor or\n      tf.Variable.\n\n  Returns:\n    The return value of func is returned.\n  '
    original_value = tf.Variable.value

    def custom_value(self):
        if False:
            print('Hello World!')
        if self.name in variable_mapping:
            replacement = variable_mapping[self.name]
            tf.logging.info('Replaced %s with %s' % (self.name, replacement))
            if isinstance(replacement, tf.Variable):
                replacement = original_value(replacement)
            return replacement
        else:
            return original_value(self)
    with mock.patch.object(tf.Variable, 'value', custom_value):
        with mock.patch.object(tf.Variable, '_AsTensor', custom_value):
            return func()