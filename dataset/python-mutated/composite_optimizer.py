"""An optimizer that switches between several methods."""
import functools
import tensorflow as tf
from tensorflow.python.training import optimizer

class CompositeOptimizer(optimizer.Optimizer):
    """Optimizer that switches between several methods.
  """

    def __init__(self, optimizer1, optimizer2, switch, use_locking=False, name='Composite'):
        if False:
            for i in range(10):
                print('nop')
        'Construct a new Composite optimizer.\n\n    Args:\n      optimizer1: A tf.python.training.optimizer.Optimizer object.\n      optimizer2: A tf.python.training.optimizer.Optimizer object.\n      switch: A tf.bool Tensor, selecting whether to use the first or the second\n        optimizer.\n      use_locking: Bool. If True apply use locks to prevent concurrent updates\n        to variables.\n      name: Optional name prefix for the operations created when applying\n        gradients.  Defaults to "Composite".\n    '
        super(CompositeOptimizer, self).__init__(use_locking, name)
        self._optimizer1 = optimizer1
        self._optimizer2 = optimizer2
        self._switch = switch

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        if False:
            return 10
        return tf.cond(self._switch, functools.partial(self._optimizer1.apply_gradients, grads_and_vars, global_step, name), functools.partial(self._optimizer2.apply_gradients, grads_and_vars, global_step, name))

    def get_slot(self, var, name):
        if False:
            i = 10
            return i + 15
        if name.startswith('c1-'):
            return self._optimizer1.get_slot(var, name[3:])
        else:
            return self._optimizer2.get_slot(var, name[3:])

    def get_slot_names(self):
        if False:
            i = 10
            return i + 15
        opt1_names = self._optimizer1.get_slot_names()
        opt2_names = self._optimizer2.get_slot_names()
        return sorted(['c1-{}'.format(name) for name in opt1_names] + ['c2-{}'.format(name) for name in opt2_names])