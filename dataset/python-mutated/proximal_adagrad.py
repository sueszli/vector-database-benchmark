"""ProximalAdagrad for TensorFlow."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['train.ProximalAdagradOptimizer'])
class ProximalAdagradOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Proximal Adagrad algorithm.

  References:
    Adaptive Subgradient Methods for Online Learning and Stochastic Optimization:
      [Duchi et al., 2011](http://jmlr.org/papers/v12/duchi11a.html)
      ([pdf](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf))
    Efficient Learning using Forward-Backward Splitting:
      [Duchi et al., 2009](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting)
      ([pdf](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf))
  """

    def __init__(self, learning_rate, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='ProximalAdagrad'):
        if False:
            while True:
                i = 10
        'Construct a new ProximalAdagrad optimizer.\n\n    Args:\n      learning_rate: A `Tensor` or a floating point value.  The learning rate.\n      initial_accumulator_value: A floating point value.\n        Starting value for the accumulators, must be positive.\n      l1_regularization_strength: A float value, must be greater than or\n        equal to zero.\n      l2_regularization_strength: A float value, must be greater than or\n        equal to zero.\n      use_locking: If `True` use locks for update operations.\n      name: Optional name prefix for the operations created when applying\n        gradients.  Defaults to "Adagrad".\n\n    Raises:\n      ValueError: If the `initial_accumulator_value` is invalid.\n    '
        if initial_accumulator_value <= 0.0:
            raise ValueError('initial_accumulator_value must be positive: %s' % initial_accumulator_value)
        super(ProximalAdagradOptimizer, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._initial_accumulator_value = initial_accumulator_value
        self._l1_regularization_strength = l1_regularization_strength
        self._l2_regularization_strength = l2_regularization_strength
        self._l1_regularization_strength_tensor = None
        self._l2_regularization_strength_tensor = None
        self._learning_rate_tensor = None

    def _create_slots(self, var_list):
        if False:
            print('Hello World!')
        for v in var_list:
            with ops.colocate_with(v):
                val = constant_op.constant(self._initial_accumulator_value, shape=v.get_shape(), dtype=v.dtype.base_dtype)
            self._get_or_make_slot(v, val, 'accumulator', self._name)

    def _prepare(self):
        if False:
            while True:
                i = 10
        self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate, name='learning_rate')
        self._l1_regularization_strength_tensor = ops.convert_to_tensor(self._l1_regularization_strength, name='l1_regularization_strength')
        self._l2_regularization_strength_tensor = ops.convert_to_tensor(self._l2_regularization_strength, name='l2_regularization_strength')

    def _apply_dense(self, grad, var):
        if False:
            while True:
                i = 10
        acc = self.get_slot(var, 'accumulator')
        return gen_training_ops.apply_proximal_adagrad(var, acc, self._learning_rate_tensor, self._l1_regularization_strength_tensor, self._l2_regularization_strength_tensor, grad, use_locking=self._use_locking)

    def _resource_apply_dense(self, grad, var):
        if False:
            for i in range(10):
                print('nop')
        acc = self.get_slot(var, 'accumulator')
        return gen_training_ops.resource_apply_proximal_adagrad(var.handle, acc.handle, self._learning_rate_tensor, self._l1_regularization_strength_tensor, self._l2_regularization_strength_tensor, grad, use_locking=self._use_locking)

    def _apply_sparse(self, grad, var):
        if False:
            print('Hello World!')
        acc = self.get_slot(var, 'accumulator')
        return gen_training_ops.sparse_apply_proximal_adagrad(var, acc, self._learning_rate_tensor, self._l1_regularization_strength_tensor, self._l2_regularization_strength_tensor, grad.values, grad.indices, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices):
        if False:
            for i in range(10):
                print('nop')
        acc = self.get_slot(var, 'accumulator')
        return gen_training_ops.resource_sparse_apply_proximal_adagrad(var.handle, acc.handle, math_ops.cast(self._learning_rate_tensor, grad.dtype), math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype), math_ops.cast(self._l2_regularization_strength_tensor, grad.dtype), grad, indices, use_locking=self._use_locking)