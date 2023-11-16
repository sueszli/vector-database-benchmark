"""ProximalGradientDescent for TensorFlow."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['train.ProximalGradientDescentOptimizer'])
class ProximalGradientDescentOptimizer(optimizer.Optimizer):
    """Optimizer that implements the proximal gradient descent algorithm.

  References:
    Efficient Learning using Forward-Backward Splitting:
      [Duchi et al., 2009](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting)
      ([pdf](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf))
  """

    def __init__(self, learning_rate, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='ProximalGradientDescent'):
        if False:
            for i in range(10):
                print('nop')
        'Construct a new proximal gradient descent optimizer.\n\n    Args:\n      learning_rate: A Tensor or a floating point value.  The learning\n        rate to use.\n      l1_regularization_strength: A float value, must be greater than or\n        equal to zero.\n      l2_regularization_strength: A float value, must be greater than or\n        equal to zero.\n      use_locking: If True use locks for update operations.\n      name: Optional name prefix for the operations created when applying\n        gradients. Defaults to "GradientDescent".\n    '
        super(ProximalGradientDescentOptimizer, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._l1_regularization_strength = l1_regularization_strength
        self._l2_regularization_strength = l2_regularization_strength
        self._l1_regularization_strength_tensor = None
        self._l2_regularization_strength_tensor = None

    def _apply_dense(self, grad, var):
        if False:
            print('Hello World!')
        return gen_training_ops.apply_proximal_gradient_descent(var, self._learning_rate_tensor, self._l1_regularization_strength_tensor, self._l2_regularization_strength_tensor, grad, use_locking=self._use_locking).op

    def _resource_apply_dense(self, grad, var):
        if False:
            while True:
                i = 10
        return gen_training_ops.resource_apply_proximal_gradient_descent(var.handle, self._learning_rate_tensor, self._l1_regularization_strength_tensor, self._l2_regularization_strength_tensor, grad, use_locking=self._use_locking)

    def _apply_sparse(self, grad, var):
        if False:
            for i in range(10):
                print('nop')
        return gen_training_ops.sparse_apply_proximal_gradient_descent(var, self._learning_rate_tensor, self._l1_regularization_strength_tensor, self._l2_regularization_strength_tensor, grad.values, grad.indices, use_locking=self._use_locking).op

    def _resource_apply_sparse(self, grad, var, indices):
        if False:
            i = 10
            return i + 15
        return gen_training_ops.resource_sparse_apply_proximal_gradient_descent(var.handle, math_ops.cast(self._learning_rate_tensor, grad.dtype), math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype), math_ops.cast(self._l2_regularization_strength_tensor, grad.dtype), grad, indices, use_locking=self._use_locking)

    def _prepare(self):
        if False:
            for i in range(10):
                print('nop')
        self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate, name='learning_rate')
        self._l1_regularization_strength_tensor = ops.convert_to_tensor(self._l1_regularization_strength, name='l1_regularization_strength')
        self._l2_regularization_strength_tensor = ops.convert_to_tensor(self._l2_regularization_strength, name='l2_regularization_strength')