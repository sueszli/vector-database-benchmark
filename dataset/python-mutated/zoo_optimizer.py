from bigdl.dllib.optim.optimizer import OptimMethod
from bigdl.dllib.utils.tf import process_grad
from bigdl.dllib.utils.log4Error import invalidInputError

class FakeOptimMethod(OptimMethod):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(FakeOptimMethod, self).__init__(None, 'float')
import tensorflow as tf

def get_gradients_for_keras(optimizer, loss, params):
    if False:
        print('Hello World!')
    from tensorflow.python.util import nest
    from tensorflow.python.keras import backend
    from tensorflow.python.ops import gradients
    from tensorflow.python.ops import clip_ops
    from tensorflow.python.keras.optimizers import TFOptimizer
    params = nest.flatten(params)
    if isinstance(optimizer, TFOptimizer):
        scope_name = optimizer.optimizer._name
    else:
        scope_name = optimizer._name
    with backend.get_graph().as_default(), backend.name_scope(scope_name + '/gradients'):
        grads = gradients.gradients(loss, params)
        all_reduced_grads = []
        for (grad, param) in zip(grads, params):
            if grad is None:
                invalidInputError(False, 'Variable {} has `None` for gradient. Please make sure that all of your ops have a gradient defined (i.e. are differentiable). Common ops without gradient: K.argmax, K.round, K.eval.'.format(param))
            grad = process_grad(grad)
            with tf.control_dependencies([param]):
                grad_i = tf.identity(grad, name='zoo_identity_op_for_grad')
            all_reduced_grads.append(grad_i)
        grads = all_reduced_grads
        if hasattr(optimizer, 'clipnorm'):
            grads = [clip_ops.clip_by_norm(g, optimizer.clipnorm) for g in grads]
        if hasattr(optimizer, 'clipvalue'):
            grads = [clip_ops.clip_by_value(g, -optimizer.clipvalue, optimizer.clipvalue) for g in grads]
    return grads

class ZooOptimizer(tf.train.Optimizer):
    """An optimizer that wraps another tf.Optimizer, using an allreduce to
    combine gradient values before applying gradients to model weights."""

    def __init__(self, optimizer, name=None):
        if False:
            while True:
                i = 10
        if name is None:
            name = 'Zoo{}'.format(type(optimizer).__name__)
        super(ZooOptimizer, self).__init__(name=name, use_locking=False)
        self._optimizer = optimizer

    def compute_gradients(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Compute gradients of all trainable variables.\n        See Optimizer.compute_gradients() for more info.\n        In DistributedOptimizer, compute_gradients() is overriden to also\n        allreduce the gradients before returning them.\n        '
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        results = []
        for grad_var in gradients:
            grad = grad_var[0]
            var = grad_var[1]
            grad = process_grad(grad)
            if grad is not None:
                with tf.control_dependencies([var]):
                    grad_i = tf.identity(grad, name='zoo_identity_op_for_grad')
                results.append((grad_i, var))
            else:
                results.append((grad, var))
        return results

    def apply_gradients(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Calls this same method on the underlying optimizer.'
        return self._optimizer.apply_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        if False:
            return 10
        'Calls this same method on the underlying optimizer.'
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Calls this same method on the underlying optimizer.'
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        if False:
            return 10
        'Calls this same method on the underlying optimizer.'
        return self._optimizer.variables(*args, **kwargs)

    def _resource_apply_sparse(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._optimizer._resource_apply_sparse(*args, **kwargs)

    def _resource_apply_dense(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self._optimizer._resource_apply_sparse(*args, **kwargs)

    def _apply_sparse(self, *args, **kwargs):
        if False:
            return 10
        self._optimizer._apply_sparse(*args, **kwargs)

    def _apply_dense(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._optimizer._apply_dense(*args, **kwargs)