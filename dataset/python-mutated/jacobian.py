"""Helper functions for computing jacobian."""
import tensorflow.compat.v2 as tf

def jacobian(func, x, unconnected_gradients=None, parallel_iterations=None, experimental_use_pfor=True, name=None):
    if False:
        while True:
            i = 10
    "Computes the jacobian of `func` wrt to `x`.\n\n  Args:\n    func: Python callable accepting one `Tensor` of shape of `x` and returning\n      a `Tensor` of any shape. The function whose jacobian is to be computed.\n    x: A `Tensor` with respect to which the gradient is to be computed.\n    unconnected_gradients: An enum `tf.UnconnectedGradients` which specifies\n      the gradient value returned when the given input tensors are\n      unconnected. Default value: `None`, which maps to\n      `tf.UnconnectedGradients.NONE`.\n    parallel_iterations: A knob to control how many iterations are dispatched\n      in parallel. This knob can be used to control the total memory usage.\n    experimental_use_pfor: If true, uses pfor for computing the Jacobian.\n      Else uses a tf.while_loop.\n    name: Python `str` name prefixed to ops created by this function.\n      Default value: `None` (i.e., 'jacobian').\n\n  Returns:\n    A `Tensor` with the gradient of `y` wrt each of `x`.\n  "
    unconnected_gradients = unconnected_gradients or tf.UnconnectedGradients.NONE
    (x, is_x_batch_size) = _prepare_args(x)
    with tf.name_scope(name or 'jacobian'):
        if not callable(func):
            raise ValueError('`func` should be a callable in eager mode or when `tf.GradientTape` is used.')
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = func(x)
        jac = tape.batch_jacobian(y, x, unconnected_gradients=unconnected_gradients, parallel_iterations=parallel_iterations, experimental_use_pfor=experimental_use_pfor)
        if is_x_batch_size:
            return jac
        return jac[0]

def value_and_jacobian(f, x, unconnected_gradients=None, name=None, parallel_iterations=None, experimental_use_pfor=True):
    if False:
        return 10
    "Computes `f(x)` and its jacobian wrt to `x`.\n\n  Args:\n    f: Python `callable` to be differentiated. If `f` returns a scalar, this\n      scalar will be differentiated. If `f` returns a tensor or list of\n      tensors, by default a scalar will be computed by adding all their values\n      to produce a single scalar. If desired, the tensors can be elementwise\n      multiplied by the tensors passed as the `dy` keyword argument to the\n      returned jacobian function.\n    x: A `Tensor` with respect to which the gradient is to be computed.\n    unconnected_gradients: An enum `tf.UnconnectedGradients` which specifies\n      the gradient value returned when the given input tensors are\n      unconnected. Default value: `None`, which maps to\n      `tf.UnconnectedGradients.NONE`.\n    name: Python `str` name prefixed to ops created by this function.\n      Default value: `None` (i.e., `'value_and_jacobian'`).\n    parallel_iterations: A knob to control how many iterations are dispatched\n      in parallel. This knob can be used to control the total memory usage.\n    experimental_use_pfor: If true, uses pfor for computing the Jacobian.\n      Else uses a tf.while_loop.\n\n  Returns:\n    A tuple of two elements. The first one is a `Tensor` representing the value\n    of the function at `x` and the second one is a `Tensor` representing\n    jacobian of `f(x)` wrt `x`.\n    y: `y = f(x)`.\n    dydx: Jacobian of `y` wrt `x_i`, where `x_i` is the i-th parameter in\n    `x`.\n  "
    unconnected_gradients = unconnected_gradients or tf.UnconnectedGradients.NONE
    (x, is_x_batch_size) = _prepare_args(x)
    with tf.name_scope(name or 'value_and_jacobian'):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = f(x)
        jac = tape.batch_jacobian(y, x, unconnected_gradients=unconnected_gradients, parallel_iterations=parallel_iterations, experimental_use_pfor=experimental_use_pfor)
        if is_x_batch_size:
            return (y, jac)
        return (y[0], jac[0])

def _prepare_args(x):
    if False:
        while True:
            i = 10
    'Converts `x` to a batched dimension if necessary.'
    if len(x.shape) == 1:
        return (tf.expand_dims(x, axis=0), False)
    return (x, True)