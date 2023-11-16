"""Custom implementations of loops for improved performance."""
import tensorflow.compat.v2 as tf
from tf_quant_finance import utils as tff_utils

def for_loop(body_fn, initial_state, params, num_iterations, name=None):
    if False:
        i = 10
        return i + 15
    'A for loop with a custom batched gradient.\n\n  A for loop with a custom gradient that in certain cases outperforms the\n  tf.while_loop gradient implementation.\n\n  This is not a general replacement for tf.while_loop as imposes a number of\n  restrictions on the inputs:\n  - All tensors in loop state must have the same shape except for the last\n  dimension.\n  - All dimensions except for the last one are treated as batch dimensions, i.e.\n  it is assumed that batch elements don\'t interact with each other inside the\n  loop body.\n  - The last dimensions and the number of parameters must be statically known\n  and be reasonably small (so that the full Jacobian matrix with respect to them\n  can be calculated efficiently, see below).\n  - It requires an explicit list of parameters used in the loop body, with\n  respect to which one wishes to calculate derivatives. This is different from\n  tf.while_loop which automatically deals with tensors captured in the closure.\n  - Parameters must be a sequence of zero-dimensional tensors.\n  - Arbitrary nested structure of state is not supported, the state must be a\n  flat sequence of tensors.\n\n  The issue this implementation addresses is the additional while loops created\n  by the gradient of `tf.while_loop`. To compute the backward gradient\n  (more precisely, the vector-Jacobian product) of a while loop, one needs to\n  run the loop "backwards". This implementation avoids creating a second loop by\n  calculating the full (batched) Jacobian matrix in the forward pass. It is\n  efficient when the non-batched part of the shape of the Jacobian is small.\n  This part has size `nd * (nd + p)` where `nd` is the sum of last dimensions of\n  tensors in the state and `p` is the number of parameters.\n\n  This implementation is suitable for e.g. Monte-Carlo sampling, where the state\n  represents a batch of independent paths.\n\n  #### Example:\n\n  ```python\n  x = tf.constant([[3.0, 4.0], [30.0, 40.0]])\n  y = tf.constant([[7.0, 8.0], [70.0, 80.0]])\n  alpha = tf.constant(2.0)\n  beta = tf.constant(1.0)\n\n  with tf.GradientTape(persistent=True) as tape:\n    tape.watch([alpha, beta])\n    def body(i, state):\n      x, y = state\n      return [x * alpha - beta, y * beta + x]\n    x_out, y_out = for_loop(body, [x, y], [alpha, beta], 3)\n\n  grad = tape.gradient(y_out, beta)  # Returns tf.Tensor(783.0)\n  ```\n\n  Args:\n    body_fn: A Callable. Accepts an iteration index as a 0-dimensional int32\n      tensor and state - a tuple of Tensors of same shape as `initial_state`.\n      Should return the output state with the same structure as the input state.\n    initial_state: A sequence of Tensors with common batch shape. All dimensions\n      except the last are treated as batch shape (i.e. not mixed in loop body).\n    params: A list of zero-dimensional Tensors - tensors that `body_fn` uses,\n      and with respect to which the differentiation is going to happen.\n    num_iterations: A rank 0 or rank 1 integer tensor. If the rank is 1, the\n      entries are expected to be unique and ordered and  the output will contain\n      results obtained at each iteration number specified in `num_iterations`,\n      stacked along the first dimension. E.g. if `initial_state` has shapes\n      `(10, 20, 2)` and `(10, 20, 3)`, and `num_iterations = [2, 5, 7, 10]` the\n      output is a list of tensors with shapes `(4, 10, 20, 2)` and\n      `(4, 10, 20, 3)`.\n\n    name: Python str. The name to give to the ops created by this function,\n      \'for_loop\' by default.\n\n  Returns:\n   A list of Tensors of the same shape as `initial_state`, if `num_iterations`\n   is a single integer, or with extra first dimension of size\n   `len(num_iterations)` otherwise.\n   The outputs are differentiable with respect to `initial_state` and `params`,\n   but not any other tensors that are captured by `body_fn`. Differentiating\n   with respect to an element of `initial_state` yields a tensor with the same\n   shape as that element. Differentiating with respect to one of `params` yields\n   a tensor of zero shape. If the output state doesn\'t depend on the given\n   parameter, the tensor will be filled with zeros.\n  '
    num_iterations = tf.convert_to_tensor(num_iterations, dtype=tf.int32, name='num_iterations')
    num_iterations_shape = num_iterations.shape.as_list()
    if num_iterations_shape is None:
        raise ValueError('Rank of num_iterations must be statically known.')
    if len(num_iterations_shape) > 1:
        raise ValueError('Rank of num_iterations must be 0 or 1')
    if len(num_iterations_shape) == 1:
        return _accumulating_for_loop(body_fn, initial_state, params, num_iterations, name)
    with tf.name_scope(name or 'for_loop'):
        initial_jac = _make_unit_jacobian(initial_state, params)
        n = len(initial_state)

        @tf.custom_gradient
        def inner(*args):
            if False:
                return 10
            (initial_state, params) = (args[:n], args[n:])

            def while_cond(i, state, jac):
                if False:
                    print('Hello World!')
                del state, jac
                return i < num_iterations

            def while_body(i, state, jac):
                if False:
                    for i in range(10):
                        print('nop')
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(state)
                    tape.watch(params)
                    next_state = tuple(body_fn(i, state))
                step_jac = _compute_step_jacobian(state, next_state, params, tape)
                next_jac = _multiply_jacobians(step_jac, jac)
                return (i + 1, next_state, next_jac)
            loop_vars = (0, initial_state, initial_jac)
            (_, state, jac) = tf.compat.v2.while_loop(while_cond, while_body, loop_vars=loop_vars, maximum_iterations=num_iterations)

            def gradient(*ws):
                if False:
                    i = 10
                    return i + 15
                ws = [tf.expand_dims(w, axis=-2) for w in ws]
                ws = [ws]
                (js, jp) = jac
                (ws_js, ws_jp) = (_block_matmul(ws, js), _block_matmul(ws, jp))
                (ws_js, ws_jp) = (ws_js[0], ws_jp[0])
                ws_js = [tf.squeeze(t, axis=-2) for t in ws_js]
                ws_jp = [tf.reduce_sum(t) for t in ws_jp]
                return ws_js + ws_jp
            return (state, gradient)
        args = tuple(initial_state + params)
        return inner(*args)

def _make_unit_jacobian(initial_state, params):
    if False:
        i = 10
        return i + 15
    'Creates a unit Jacobian matrix.'
    n = len(initial_state)
    d = [initial_state[i].shape.as_list()[-1] for i in range(n)]
    if None in d:
        raise ValueError('Last dimensions of initial_state Tensors must be known.')
    p = len(params)
    dtype = initial_state[0].dtype

    def make_js_block(i, j):
        if False:
            return 10
        shape = initial_state[i].shape.concatenate((d[j],))
        if i != j:
            return tf.zeros(shape, dtype=dtype)
        eye = tf.eye(d[i], dtype=dtype)
        return tf.broadcast_to(eye, shape)

    def make_jp_block(i, j):
        if False:
            i = 10
            return i + 15
        del j
        shape = initial_state[i].shape.concatenate((1,))
        return tf.zeros(shape, dtype=dtype)
    js = [[make_js_block(i, j) for j in range(n)] for i in range(n)]
    jp = [[make_jp_block(i, j) for j in range(p)] for i in range(n)]
    return (js, jp)

def _compute_step_jacobian(state, next_state, params, tape):
    if False:
        i = 10
        return i + 15
    'Computes a Jacobian of a transformation next_state = f(state, params).'
    n = len(state)
    p = len(params)
    js = [[_batch_jacobian(next_state[i], state[j], tape) for j in range(n)] for i in range(n)]
    jp = [[_jacobian_wrt_parameter(next_state[i], params[j], tape) for j in range(p)] for i in range(n)]
    return (js, jp)

def _batch_jacobian(y, x, tape):
    if False:
        print('Hello World!')
    'Computes a Jacobian w.r.t. last dimensions of y and x.'
    d = y.shape.as_list()[-1]
    if d is None:
        raise ValueError('Last dimension of state Tensors must be known.')
    grads = []
    for i in range(d):
        w = tf.broadcast_to(tf.one_hot(i, d, dtype=y.dtype), y.shape)
        grad = tape.gradient(y, x, output_gradients=w, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        grads.append(grad)
    return tf.stack(grads, axis=-2)

def _jacobian_wrt_parameter(y, param, tape):
    if False:
        print('Hello World!')
    'Computes a Jacobian w.r.t. a parameter.'
    with tf.GradientTape() as w_tape:
        w = tf.zeros_like(y)
        w_tape.watch(w)
        vjp = tape.gradient(y, param, output_gradients=w)
    if vjp is None:
        return tf.expand_dims(tf.zeros_like(y), axis=-1)
    return tf.expand_dims(w_tape.gradient(vjp, w), axis=-1)

def _multiply_jacobians(jac1, jac2):
    if False:
        print('Hello World!')
    'Multiplies two Jacobians.'
    (js1, jp1) = jac1
    (js2, jp2) = jac2
    return (_block_matmul(js1, js2), _block_add(_block_matmul(js1, jp2), jp1))

def _block_matmul(m1, m2):
    if False:
        print('Hello World!')
    'Multiplies block matrices represented as nested lists.'
    if isinstance(m1, tf.Tensor):
        assert isinstance(m2, tf.Tensor)
        return tf.matmul(m1, m2)
    assert _is_nested_list(m1) and _is_nested_list(m2)
    i_max = len(m1)
    k_max = len(m2)
    j_max = 0 if k_max == 0 else len(m2[0])
    if i_max > 0:
        assert len(m1[0]) == k_max

    def row_by_column(i, j):
        if False:
            print('Hello World!')
        return _block_add(*[_block_matmul(m1[i][k], m2[k][j]) for k in range(k_max)])
    return [[row_by_column(i, j) for j in range(j_max)] for i in range(i_max)]

def _block_add(*ms):
    if False:
        print('Hello World!')
    'Adds block matrices represented as nested lists.'
    if len(ms) == 1:
        return ms[0]
    if isinstance(ms[0], tf.Tensor):
        assert all((isinstance(m, tf.Tensor) for m in ms[1:]))
        return tf.math.add_n(ms)
    assert all((_is_nested_list(m) for m in ms))
    for i in range(1, len(ms)):
        tf.nest.assert_same_structure(ms[0], ms[i])
    i_max = len(ms[0])
    j_max = 0 if i_max == 0 else len(ms[0][0])
    return [[_block_add(*[ms[k][i][j] for k in range(len(ms))]) for j in range(j_max)] for i in range(i_max)]

def _is_nested_list(m):
    if False:
        return 10
    return isinstance(m, list) and (not m or isinstance(m[0], list))

def _accumulating_for_loop(body_fn, initial_state, params, num_iterations, name=None):
    if False:
        while True:
            i = 10
    'Version of for_loop with multiple values of num_iterations.'
    with tf.name_scope(name or 'accumulating_for_loop'):
        max_iterations = tf.math.reduce_max(num_iterations)
        acc_size = tff_utils.get_shape(num_iterations)[0]
        mask = tf.scatter_nd(indices=tf.expand_dims(num_iterations, axis=-1), updates=tf.ones_like(num_iterations), shape=(max_iterations + 1,))
        n = len(initial_state)

        @tf.custom_gradient
        def inner(*args):
            if False:
                for i in range(10):
                    print('nop')
            (initial_state, params) = (args[:n], args[n:])

            def while_cond(i, acc_index, state, jac, acc_state, acc_jac):
                if False:
                    print('Hello World!')
                del acc_index, state, jac, acc_state, acc_jac
                return i < max_iterations

            def while_body(i, acc_index, state, jac, acc_state, acc_jac):
                if False:
                    i = 10
                    return i + 15
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(state)
                    tape.watch(params)
                    next_state = tuple(body_fn(i, state))
                step_jac = _compute_step_jacobian(state, next_state, params, tape)
                next_jac = _multiply_jacobians(step_jac, jac)
                acc_index += mask[i]
                acc_state = _write_to_accumulators(acc_state, next_state, acc_index)
                acc_jac = _write_to_accumulators(acc_jac, next_jac, acc_index)
                return (i + 1, acc_index, next_state, next_jac, acc_state, acc_jac)
            initial_acc_state = _create_accumulators(initial_state, acc_size)
            initial_acc_state = _write_to_accumulators(initial_acc_state, initial_state, 0)
            initial_jac = _make_unit_jacobian(initial_state, params)
            initial_acc_jac = _create_accumulators(initial_jac, acc_size)
            initial_acc_jac = _write_to_accumulators(initial_acc_jac, initial_jac, 0)
            loop_vars = (0, 0, initial_state, initial_jac, initial_acc_state, initial_acc_jac)
            (_, _, _, _, final_acc_state, final_acc_jac) = tf.compat.v2.while_loop(while_cond, while_body, loop_vars=loop_vars, maximum_iterations=max_iterations)
            final_acc_state = _stack_accumulators(final_acc_state)
            final_acc_jac = _stack_accumulators(final_acc_jac)

            def gradient(*ws):
                if False:
                    print('Hello World!')
                ws = [tf.expand_dims(w, axis=-2) for w in ws]
                ws = [ws]
                (js, jp) = final_acc_jac
                (ws_js, ws_jp) = (_block_matmul(ws, js), _block_matmul(ws, jp))
                (ws_js, ws_jp) = (ws_js[0], ws_jp[0])
                ws_js = [tf.squeeze(t, axis=-2) for t in ws_js]
                ws_jp = [tf.squeeze(t, axis=[-2, -1]) for t in ws_jp]
                ws_js = [tf.math.reduce_sum(t, axis=0) for t in ws_js]
                ws_jp = [tf.math.reduce_sum(t) for t in ws_jp]
                return ws_js + ws_jp
            return (final_acc_state, gradient)
        args = tuple(initial_state + params)
        return inner(*args)

def _create_accumulators(nested_tensor, size):
    if False:
        return 10
    if isinstance(nested_tensor, tf.Tensor):
        return tf.TensorArray(dtype=nested_tensor.dtype, size=size, element_shape=tff_utils.get_shape(nested_tensor), clear_after_read=False)
    return [_create_accumulators(t, size) for t in nested_tensor]

def _write_to_accumulators(nested_acc, nested_tensor, index):
    if False:
        while True:
            i = 10
    if isinstance(nested_tensor, tf.Tensor):
        return nested_acc.write(index, nested_tensor)
    return [_write_to_accumulators(acc, t, index) for (acc, t) in zip(nested_acc, nested_tensor)]

def _stack_accumulators(nested_acc):
    if False:
        i = 10
        return i + 15
    if isinstance(nested_acc, tf.TensorArray):
        return nested_acc.stack()
    return [_stack_accumulators(acc) for acc in nested_acc]
__all__ = []