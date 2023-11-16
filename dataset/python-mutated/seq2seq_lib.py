"""seq2seq library codes copied from elsewhere for customization."""
import tensorflow as tf

def sequence_loss_by_example(inputs, targets, weights, loss_function, average_across_timesteps=True, name=None):
    if False:
        i = 10
        return i + 15
    "Sampled softmax loss for a sequence of inputs (per example).\n\n  Args:\n    inputs: List of 2D Tensors of shape [batch_size x hid_dim].\n    targets: List of 1D batch-sized int32 Tensors of the same length as logits.\n    weights: List of 1D batch-sized float-Tensors of the same length as logits.\n    loss_function: Sampled softmax function (inputs, labels) -> loss\n    average_across_timesteps: If set, divide the returned cost by the total\n      label weight.\n    name: Optional name for this operation, default: 'sequence_loss_by_example'.\n\n  Returns:\n    1D batch-sized float Tensor: The log-perplexity for each sequence.\n\n  Raises:\n    ValueError: If len(inputs) is different from len(targets) or len(weights).\n  "
    if len(targets) != len(inputs) or len(weights) != len(inputs):
        raise ValueError('Lengths of logits, weights, and targets must be the same %d, %d, %d.' % (len(inputs), len(weights), len(targets)))
    with tf.name_scope(values=inputs + targets + weights, name=name, default_name='sequence_loss_by_example'):
        log_perp_list = []
        for (inp, target, weight) in zip(inputs, targets, weights):
            crossent = loss_function(inp, target)
            log_perp_list.append(crossent * weight)
        log_perps = tf.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = tf.add_n(weights)
            total_size += 1e-12
            log_perps /= total_size
    return log_perps

def sampled_sequence_loss(inputs, targets, weights, loss_function, average_across_timesteps=True, average_across_batch=True, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Weighted cross-entropy loss for a sequence of logits, batch-collapsed.\n\n  Args:\n    inputs: List of 2D Tensors of shape [batch_size x hid_dim].\n    targets: List of 1D batch-sized int32 Tensors of the same length as inputs.\n    weights: List of 1D batch-sized float-Tensors of the same length as inputs.\n    loss_function: Sampled softmax function (inputs, labels) -> loss\n    average_across_timesteps: If set, divide the returned cost by the total\n      label weight.\n    average_across_batch: If set, divide the returned cost by the batch size.\n    name: Optional name for this operation, defaults to 'sequence_loss'.\n\n  Returns:\n    A scalar float Tensor: The average log-perplexity per symbol (weighted).\n\n  Raises:\n    ValueError: If len(inputs) is different from len(targets) or len(weights).\n  "
    with tf.name_scope(values=inputs + targets + weights, name=name, default_name='sampled_sequence_loss'):
        cost = tf.reduce_sum(sequence_loss_by_example(inputs, targets, weights, loss_function, average_across_timesteps=average_across_timesteps))
        if average_across_batch:
            batch_size = tf.shape(targets[0])[0]
            return cost / tf.cast(batch_size, tf.float32)
        else:
            return cost

def linear(args, output_size, bias, bias_start=0.0, scope=None):
    if False:
        return 10
    'Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.\n\n  Args:\n    args: a 2D Tensor or a list of 2D, batch x n, Tensors.\n    output_size: int, second dimension of W[i].\n    bias: boolean, whether to add a bias term or not.\n    bias_start: starting value to initialize the bias; 0 by default.\n    scope: VariableScope for the created subgraph; defaults to "Linear".\n\n  Returns:\n    A 2D Tensor with shape [batch x output_size] equal to\n    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.\n\n  Raises:\n    ValueError: if some of the arguments has unspecified or wrong shape.\n  '
    if args is None or (isinstance(args, (list, tuple)) and (not args)):
        raise ValueError('`args` must be specified')
    if not isinstance(args, (list, tuple)):
        args = [args]
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError('Linear is expecting 2D arguments: %s' % str(shapes))
        if not shape[1]:
            raise ValueError('Linear expects shape[1] of arguments: %s' % str(shapes))
        else:
            total_arg_size += shape[1]
    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable('Matrix', [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable('Bias', [output_size], initializer=tf.constant_initializer(bias_start))
    return res + bias_term