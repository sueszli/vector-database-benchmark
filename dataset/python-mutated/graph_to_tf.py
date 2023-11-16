import tensorflow as tf
from rnn import XGRUCell
from util import dropout
from graph import LayerType

def normalize(inputs, epsilon=1e-08, scope='ln'):
    if False:
        print('Hello World!')
    'Applies layer normalization.\n\n    Args:\n      inputs: A tensor with 2 or more dimensions, where the first dimension has\n        `batch_size`.\n      epsilon: A floating number. A very small number for preventing ZeroDivision Error.\n      scope: Optional scope for `variable_scope`.\n      reuse: Boolean, whether to reuse the weights of a previous layer\n        by the same name.\n\n    Returns:\n      A tensor with the same shape and data dtype as `inputs`.\n    '
    with tf.variable_scope(scope):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        (mean, variance) = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / (variance + epsilon) ** 0.5
        outputs = gamma * normalized + beta
    return outputs

def multihead_attention(queries, keys, scope='multihead_attention', num_units=None, num_heads=4, dropout_rate=0, is_training=True, causality=False):
    if False:
        return 10
    'Applies multihead attention.\n\n    Args:\n      queries: A 3d tensor with shape of [N, T_q, C_q].\n      keys: A 3d tensor with shape of [N, T_k, C_k].\n      num_units: A cdscalar. Attention size.\n      dropout_rate: A floating point number.\n      is_training: Boolean. Controller of mechanism for dropout.\n      causality: Boolean. If true, units that reference the future are masked.\n      num_heads: An int. Number of heads.\n      scope: Optional scope for `variable_scope`.\n      reuse: Boolean, whether to reuse the weights of a previous layer\n        by the same name.\n\n    Returns\n      A 3d tensor with shape of (N, T_q, C)\n    '
    global look5
    with tf.variable_scope(scope):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        Q_ = []
        K_ = []
        V_ = []
        for _ in range(num_heads):
            Q = tf.layers.dense(queries, num_units / num_heads, activation=tf.nn.relu)
            K = tf.layers.dense(keys, num_units / num_heads, activation=tf.nn.relu)
            V = tf.layers.dense(keys, num_units / num_heads, activation=tf.nn.relu)
            Q_.append(Q)
            K_.append(K)
            V_.append(V)
        Q_ = tf.concat(Q_, axis=0)
        K_ = tf.concat(K_, axis=0)
        V_ = tf.concat(V_, axis=0)
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        outputs = outputs / K_.get_shape().as_list()[-1] ** 0.5
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)
        look5 = outputs
        outputs = tf.nn.softmax(outputs)
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        outputs *= query_masks
        outputs = dropout(outputs, dropout_rate, is_training)
        outputs = tf.matmul(outputs, V_)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        if queries.get_shape().as_list()[-1] == num_units:
            outputs += queries
        outputs = normalize(outputs, scope=scope)
    return outputs

def positional_encoding(inputs, num_units=None, zero_pad=True, scale=True, scope='positional_encoding', reuse=None):
    if False:
        while True:
            i = 10
    '\n    Return positinal embedding.\n    '
    Shape = tf.shape(inputs)
    N = Shape[0]
    T = Shape[1]
    num_units = Shape[2]
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])
        X = tf.expand_dims(tf.cast(tf.range(T), tf.float32), axis=1)
        Y = tf.expand_dims(tf.cast(10000 ** (-(2 * tf.range(num_units) / num_units)), tf.float32), axis=0)
        h1 = tf.cast((tf.range(num_units) + 1) % 2, tf.float32)
        h2 = tf.cast(tf.range(num_units) % 2, tf.float32)
        position_enc = tf.multiply(X, Y)
        position_enc = tf.sin(position_enc) * tf.multiply(tf.ones_like(X), h1) + tf.cos(position_enc) * tf.multiply(tf.ones_like(X), h2)
        lookup_table = position_enc
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
        if scale:
            outputs = outputs * tf.sqrt(tf.cast(num_units, tf.float32))
        return outputs

def feedforward(inputs, num_units, scope='multihead_attention'):
    if False:
        return 10
    'Point-wise feed forward net.\n\n    Args:\n      inputs: A 3d tensor with shape of [N, T, C].\n      num_units: A list of two integers.\n      scope: Optional scope for `variable_scope`.\n      reuse: Boolean, whether to reuse the weights of a previous layer\n        by the same name.\n\n    Returns:\n      A 3d tensor with the same shape and dtype as inputs\n    '
    with tf.variable_scope(scope):
        params = {'inputs': inputs, 'filters': num_units[0], 'kernel_size': 1, 'activation': tf.nn.relu, 'use_bias': True}
        outputs = tf.layers.conv1d(**params)
        params = {'inputs': outputs, 'filters': num_units[1], 'kernel_size': 1, 'activation': None, 'use_bias': True}
        outputs = tf.layers.conv1d(**params)
        outputs += inputs
        outputs = normalize(outputs)
    return outputs

def rnn(input_states, sequence_lengths, dropout_rate, is_training, num_units):
    if False:
        return 10
    layer_cnt = 1
    states = []
    xs = tf.transpose(input_states, perm=[1, 0, 2])
    for i in range(0, layer_cnt):
        xs = dropout(xs, dropout_rate, is_training)
        with tf.variable_scope('layer_' + str(i)):
            cell_fw = XGRUCell(num_units)
            cell_bw = XGRUCell(num_units)
            (outputs, _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, dtype=tf.float32, sequence_length=sequence_lengths, inputs=xs, time_major=True)
        (y_lr, y_rl) = outputs
        xs = tf.concat([y_lr, y_rl], 2)
        states.append(xs)
    return tf.transpose(dropout(tf.concat(states, axis=2), dropout_rate, is_training), perm=[1, 0, 2])

def graph_to_network(input1, input2, input1_lengths, input2_lengths, graph, dropout_rate, is_training, num_heads=1, rnn_units=256):
    if False:
        for i in range(10):
            print('nop')
    topology = graph.is_topology()
    layers = dict()
    layers_sequence_lengths = dict()
    num_units = input1.get_shape().as_list()[-1]
    layers[0] = input1 * tf.sqrt(tf.cast(num_units, tf.float32)) + positional_encoding(input1, scale=False, zero_pad=False)
    layers[1] = input2 * tf.sqrt(tf.cast(num_units, tf.float32))
    layers[0] = dropout(layers[0], dropout_rate, is_training)
    layers[1] = dropout(layers[1], dropout_rate, is_training)
    layers_sequence_lengths[0] = input1_lengths
    layers_sequence_lengths[1] = input2_lengths
    for (_, topo_i) in enumerate(topology):
        if topo_i == '|':
            continue
        if graph.layers[topo_i].graph_type == LayerType.input.value:
            continue
        elif graph.layers[topo_i].graph_type == LayerType.attention.value:
            with tf.variable_scope('attation_%d' % topo_i):
                layer = multihead_attention(layers[graph.layers[topo_i].input[0]], layers[graph.layers[topo_i].input[1]], scope='multihead_attention%d' % topo_i, dropout_rate=dropout_rate, is_training=is_training, num_heads=num_heads, num_units=rnn_units * 2)
                layer = feedforward(layer, scope='feedforward%d' % topo_i, num_units=[rnn_units * 2 * 4, rnn_units * 2])
            layers[topo_i] = layer
            layers_sequence_lengths[topo_i] = layers_sequence_lengths[graph.layers[topo_i].input[0]]
        elif graph.layers[topo_i].graph_type == LayerType.self_attention.value:
            with tf.variable_scope('self-attation_%d' % topo_i):
                layer = multihead_attention(layers[graph.layers[topo_i].input[0]], layers[graph.layers[topo_i].input[0]], scope='multihead_attention%d' % topo_i, dropout_rate=dropout_rate, is_training=is_training, num_heads=num_heads, num_units=rnn_units * 2)
                layer = feedforward(layer, scope='feedforward%d' % topo_i, num_units=[rnn_units * 2 * 4, rnn_units * 2])
            layers[topo_i] = layer
            layers_sequence_lengths[topo_i] = layers_sequence_lengths[graph.layers[topo_i].input[0]]
        elif graph.layers[topo_i].graph_type == LayerType.rnn.value:
            with tf.variable_scope('rnn_%d' % topo_i):
                layer = rnn(layers[graph.layers[topo_i].input[0]], layers_sequence_lengths[graph.layers[topo_i].input[0]], dropout_rate, is_training, rnn_units)
            layers[topo_i] = layer
            layers_sequence_lengths[topo_i] = layers_sequence_lengths[graph.layers[topo_i].input[0]]
        elif graph.layers[topo_i].graph_type == LayerType.output.value:
            layers[topo_i] = layers[graph.layers[topo_i].input[0]]
            if layers[topo_i].get_shape().as_list()[-1] != rnn_units * 1 * 2:
                with tf.variable_scope('add_dense'):
                    layers[topo_i] = tf.layers.dense(layers[topo_i], units=rnn_units * 2)
    return (layers[2], layers[3])