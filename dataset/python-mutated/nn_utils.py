"""Author: aneelakantan (Arvind Neelakantan)
"""
import tensorflow as tf

def get_embedding(word, utility, params):
    if False:
        return 10
    return tf.nn.embedding_lookup(params['word'], word)

def apply_dropout(x, dropout_rate, mode):
    if False:
        i = 10
        return i + 15
    if dropout_rate > 0.0:
        if mode == 'train':
            x = tf.nn.dropout(x, dropout_rate)
        else:
            x = x
    return x

def LSTMCell(x, mprev, cprev, key, params):
    if False:
        for i in range(10):
            print('nop')
    'Create an LSTM cell.\n\n  Implements the equations in pg.2 from\n  "Long Short-Term Memory Based Recurrent Neural Network Architectures\n  For Large Vocabulary Speech Recognition",\n  Hasim Sak, Andrew Senior, Francoise Beaufays.\n\n  Args:\n    w: A dictionary of the weights and optional biases as returned\n      by LSTMParametersSplit().\n    x: Inputs to this cell.\n    mprev: m_{t-1}, the recurrent activations (same as the output)\n      from the previous cell.\n    cprev: c_{t-1}, the cell activations from the previous cell.\n    keep_prob: Keep probability on the input and the outputs of a cell.\n\n  Returns:\n    m: Outputs of this cell.\n    c: Cell Activations.\n    '
    i = tf.matmul(x, params[key + '_ix']) + tf.matmul(mprev, params[key + '_im'])
    i = tf.nn.bias_add(i, params[key + '_i'])
    f = tf.matmul(x, params[key + '_fx']) + tf.matmul(mprev, params[key + '_fm'])
    f = tf.nn.bias_add(f, params[key + '_f'])
    c = tf.matmul(x, params[key + '_cx']) + tf.matmul(mprev, params[key + '_cm'])
    c = tf.nn.bias_add(c, params[key + '_c'])
    o = tf.matmul(x, params[key + '_ox']) + tf.matmul(mprev, params[key + '_om'])
    o = tf.nn.bias_add(o, params[key + '_o'])
    i = tf.sigmoid(i, name='i_gate')
    f = tf.sigmoid(f, name='f_gate')
    o = tf.sigmoid(o, name='o_gate')
    c = f * cprev + i * tf.tanh(c)
    m = o * c
    return (m, c)