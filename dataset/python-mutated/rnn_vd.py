"""Simple RNN model definitions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import tensorflow as tf
from regularization import variational_dropout
FLAGS = tf.app.flags.FLAGS

def discriminator(hparams, sequence, is_training, reuse=None, initial_state=None):
    if False:
        for i in range(10):
            print('nop')
    'Define the Discriminator graph.'
    tf.logging.info('Undirectional Discriminative model is not a useful model for this MaskGAN because future context is needed.  Use only for debugging purposes.')
    sequence = tf.cast(sequence, tf.int32)
    if FLAGS.dis_share_embedding:
        assert hparams.dis_rnn_size == hparams.gen_rnn_size, 'If you wish to share Discriminator/Generator embeddings, they must be same dimension.'
        with tf.variable_scope('gen/decoder/rnn', reuse=True):
            embedding = tf.get_variable('embedding', [FLAGS.vocab_size, hparams.gen_rnn_size])
    with tf.variable_scope('dis', reuse=reuse):

        def lstm_cell():
            if False:
                i = 10
                return i + 15
            return tf.contrib.rnn.BasicLSTMCell(hparams.dis_rnn_size, forget_bias=0.0, state_is_tuple=True, reuse=reuse)
        attn_cell = lstm_cell
        if is_training and hparams.dis_vd_keep_prob < 1:

            def attn_cell():
                if False:
                    print('Hello World!')
                return variational_dropout.VariationalDropoutWrapper(lstm_cell(), FLAGS.batch_size, hparams.dis_rnn_size, hparams.dis_vd_keep_prob, hparams.dis_vd_keep_prob)
        cell_dis = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(hparams.dis_num_layers)], state_is_tuple=True)
        if initial_state:
            state_dis = [[tf.identity(x) for x in inner_initial_state] for inner_initial_state in initial_state]
        else:
            state_dis = cell_dis.zero_state(FLAGS.batch_size, tf.float32)

        def make_mask(keep_prob, units):
            if False:
                print('Hello World!')
            random_tensor = keep_prob
            random_tensor += tf.random_uniform(tf.stack([FLAGS.batch_size, units]))
            return tf.floor(random_tensor) / keep_prob
        if is_training:
            output_mask = make_mask(hparams.dis_vd_keep_prob, hparams.dis_rnn_size)
        with tf.variable_scope('rnn') as vs:
            (predictions, rnn_outs) = ([], [])
            if not FLAGS.dis_share_embedding:
                embedding = tf.get_variable('embedding', [FLAGS.vocab_size, hparams.dis_rnn_size])
            rnn_inputs = tf.nn.embedding_lookup(embedding, sequence)
            for t in xrange(FLAGS.sequence_length):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                rnn_in = rnn_inputs[:, t]
                (rnn_out, state_dis) = cell_dis(rnn_in, state_dis)
                if is_training:
                    rnn_out *= output_mask
                pred = tf.contrib.layers.linear(rnn_out, 1, scope=vs)
                predictions.append(pred)
                rnn_outs.append(rnn_out)
    predictions = tf.stack(predictions, axis=1)
    if FLAGS.baseline_method == 'critic':
        with tf.variable_scope('critic', reuse=reuse) as critic_scope:
            rnn_outs = tf.stack(rnn_outs, axis=1)
            values = tf.contrib.layers.linear(rnn_outs, 1, scope=critic_scope)
        return (tf.squeeze(predictions, axis=2), tf.squeeze(values, axis=2))
    else:
        return (tf.squeeze(predictions, axis=2), None)