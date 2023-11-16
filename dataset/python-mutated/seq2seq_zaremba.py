"""Simple seq2seq model definitions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from six.moves import xrange
from models import attention_utils
FLAGS = tf.app.flags.FLAGS

def transform_input_with_is_missing_token(inputs, targets_present):
    if False:
        i = 10
        return i + 15
    "Transforms the inputs to have missing tokens when it's masked out.  The\n  mask is for the targets, so therefore, to determine if an input at time t is\n  masked, we have to check if the target at time t - 1 is masked out.\n\n  e.g.\n    inputs = [a, b, c, d]\n    targets = [b, c, d, e]\n    targets_present = [1, 0, 1, 0]\n\n  then,\n    transformed_input = [a, b, <missing>, d]\n\n  Args:\n    inputs:  tf.int32 Tensor of shape [batch_size, sequence_length] with tokens\n      up to, but not including, vocab_size.\n    targets_present:  tf.bool Tensor of shape [batch_size, sequence_length] with\n      True representing the presence of the word.\n\n  Returns:\n    transformed_input:  tf.int32 Tensor of shape [batch_size, sequence_length]\n      which takes on value of inputs when the input is present and takes on\n      value=vocab_size to indicate a missing token.\n  "
    input_missing = tf.constant(FLAGS.vocab_size, dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.sequence_length])
    zeroth_input_present = tf.constant(True, tf.bool, shape=[FLAGS.batch_size, 1])
    inputs_present = tf.concat([zeroth_input_present, targets_present[:, :-1]], axis=1)
    transformed_input = tf.where(inputs_present, inputs, input_missing)
    return transformed_input

def gen_encoder(hparams, inputs, targets_present, is_training, reuse=None):
    if False:
        return 10
    'Define the Encoder graph.\n\n\n  Args:\n    hparams:  Hyperparameters for the MaskGAN.\n    inputs:  tf.int32 Tensor of shape [batch_size, sequence_length] with tokens\n      up to, but not including, vocab_size.\n    targets_present:  tf.bool Tensor of shape [batch_size, sequence_length] with\n      True representing the presence of the target.\n    is_training:  Boolean indicating operational mode (train/inference).\n    reuse (Optional):   Whether to reuse the variables.\n\n  Returns:\n    Tuple of (hidden_states, final_state).\n  '
    with tf.variable_scope('encoder', reuse=reuse):

        def lstm_cell():
            if False:
                print('Hello World!')
            return tf.contrib.rnn.BasicLSTMCell(hparams.gen_rnn_size, forget_bias=0.0, state_is_tuple=True, reuse=reuse)
        attn_cell = lstm_cell
        if is_training and FLAGS.keep_prob < 1:

            def attn_cell():
                if False:
                    while True:
                        i = 10
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=FLAGS.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(hparams.gen_num_layers)], state_is_tuple=True)
        initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)
        real_inputs = inputs
        masked_inputs = transform_input_with_is_missing_token(inputs, targets_present)
        with tf.variable_scope('rnn'):
            hidden_states = []
            embedding = tf.get_variable('embedding', [FLAGS.vocab_size, hparams.gen_rnn_size])
            missing_embedding = tf.get_variable('missing_embedding', [1, hparams.gen_rnn_size])
            embedding = tf.concat([embedding, missing_embedding], axis=0)
            real_rnn_inputs = tf.nn.embedding_lookup(embedding, real_inputs)
            masked_rnn_inputs = tf.nn.embedding_lookup(embedding, masked_inputs)
            if is_training and FLAGS.keep_prob < 1:
                masked_rnn_inputs = tf.nn.dropout(masked_rnn_inputs, FLAGS.keep_prob)
            state = initial_state
            for t in xrange(FLAGS.sequence_length):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                rnn_inp = masked_rnn_inputs[:, t]
                (rnn_out, state) = cell(rnn_inp, state)
                hidden_states.append(rnn_out)
            final_masked_state = state
            hidden_states = tf.stack(hidden_states, axis=1)
            real_state = initial_state
            for t in xrange(FLAGS.sequence_length):
                tf.get_variable_scope().reuse_variables()
                rnn_inp = real_rnn_inputs[:, t]
                (rnn_out, real_state) = cell(rnn_inp, real_state)
            final_state = real_state
    return ((hidden_states, final_masked_state), initial_state, final_state)

def gen_decoder(hparams, inputs, targets, targets_present, encoding_state, is_training, is_validating, reuse=None):
    if False:
        while True:
            i = 10
    'Define the Decoder graph. The Decoder will now impute tokens that\n      have been masked from the input seqeunce.\n  '
    gen_decoder_rnn_size = hparams.gen_rnn_size
    with tf.variable_scope('decoder', reuse=reuse):

        def lstm_cell():
            if False:
                return 10
            return tf.contrib.rnn.BasicLSTMCell(gen_decoder_rnn_size, forget_bias=0.0, state_is_tuple=True, reuse=reuse)
        attn_cell = lstm_cell
        if is_training and FLAGS.keep_prob < 1:

            def attn_cell():
                if False:
                    i = 10
                    return i + 15
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=FLAGS.keep_prob)
        cell_gen = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(hparams.gen_num_layers)], state_is_tuple=True)
        hidden_vector_encodings = encoding_state[0]
        state_gen = encoding_state[1]
        if FLAGS.attention_option is not None:
            (attention_keys, attention_values, _, attention_construct_fn) = attention_utils.prepare_attention(hidden_vector_encodings, FLAGS.attention_option, num_units=gen_decoder_rnn_size, reuse=reuse)
        with tf.variable_scope('rnn'):
            (sequence, logits, log_probs) = ([], [], [])
            embedding = tf.get_variable('embedding', [FLAGS.vocab_size, hparams.gen_rnn_size])
            softmax_w = tf.matrix_transpose(embedding)
            softmax_b = tf.get_variable('softmax_b', [FLAGS.vocab_size])
            rnn_inputs = tf.nn.embedding_lookup(embedding, inputs)
            if is_training and FLAGS.keep_prob < 1:
                rnn_inputs = tf.nn.dropout(rnn_inputs, FLAGS.keep_prob)
            rnn_outs = []
            fake = None
            for t in xrange(FLAGS.sequence_length):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                if t == 0:
                    rnn_inp = rnn_inputs[:, t]
                else:
                    real_rnn_inp = rnn_inputs[:, t]
                    if is_validating or FLAGS.gen_training_strategy == 'cross_entropy':
                        rnn_inp = real_rnn_inp
                    else:
                        fake_rnn_inp = tf.nn.embedding_lookup(embedding, fake)
                        rnn_inp = tf.where(targets_present[:, t - 1], real_rnn_inp, fake_rnn_inp)
                (rnn_out, state_gen) = cell_gen(rnn_inp, state_gen)
                if FLAGS.attention_option is not None:
                    rnn_out = attention_construct_fn(rnn_out, attention_keys, attention_values)
                rnn_outs.append(rnn_out)
                if FLAGS.gen_training_strategy != 'cross_entropy':
                    logit = tf.nn.bias_add(tf.matmul(rnn_out, softmax_w), softmax_b)
                    real = targets[:, t]
                    categorical = tf.contrib.distributions.Categorical(logits=logit)
                    fake = categorical.sample()
                    log_prob = categorical.log_prob(fake)
                    output = tf.where(targets_present[:, t], real, fake)
                else:
                    batch_size = tf.shape(rnn_out)[0]
                    logit = tf.zeros(tf.stack([batch_size, FLAGS.vocab_size]))
                    log_prob = tf.zeros(tf.stack([batch_size]))
                    output = targets[:, t]
                sequence.append(output)
                log_probs.append(log_prob)
                logits.append(logit)
            if FLAGS.gen_training_strategy == 'cross_entropy':
                logits = tf.nn.bias_add(tf.matmul(tf.reshape(tf.stack(rnn_outs, 1), [-1, gen_decoder_rnn_size]), softmax_w), softmax_b)
                logits = tf.reshape(logits, [-1, FLAGS.sequence_length, FLAGS.vocab_size])
            else:
                logits = tf.stack(logits, axis=1)
    return (tf.stack(sequence, axis=1), logits, tf.stack(log_probs, axis=1))

def generator(hparams, inputs, targets, targets_present, is_training, is_validating, reuse=None):
    if False:
        for i in range(10):
            print('nop')
    'Define the Generator graph.'
    with tf.variable_scope('gen', reuse=reuse):
        (encoder_states, initial_state, final_state) = gen_encoder(hparams, inputs, targets_present, is_training=is_training, reuse=reuse)
        (stacked_sequence, stacked_logits, stacked_log_probs) = gen_decoder(hparams, inputs, targets, targets_present, encoder_states, is_training=is_training, is_validating=is_validating, reuse=reuse)
        return (stacked_sequence, stacked_logits, stacked_log_probs, initial_state, final_state)