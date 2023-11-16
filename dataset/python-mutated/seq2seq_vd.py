"""Simple seq2seq model definitions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import tensorflow as tf
from models import attention_utils
from regularization import variational_dropout
FLAGS = tf.app.flags.FLAGS

def transform_input_with_is_missing_token(inputs, targets_present):
    if False:
        return 10
    "Transforms the inputs to have missing tokens when it's masked out.  The\n  mask is for the targets, so therefore, to determine if an input at time t is\n  masked, we have to check if the target at time t - 1 is masked out.\n\n  e.g.\n    inputs = [a, b, c, d]\n    targets = [b, c, d, e]\n    targets_present = [1, 0, 1, 0]\n\n  which computes,\n    inputs_present = [1, 1, 0, 1]\n\n  and outputs,\n    transformed_input = [a, b, <missing>, d]\n\n  Args:\n    inputs:  tf.int32 Tensor of shape [batch_size, sequence_length] with tokens\n      up to, but not including, vocab_size.\n    targets_present:  tf.bool Tensor of shape [batch_size, sequence_length] with\n      True representing the presence of the word.\n\n  Returns:\n    transformed_input:  tf.int32 Tensor of shape [batch_size, sequence_length]\n      which takes on value of inputs when the input is present and takes on\n      value=vocab_size to indicate a missing token.\n  "
    input_missing = tf.constant(FLAGS.vocab_size, dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.sequence_length])
    zeroth_input_present = tf.constant(True, tf.bool, shape=[FLAGS.batch_size, 1])
    inputs_present = tf.concat([zeroth_input_present, targets_present[:, :-1]], axis=1)
    transformed_input = tf.where(inputs_present, inputs, input_missing)
    return transformed_input

def gen_encoder(hparams, inputs, targets_present, is_training, reuse=None):
    if False:
        i = 10
        return i + 15
    'Define the Encoder graph.\n\n  Args:\n    hparams:  Hyperparameters for the MaskGAN.\n    inputs:  tf.int32 Tensor of shape [batch_size, sequence_length] with tokens\n      up to, but not including, vocab_size.\n    targets_present:  tf.bool Tensor of shape [batch_size, sequence_length] with\n      True representing the presence of the target.\n    is_training:  Boolean indicating operational mode (train/inference).\n    reuse (Optional):   Whether to reuse the variables.\n\n  Returns:\n    Tuple of (hidden_states, final_state).\n  '
    if FLAGS.seq2seq_share_embedding:
        with tf.variable_scope('decoder/rnn'):
            embedding = tf.get_variable('embedding', [FLAGS.vocab_size, hparams.gen_rnn_size])
    with tf.variable_scope('encoder', reuse=reuse):

        def lstm_cell():
            if False:
                i = 10
                return i + 15
            return tf.contrib.rnn.BasicLSTMCell(hparams.gen_rnn_size, forget_bias=0.0, state_is_tuple=True, reuse=reuse)
        attn_cell = lstm_cell
        if is_training and hparams.gen_vd_keep_prob < 1:

            def attn_cell():
                if False:
                    print('Hello World!')
                return variational_dropout.VariationalDropoutWrapper(lstm_cell(), FLAGS.batch_size, hparams.gen_rnn_size, hparams.gen_vd_keep_prob, hparams.gen_vd_keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(hparams.gen_num_layers)], state_is_tuple=True)
        initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)
        real_inputs = inputs
        masked_inputs = transform_input_with_is_missing_token(inputs, targets_present)
        with tf.variable_scope('rnn') as scope:
            hidden_states = []
            if not FLAGS.seq2seq_share_embedding:
                embedding = tf.get_variable('embedding', [FLAGS.vocab_size, hparams.gen_rnn_size])
            missing_embedding = tf.get_variable('missing_embedding', [1, hparams.gen_rnn_size])
            embedding = tf.concat([embedding, missing_embedding], axis=0)
            real_rnn_inputs = tf.nn.embedding_lookup(embedding, real_inputs)
            masked_rnn_inputs = tf.nn.embedding_lookup(embedding, masked_inputs)
            state = initial_state

            def make_mask(keep_prob, units):
                if False:
                    print('Hello World!')
                random_tensor = keep_prob
                random_tensor += tf.random_uniform(tf.stack([FLAGS.batch_size, 1, units]))
                return tf.floor(random_tensor) / keep_prob
            if is_training:
                output_mask = make_mask(hparams.gen_vd_keep_prob, hparams.gen_rnn_size)
            (hidden_states, state) = tf.nn.dynamic_rnn(cell, masked_rnn_inputs, initial_state=state, scope=scope)
            if is_training:
                hidden_states *= output_mask
            final_masked_state = state
            real_state = initial_state
            (_, real_state) = tf.nn.dynamic_rnn(cell, real_rnn_inputs, initial_state=real_state, scope=scope)
            final_state = real_state
    return ((hidden_states, final_masked_state), initial_state, final_state)

def gen_encoder_cnn(hparams, inputs, targets_present, is_training, reuse=None):
    if False:
        for i in range(10):
            print('nop')
    'Define the CNN Encoder graph.'
    del reuse
    sequence = transform_input_with_is_missing_token(inputs, targets_present)
    dis_filter_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    with tf.variable_scope('encoder', reuse=True):
        with tf.variable_scope('rnn'):
            embedding = tf.get_variable('embedding', [FLAGS.vocab_size, hparams.gen_rnn_size])
    cnn_inputs = tf.nn.embedding_lookup(embedding, sequence)
    conv_outputs = []
    for filter_size in dis_filter_sizes:
        with tf.variable_scope('conv-%s' % filter_size):
            filter_shape = [filter_size, hparams.gen_rnn_size, hparams.dis_num_filters]
            W = tf.get_variable(name='W', initializer=tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.get_variable(name='b', initializer=tf.constant(0.1, shape=[hparams.dis_num_filters]))
            conv = tf.nn.conv1d(cnn_inputs, W, stride=1, padding='SAME', name='conv')
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            conv_outputs.append(h)
    dis_num_filters_total = hparams.dis_num_filters * len(dis_filter_sizes)
    h_conv = tf.concat(conv_outputs, axis=2)
    h_conv_flat = tf.reshape(h_conv, [-1, dis_num_filters_total])
    if is_training:
        with tf.variable_scope('dropout'):
            h_conv_flat = tf.nn.dropout(h_conv_flat, hparams.gen_vd_keep_prob)
    with tf.variable_scope('output'):
        W = tf.get_variable('W', shape=[dis_num_filters_total, hparams.gen_rnn_size], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b', initializer=tf.constant(0.1, shape=[hparams.gen_rnn_size]))
        predictions = tf.nn.xw_plus_b(h_conv_flat, W, b, name='predictions')
        predictions = tf.reshape(predictions, shape=[FLAGS.batch_size, FLAGS.sequence_length, hparams.gen_rnn_size])
    final_state = tf.reduce_mean(predictions, 1)
    return (predictions, (final_state, final_state))

def gen_decoder(hparams, inputs, targets, targets_present, encoding_state, is_training, is_validating, reuse=None):
    if False:
        print('Hello World!')
    'Define the Decoder graph. The Decoder will now impute tokens that\n      have been masked from the input seqeunce.\n  '
    gen_decoder_rnn_size = hparams.gen_rnn_size
    targets = tf.Print(targets, [targets], message='targets', summarize=50)
    if FLAGS.seq2seq_share_embedding:
        with tf.variable_scope('decoder/rnn', reuse=True):
            embedding = tf.get_variable('embedding', [FLAGS.vocab_size, hparams.gen_rnn_size])
    with tf.variable_scope('decoder', reuse=reuse):

        def lstm_cell():
            if False:
                return 10
            return tf.contrib.rnn.BasicLSTMCell(gen_decoder_rnn_size, forget_bias=0.0, state_is_tuple=True, reuse=reuse)
        attn_cell = lstm_cell
        if is_training and hparams.gen_vd_keep_prob < 1:

            def attn_cell():
                if False:
                    print('Hello World!')
                return variational_dropout.VariationalDropoutWrapper(lstm_cell(), FLAGS.batch_size, hparams.gen_rnn_size, hparams.gen_vd_keep_prob, hparams.gen_vd_keep_prob)
        cell_gen = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(hparams.gen_num_layers)], state_is_tuple=True)
        hidden_vector_encodings = encoding_state[0]
        state_gen = encoding_state[1]
        if FLAGS.attention_option is not None:
            (attention_keys, attention_values, _, attention_construct_fn) = attention_utils.prepare_attention(hidden_vector_encodings, FLAGS.attention_option, num_units=gen_decoder_rnn_size, reuse=reuse)

        def make_mask(keep_prob, units):
            if False:
                i = 10
                return i + 15
            random_tensor = keep_prob
            random_tensor += tf.random_uniform(tf.stack([FLAGS.batch_size, units]))
            return tf.floor(random_tensor) / keep_prob
        if is_training:
            output_mask = make_mask(hparams.gen_vd_keep_prob, hparams.gen_rnn_size)
        with tf.variable_scope('rnn'):
            (sequence, logits, log_probs) = ([], [], [])
            if not FLAGS.seq2seq_share_embedding:
                embedding = tf.get_variable('embedding', [FLAGS.vocab_size, hparams.gen_rnn_size])
            softmax_w = tf.matrix_transpose(embedding)
            softmax_b = tf.get_variable('softmax_b', [FLAGS.vocab_size])
            rnn_inputs = tf.nn.embedding_lookup(embedding, inputs)
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
                if is_training:
                    rnn_out *= output_mask
                rnn_outs.append(rnn_out)
                if FLAGS.gen_training_strategy != 'cross_entropy':
                    logit = tf.nn.bias_add(tf.matmul(rnn_out, softmax_w), softmax_b)
                    real = targets[:, t]
                    categorical = tf.contrib.distributions.Categorical(logits=logit)
                    if FLAGS.use_gen_mode:
                        fake = categorical.mode()
                    else:
                        fake = categorical.sample()
                    log_prob = categorical.log_prob(fake)
                    output = tf.where(targets_present[:, t], real, fake)
                else:
                    real = targets[:, t]
                    logit = tf.zeros(tf.stack([FLAGS.batch_size, FLAGS.vocab_size]))
                    log_prob = tf.zeros(tf.stack([FLAGS.batch_size]))
                    output = real
                sequence.append(output)
                log_probs.append(log_prob)
                logits.append(logit)
            if FLAGS.gen_training_strategy == 'cross_entropy':
                logits = tf.nn.bias_add(tf.matmul(tf.reshape(tf.stack(rnn_outs, 1), [-1, gen_decoder_rnn_size]), softmax_w), softmax_b)
                logits = tf.reshape(logits, [-1, FLAGS.sequence_length, FLAGS.vocab_size])
            else:
                logits = tf.stack(logits, axis=1)
    return (tf.stack(sequence, axis=1), logits, tf.stack(log_probs, axis=1))

def dis_encoder(hparams, masked_inputs, is_training, reuse=None, embedding=None):
    if False:
        print('Hello World!')
    'Define the Discriminator encoder.  Reads in the masked inputs for context\n  and produces the hidden states of the encoder.'
    with tf.variable_scope('encoder', reuse=reuse):

        def lstm_cell():
            if False:
                while True:
                    i = 10
            return tf.contrib.rnn.BasicLSTMCell(hparams.dis_rnn_size, forget_bias=0.0, state_is_tuple=True, reuse=reuse)
        attn_cell = lstm_cell
        if is_training and hparams.dis_vd_keep_prob < 1:

            def attn_cell():
                if False:
                    for i in range(10):
                        print('nop')
                return variational_dropout.VariationalDropoutWrapper(lstm_cell(), FLAGS.batch_size, hparams.dis_rnn_size, hparams.dis_vd_keep_prob, hparams.dis_vd_keep_prob)
        cell_dis = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(hparams.dis_num_layers)], state_is_tuple=True)
        state_dis = cell_dis.zero_state(FLAGS.batch_size, tf.float32)
        with tf.variable_scope('rnn'):
            hidden_states = []
            missing_embedding = tf.get_variable('missing_embedding', [1, hparams.dis_rnn_size])
            embedding = tf.concat([embedding, missing_embedding], axis=0)
            masked_rnn_inputs = tf.nn.embedding_lookup(embedding, masked_inputs)

            def make_mask(keep_prob, units):
                if False:
                    for i in range(10):
                        print('nop')
                random_tensor = keep_prob
                random_tensor += tf.random_uniform(tf.stack([FLAGS.batch_size, units]))
                return tf.floor(random_tensor) / keep_prob
            if is_training:
                output_mask = make_mask(hparams.dis_vd_keep_prob, hparams.dis_rnn_size)
            for t in xrange(FLAGS.sequence_length):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                rnn_in = masked_rnn_inputs[:, t]
                (rnn_out, state_dis) = cell_dis(rnn_in, state_dis)
                if is_training:
                    rnn_out *= output_mask
                hidden_states.append(rnn_out)
            final_state = state_dis
    return (tf.stack(hidden_states, axis=1), final_state)

def dis_decoder(hparams, sequence, encoding_state, is_training, reuse=None, embedding=None):
    if False:
        for i in range(10):
            print('nop')
    'Define the Discriminator decoder.  Read in the sequence and predict\n    at each time point.'
    sequence = tf.cast(sequence, tf.int32)
    with tf.variable_scope('decoder', reuse=reuse):

        def lstm_cell():
            if False:
                print('Hello World!')
            return tf.contrib.rnn.BasicLSTMCell(hparams.dis_rnn_size, forget_bias=0.0, state_is_tuple=True, reuse=reuse)
        attn_cell = lstm_cell
        if is_training and hparams.dis_vd_keep_prob < 1:

            def attn_cell():
                if False:
                    return 10
                return variational_dropout.VariationalDropoutWrapper(lstm_cell(), FLAGS.batch_size, hparams.dis_rnn_size, hparams.dis_vd_keep_prob, hparams.dis_vd_keep_prob)
        cell_dis = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(hparams.dis_num_layers)], state_is_tuple=True)
        hidden_vector_encodings = encoding_state[0]
        state = encoding_state[1]
        if FLAGS.attention_option is not None:
            (attention_keys, attention_values, _, attention_construct_fn) = attention_utils.prepare_attention(hidden_vector_encodings, FLAGS.attention_option, num_units=hparams.dis_rnn_size, reuse=reuse)

        def make_mask(keep_prob, units):
            if False:
                while True:
                    i = 10
            random_tensor = keep_prob
            random_tensor += tf.random_uniform(tf.stack([FLAGS.batch_size, units]))
            return tf.floor(random_tensor) / keep_prob
        if is_training:
            output_mask = make_mask(hparams.dis_vd_keep_prob, hparams.dis_rnn_size)
        with tf.variable_scope('rnn') as vs:
            predictions = []
            rnn_inputs = tf.nn.embedding_lookup(embedding, sequence)
            for t in xrange(FLAGS.sequence_length):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                rnn_in = rnn_inputs[:, t]
                (rnn_out, state) = cell_dis(rnn_in, state)
                if FLAGS.attention_option is not None:
                    rnn_out = attention_construct_fn(rnn_out, attention_keys, attention_values)
                if is_training:
                    rnn_out *= output_mask
                pred = tf.contrib.layers.linear(rnn_out, 1, scope=vs)
                predictions.append(pred)
    predictions = tf.stack(predictions, axis=1)
    return tf.squeeze(predictions, axis=2)

def discriminator(hparams, inputs, targets_present, sequence, is_training, reuse=None):
    if False:
        while True:
            i = 10
    'Define the Discriminator graph.'
    if FLAGS.dis_share_embedding:
        assert hparams.dis_rnn_size == hparams.gen_rnn_size, 'If you wish to share Discriminator/Generator embeddings, they must be same dimension.'
        with tf.variable_scope('gen/decoder/rnn', reuse=True):
            embedding = tf.get_variable('embedding', [FLAGS.vocab_size, hparams.gen_rnn_size])
    else:
        with tf.variable_scope('dis/decoder/rnn', reuse=reuse):
            embedding = tf.get_variable('embedding', [FLAGS.vocab_size, hparams.dis_rnn_size])
    masked_inputs = transform_input_with_is_missing_token(inputs, targets_present)
    masked_inputs = tf.Print(masked_inputs, [inputs, targets_present, masked_inputs, sequence], message='inputs, targets_present, masked_inputs, sequence', summarize=10)
    with tf.variable_scope('dis', reuse=reuse):
        encoder_states = dis_encoder(hparams, masked_inputs, is_training=is_training, reuse=reuse, embedding=embedding)
        predictions = dis_decoder(hparams, sequence, encoder_states, is_training=is_training, reuse=reuse, embedding=embedding)
    return predictions

def generator(hparams, inputs, targets, targets_present, is_training, is_validating, reuse=None):
    if False:
        while True:
            i = 10
    'Define the Generator graph.'
    with tf.variable_scope('gen', reuse=reuse):
        (encoder_states, initial_state, final_state) = gen_encoder(hparams, inputs, targets_present, is_training=is_training, reuse=reuse)
        (stacked_sequence, stacked_logits, stacked_log_probs) = gen_decoder(hparams, inputs, targets, targets_present, encoder_states, is_training=is_training, is_validating=is_validating, reuse=reuse)
        return (stacked_sequence, stacked_logits, stacked_log_probs, initial_state, final_state, encoder_states)