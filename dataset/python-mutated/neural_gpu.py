"""The Neural GPU Model."""
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.framework import function
import data_utils as data
do_jit = False
jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

def conv_linear(args, kw, kh, nin, nout, rate, do_bias, bias_start, prefix):
    if False:
        return 10
    'Convolutional linear map.'
    if not isinstance(args, (list, tuple)):
        args = [args]
    with tf.variable_scope(prefix):
        with tf.device('/cpu:0'):
            k = tf.get_variable('CvK', [kw, kh, nin, nout])
        if len(args) == 1:
            arg = args[0]
        else:
            arg = tf.concat(axis=3, values=args)
        res = tf.nn.convolution(arg, k, dilation_rate=(rate, 1), padding='SAME')
        if not do_bias:
            return res
        with tf.device('/cpu:0'):
            bias_term = tf.get_variable('CvB', [nout], initializer=tf.constant_initializer(bias_start))
        bias_term = tf.reshape(bias_term, [1, 1, 1, nout])
        return res + bias_term

def sigmoid_cutoff(x, cutoff):
    if False:
        i = 10
        return i + 15
    'Sigmoid with cutoff, e.g., 1.2sigmoid(x) - 0.1.'
    y = tf.sigmoid(x)
    if cutoff < 1.01:
        return y
    d = (cutoff - 1.0) / 2.0
    return tf.minimum(1.0, tf.maximum(0.0, cutoff * y - d), name='cutoff_min')

@function.Defun(tf.float32, noinline=True)
def sigmoid_cutoff_12(x):
    if False:
        while True:
            i = 10
    'Sigmoid with cutoff 1.2, specialized for speed and memory use.'
    y = tf.sigmoid(x)
    return tf.minimum(1.0, tf.maximum(0.0, 1.2 * y - 0.1), name='cutoff_min_12')

@function.Defun(tf.float32, noinline=True)
def sigmoid_hard(x):
    if False:
        print('Hello World!')
    'Hard sigmoid.'
    return tf.minimum(1.0, tf.maximum(0.0, 0.25 * x + 0.5))

def place_at14(decided, selected, it):
    if False:
        print('Hello World!')
    'Place selected at it-th coordinate of decided, dim=1 of 4.'
    slice1 = decided[:, :it, :, :]
    slice2 = decided[:, it + 1:, :, :]
    return tf.concat(axis=1, values=[slice1, selected, slice2])

def place_at13(decided, selected, it):
    if False:
        i = 10
        return i + 15
    'Place selected at it-th coordinate of decided, dim=1 of 3.'
    slice1 = decided[:, :it, :]
    slice2 = decided[:, it + 1:, :]
    return tf.concat(axis=1, values=[slice1, selected, slice2])

def tanh_cutoff(x, cutoff):
    if False:
        return 10
    'Tanh with cutoff, e.g., 1.1tanh(x) cut to [-1. 1].'
    y = tf.tanh(x)
    if cutoff < 1.01:
        return y
    d = (cutoff - 1.0) / 2.0
    return tf.minimum(1.0, tf.maximum(-1.0, (1.0 + d) * y))

@function.Defun(tf.float32, noinline=True)
def tanh_hard(x):
    if False:
        return 10
    'Hard tanh.'
    return tf.minimum(1.0, tf.maximum(0.0, x))

def layer_norm(x, nmaps, prefix, epsilon=1e-05):
    if False:
        print('Hello World!')
    'Layer normalize the 4D tensor x, averaging over the last dimension.'
    with tf.variable_scope(prefix):
        scale = tf.get_variable('layer_norm_scale', [nmaps], initializer=tf.ones_initializer())
        bias = tf.get_variable('layer_norm_bias', [nmaps], initializer=tf.zeros_initializer())
        (mean, variance) = tf.nn.moments(x, [3], keep_dims=True)
        norm_x = (x - mean) / tf.sqrt(variance + epsilon)
        return norm_x * scale + bias

def conv_gru(inpts, mem, kw, kh, nmaps, rate, cutoff, prefix, do_layer_norm, args_len=None):
    if False:
        return 10
    'Convolutional GRU.'

    def conv_lin(args, suffix, bias_start):
        if False:
            for i in range(10):
                print('nop')
        total_args_len = args_len or len(args) * nmaps
        res = conv_linear(args, kw, kh, total_args_len, nmaps, rate, True, bias_start, prefix + '/' + suffix)
        if do_layer_norm:
            return layer_norm(res, nmaps, prefix + '/' + suffix)
        else:
            return res
    if cutoff == 1.2:
        reset = sigmoid_cutoff_12(conv_lin(inpts + [mem], 'r', 1.0))
        gate = sigmoid_cutoff_12(conv_lin(inpts + [mem], 'g', 1.0))
    elif cutoff > 10:
        reset = sigmoid_hard(conv_lin(inpts + [mem], 'r', 1.0))
        gate = sigmoid_hard(conv_lin(inpts + [mem], 'g', 1.0))
    else:
        reset = sigmoid_cutoff(conv_lin(inpts + [mem], 'r', 1.0), cutoff)
        gate = sigmoid_cutoff(conv_lin(inpts + [mem], 'g', 1.0), cutoff)
    if cutoff > 10:
        candidate = tanh_hard(conv_lin(inpts + [reset * mem], 'c', 0.0))
    else:
        candidate = tf.tanh(conv_lin(inpts + [reset * mem], 'c', 0.0))
    return gate * mem + (1 - gate) * candidate
CHOOSE_K = 256

def memory_call(q, l, nmaps, mem_size, vocab_size, num_gpus, update_mem):
    if False:
        i = 10
        return i + 15
    raise ValueError('Fill for experiments with additional memory structures.')

def memory_run(step, nmaps, mem_size, batch_size, vocab_size, global_step, do_training, update_mem, decay_factor, num_gpus, target_emb_weights, output_w, gpu_targets_tn, it):
    if False:
        return 10
    'Run memory.'
    q = step[:, 0, it, :]
    mlabels = gpu_targets_tn[:, it, 0]
    (res, mask, mem_loss) = memory_call(q, mlabels, nmaps, mem_size, vocab_size, num_gpus, update_mem)
    res = tf.gather(target_emb_weights, res) * tf.expand_dims(mask[:, 0], 1)
    gold = tf.nn.dropout(tf.gather(target_emb_weights, mlabels), 0.7)
    use_gold = 1.0 - tf.cast(global_step, tf.float32) / (1000.0 * decay_factor)
    use_gold = tf.maximum(use_gold, 0.2) * do_training
    mem = tf.cond(tf.less(tf.random_uniform([]), use_gold), lambda : use_gold * gold + (1.0 - use_gold) * res, lambda : res)
    mem = tf.reshape(mem, [-1, 1, 1, nmaps])
    return (mem, mem_loss, update_mem)

@tf.RegisterGradient('CustomIdG')
def _custom_id_grad(_, grads):
    if False:
        while True:
            i = 10
    return grads

def quantize(t, quant_scale, max_value=1.0):
    if False:
        for i in range(10):
            print('nop')
    'Quantize a tensor t with each element in [-max_value, max_value].'
    t = tf.minimum(max_value, tf.maximum(t, -max_value))
    big = quant_scale * (t + max_value) + 0.5
    with tf.get_default_graph().gradient_override_map({'Floor': 'CustomIdG'}):
        res = tf.floor(big) / quant_scale - max_value
    return res

def quantize_weights_op(quant_scale, max_value):
    if False:
        print('Hello World!')
    ops = [v.assign(quantize(v, quant_scale, float(max_value))) for v in tf.trainable_variables()]
    return tf.group(*ops)

def autoenc_quantize(x, nbits, nmaps, do_training, layers=1):
    if False:
        for i in range(10):
            print('nop')
    'Autoencoder into nbits vectors of bits, using noise and sigmoids.'
    enc_x = tf.reshape(x, [-1, nmaps])
    for i in xrange(layers - 1):
        enc_x = tf.layers.dense(enc_x, nmaps, name='autoenc_%d' % i)
    enc_x = tf.layers.dense(enc_x, nbits, name='autoenc_%d' % (layers - 1))
    noise = tf.truncated_normal(tf.shape(enc_x), stddev=2.0)
    dec_x = sigmoid_cutoff_12(enc_x + noise * do_training)
    dec_x = tf.reshape(dec_x, [-1, nbits])
    for i in xrange(layers):
        dec_x = tf.layers.dense(dec_x, nmaps, name='autodec_%d' % i)
    return tf.reshape(dec_x, tf.shape(x))

def make_dense(targets, noclass, low_param):
    if False:
        i = 10
        return i + 15
    'Move a batch of targets to a dense 1-hot representation.'
    low = low_param / float(noclass - 1)
    high = 1.0 - low * (noclass - 1)
    targets = tf.cast(targets, tf.int64)
    return tf.one_hot(targets, depth=noclass, on_value=high, off_value=low)

def reorder_beam(beam_size, batch_size, beam_val, output, is_first, tensors_to_reorder):
    if False:
        for i in range(10):
            print('nop')
    'Reorder to minimize beam costs.'
    outputs = tf.split(axis=0, num_or_size_splits=beam_size, value=tf.nn.log_softmax(output))
    (all_beam_vals, all_beam_idx) = ([], [])
    beam_range = 1 if is_first else beam_size
    for i in xrange(beam_range):
        (top_out, top_out_idx) = tf.nn.top_k(outputs[i], k=beam_size)
        cur_beam_val = beam_val[:, i]
        top_out = tf.Print(top_out, [top_out, top_out_idx, beam_val, i, cur_beam_val], 'GREPO', summarize=8)
        all_beam_vals.append(top_out + tf.expand_dims(cur_beam_val, 1))
        all_beam_idx.append(top_out_idx)
    all_beam_idx = tf.reshape(tf.transpose(tf.concat(axis=1, values=all_beam_idx), [1, 0]), [-1])
    (top_beam, top_beam_idx) = tf.nn.top_k(tf.concat(axis=1, values=all_beam_vals), k=beam_size)
    top_beam_idx = tf.Print(top_beam_idx, [top_beam, top_beam_idx], 'GREP', summarize=8)
    reordered = [[] for _ in xrange(len(tensors_to_reorder) + 1)]
    top_out_idx = []
    for i in xrange(beam_size):
        which_idx = top_beam_idx[:, i] * batch_size + tf.range(batch_size)
        top_out_idx.append(tf.gather(all_beam_idx, which_idx))
        which_beam = top_beam_idx[:, i] / beam_size
        which_beam = which_beam * batch_size + tf.range(batch_size)
        reordered[0].append(tf.gather(output, which_beam))
        for (i, t) in enumerate(tensors_to_reorder):
            reordered[i + 1].append(tf.gather(t, which_beam))
    new_tensors = [tf.concat(axis=0, values=t) for t in reordered]
    top_out_idx = tf.concat(axis=0, values=top_out_idx)
    return (top_beam, new_tensors[0], top_out_idx, new_tensors[1:])

class NeuralGPU(object):
    """Neural GPU Model."""

    def __init__(self, nmaps, vec_size, niclass, noclass, dropout, max_grad_norm, cutoff, nconvs, kw, kh, height, mem_size, learning_rate, min_length, num_gpus, num_replicas, grad_noise_scale, sampling_rate, act_noise=0.0, do_rnn=False, atrous=False, beam_size=1, backward=True, do_layer_norm=False, autoenc_decay=1.0):
        if False:
            i = 10
            return i + 15
        self.nmaps = nmaps
        if backward:
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.cur_length = tf.Variable(min_length, trainable=False)
            self.cur_length_incr_op = self.cur_length.assign_add(1)
            self.lr = tf.Variable(learning_rate, trainable=False)
            self.lr_decay_op = self.lr.assign(self.lr * 0.995)
        self.do_training = tf.placeholder(tf.float32, name='do_training')
        self.update_mem = tf.placeholder(tf.int32, name='update_mem')
        self.noise_param = tf.placeholder(tf.float32, name='noise_param')
        self.input = tf.placeholder(tf.int32, name='inp')
        self.target = tf.placeholder(tf.int32, name='tgt')
        self.prev_step = tf.placeholder(tf.float32, name='prev_step')
        gpu_input = tf.split(axis=0, num_or_size_splits=num_gpus, value=self.input)
        gpu_target = tf.split(axis=0, num_or_size_splits=num_gpus, value=self.target)
        gpu_prev_step = tf.split(axis=0, num_or_size_splits=num_gpus, value=self.prev_step)
        batch_size = tf.shape(gpu_input[0])[0]
        if backward:
            adam_lr = 0.005 * self.lr
            adam = tf.train.AdamOptimizer(adam_lr, epsilon=0.001)

            def adam_update(grads):
                if False:
                    while True:
                        i = 10
                return adam.apply_gradients(zip(grads, tf.trainable_variables()), global_step=self.global_step, name='adam_update')
        if backward:
            global_step_float = tf.cast(self.global_step, tf.float32)
            sampling_decay_exponent = global_step_float / 100000.0
            sampling_decay = tf.maximum(0.05, tf.pow(0.5, sampling_decay_exponent))
            self.sampling = sampling_rate * 0.05 / sampling_decay
        else:
            self.sampling = tf.constant(0.0)
        if num_replicas > 1 or num_gpus > 1:
            with tf.device('/cpu:0'):
                caching_const = tf.constant(0)
            tf.get_variable_scope().set_caching_device(caching_const.op.device)

        def gpu_avg(l):
            if False:
                while True:
                    i = 10
            if l[0] is None:
                for elem in l:
                    assert elem is None
                return 0.0
            if len(l) < 2:
                return l[0]
            return sum(l) / float(num_gpus)
        self.length_tensor = tf.placeholder(tf.int32, name='length')
        with tf.device('/cpu:0'):
            emb_weights = tf.get_variable('embedding', [niclass, vec_size], initializer=tf.random_uniform_initializer(-1.7, 1.7))
            if beam_size > 0:
                target_emb_weights = tf.get_variable('target_embedding', [noclass, nmaps], initializer=tf.random_uniform_initializer(-1.7, 1.7))
            e0 = tf.scatter_update(emb_weights, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros([1, vec_size]))
            output_w = tf.get_variable('output_w', [nmaps, noclass], tf.float32)

        def conv_rate(layer):
            if False:
                i = 10
                return i + 15
            if atrous:
                return 2 ** layer
            return 1

        def enc_step(step):
            if False:
                for i in range(10):
                    print('nop')
            'Encoder step.'
            if autoenc_decay < 1.0:
                quant_step = autoenc_quantize(step, 16, nmaps, self.do_training)
                if backward:
                    exp_glob = tf.train.exponential_decay(1.0, self.global_step - 10000, 1000, autoenc_decay)
                    dec_factor = 1.0 - exp_glob
                    dec_factor = tf.cond(tf.less(self.global_step, 10500), lambda : tf.constant(0.05), lambda : dec_factor)
                else:
                    dec_factor = 1.0
                cur = tf.cond(tf.less(tf.random_uniform([]), dec_factor), lambda : quant_step, lambda : step)
            else:
                cur = step
            if dropout > 0.0001:
                cur = tf.nn.dropout(cur, keep_prob)
            if act_noise > 1e-05:
                cur += tf.truncated_normal(tf.shape(cur)) * act_noise_scale
            if do_jit and tf.get_variable_scope().reuse:
                with jit_scope():
                    for layer in xrange(nconvs):
                        cur = conv_gru([], cur, kw, kh, nmaps, conv_rate(layer), cutoff, 'ecgru_%d' % layer, do_layer_norm)
            else:
                for layer in xrange(nconvs):
                    cur = conv_gru([], cur, kw, kh, nmaps, conv_rate(layer), cutoff, 'ecgru_%d' % layer, do_layer_norm)
            return cur
        zero_tgt = tf.zeros([batch_size, nmaps, 1])
        zero_tgt.set_shape([None, nmaps, 1])

        def dec_substep(step, decided):
            if False:
                return 10
            'Decoder sub-step.'
            cur = step
            if dropout > 0.0001:
                cur = tf.nn.dropout(cur, keep_prob)
            if act_noise > 1e-05:
                cur += tf.truncated_normal(tf.shape(cur)) * act_noise_scale
            if do_jit and tf.get_variable_scope().reuse:
                with jit_scope():
                    for layer in xrange(nconvs):
                        cur = conv_gru([decided], cur, kw, kh, nmaps, conv_rate(layer), cutoff, 'dcgru_%d' % layer, do_layer_norm)
            else:
                for layer in xrange(nconvs):
                    cur = conv_gru([decided], cur, kw, kh, nmaps, conv_rate(layer), cutoff, 'dcgru_%d' % layer, do_layer_norm)
            return cur

        def dec_step(step, it, it_int, decided, output_ta, tgts, mloss, nupd_in, out_idx, beam_cost):
            if False:
                while True:
                    i = 10
            'Decoder step.'
            (nupd, mem_loss) = (0, 0.0)
            if mem_size > 0:
                it_incr = tf.minimum(it + 1, length - 1)
                (mem, mem_loss, nupd) = memory_run(step, nmaps, mem_size, batch_size, noclass, self.global_step, self.do_training, self.update_mem, 10, num_gpus, target_emb_weights, output_w, gpu_targets_tn, it_incr)
            step = dec_substep(step, decided)
            output_l = tf.expand_dims(tf.expand_dims(step[:, it, 0, :], 1), 1)
            output = tf.reshape(output_l, [-1, nmaps])
            output = tf.matmul(output, output_w)
            if beam_size > 1:
                (beam_cost, output, out, reordered) = reorder_beam(beam_size, batch_size, beam_cost, output, it_int == 0, [output_l, out_idx, step, decided])
                [output_l, out_idx, step, decided] = reordered
            else:
                out = tf.multinomial(tf.stop_gradient(output), 1)
                out = tf.to_int32(tf.squeeze(out, [1]))
            out_write = output_ta.write(it, output_l[:batch_size, :, :, :])
            output = tf.gather(target_emb_weights, out)
            output = tf.reshape(output, [-1, 1, nmaps])
            output = tf.concat(axis=1, values=[output] * height)
            tgt = tgts[it, :, :, :]
            selected = tf.cond(tf.less(tf.random_uniform([]), self.sampling), lambda : output, lambda : tgt)
            dec_write = place_at14(decided, tf.expand_dims(selected, 1), it)
            out_idx = place_at13(out_idx, tf.reshape(out, [beam_size * batch_size, 1, 1]), it)
            if mem_size > 0:
                mem = tf.concat(axis=2, values=[mem] * height)
                dec_write = place_at14(dec_write, mem, it_incr)
            return (step, dec_write, out_write, mloss + mem_loss, nupd_in + nupd, out_idx, beam_cost)
        gpu_outputs = []
        gpu_losses = []
        gpu_grad_norms = []
        grads_list = []
        gpu_out_idx = []
        self.after_enc_step = []
        for gpu in xrange(num_gpus):
            length = self.length_tensor
            length_float = tf.cast(length, tf.float32)
            if gpu > 0:
                tf.get_variable_scope().reuse_variables()
            gpu_outputs.append([])
            gpu_losses.append([])
            gpu_grad_norms.append([])
            with tf.name_scope('gpu%d' % gpu), tf.device('/gpu:%d' % gpu):
                data.print_out('Creating model.')
                start_time = time.time()
                with tf.device('/cpu:0'):
                    tgt_shape = tf.shape(tf.squeeze(gpu_target[gpu], [1]))
                    weights = tf.where(tf.squeeze(gpu_target[gpu], [1]) > 0, tf.ones(tgt_shape), tf.zeros(tgt_shape))
                    with tf.control_dependencies([e0]):
                        start = tf.gather(emb_weights, gpu_input[gpu])
                        gpu_targets_tn = gpu_target[gpu]
                        if beam_size > 0:
                            embedded_targets_tn = tf.gather(target_emb_weights, gpu_targets_tn)
                            embedded_targets_tn = tf.transpose(embedded_targets_tn, [2, 0, 1, 3])
                            embedded_targets_tn = tf.concat(axis=2, values=[embedded_targets_tn] * height)
                start = tf.transpose(start, [0, 2, 1, 3])
                first = conv_linear(start, 1, 1, vec_size, nmaps, 1, True, 0.0, 'input')
                first = layer_norm(first, nmaps, 'input')
                keep_prob = dropout * 3.0 / tf.sqrt(length_float)
                keep_prob = 1.0 - self.do_training * keep_prob
                act_noise_scale = act_noise * self.do_training
                step = conv_gru([gpu_prev_step[gpu]], first, kw, kh, nmaps, 1, cutoff, 'first', do_layer_norm)
                if do_rnn:
                    self.after_enc_step.append(step)

                    def lstm_cell():
                        if False:
                            for i in range(10):
                                print('nop')
                        return tf.contrib.rnn.BasicLSTMCell(height * nmaps)
                    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(nconvs)])
                    with tf.variable_scope('encoder'):
                        (encoder_outputs, encoder_state) = tf.nn.dynamic_rnn(cell, tf.reshape(step, [batch_size, length, height * nmaps]), dtype=tf.float32, time_major=False)
                    attn = tf.layers.dense(encoder_outputs, height * nmaps, name='attn1')

                    @function.Defun(noinline=True)
                    def attention_query(query, attn_v):
                        if False:
                            return 10
                        vecs = tf.tanh(attn + tf.expand_dims(query, 1))
                        mask = tf.reduce_sum(vecs * tf.reshape(attn_v, [1, 1, -1]), 2)
                        mask = tf.nn.softmax(mask)
                        return tf.reduce_sum(encoder_outputs * tf.expand_dims(mask, 2), 1)
                    with tf.variable_scope('decoder'):

                        def decoder_loop_fn(state__prev_cell_out__unused, cell_inp__cur_tgt):
                            if False:
                                for i in range(10):
                                    print('nop')
                            'Decoder loop function.'
                            (state, prev_cell_out, _) = state__prev_cell_out__unused
                            (cell_inp, cur_tgt) = cell_inp__cur_tgt
                            attn_q = tf.layers.dense(prev_cell_out, height * nmaps, name='attn_query')
                            attn_res = attention_query(attn_q, tf.get_variable('attn_v', [height * nmaps], initializer=tf.random_uniform_initializer(-0.1, 0.1)))
                            concatenated = tf.reshape(tf.concat(axis=1, values=[cell_inp, attn_res]), [batch_size, 2 * height * nmaps])
                            cell_inp = tf.layers.dense(concatenated, height * nmaps, name='attn_merge')
                            (output, new_state) = cell(cell_inp, state)
                            mem_loss = 0.0
                            if mem_size > 0:
                                (res, mask, mem_loss) = memory_call(output, cur_tgt, height * nmaps, mem_size, noclass, num_gpus, self.update_mem)
                                res = tf.gather(target_emb_weights, res)
                                res *= tf.expand_dims(mask[:, 0], 1)
                                output = tf.layers.dense(tf.concat(axis=1, values=[output, res]), height * nmaps, name='rnnmem')
                            return (new_state, output, mem_loss)
                        gpu_targets = tf.squeeze(gpu_target[gpu], [1])
                        gpu_tgt_trans = tf.transpose(gpu_targets, [1, 0])
                        dec_zero = tf.zeros([batch_size, 1], dtype=tf.int32)
                        dec_inp = tf.concat(axis=1, values=[dec_zero, gpu_targets])
                        dec_inp = dec_inp[:, :length]
                        embedded_dec_inp = tf.gather(target_emb_weights, dec_inp)
                        embedded_dec_inp_proj = tf.layers.dense(embedded_dec_inp, height * nmaps, name='dec_proj')
                        embedded_dec_inp_proj = tf.transpose(embedded_dec_inp_proj, [1, 0, 2])
                        init_vals = (encoder_state, tf.zeros([batch_size, height * nmaps]), 0.0)
                        (_, dec_outputs, mem_losses) = tf.scan(decoder_loop_fn, (embedded_dec_inp_proj, gpu_tgt_trans), initializer=init_vals)
                    mem_loss = tf.reduce_mean(mem_losses)
                    outputs = tf.layers.dense(dec_outputs, nmaps, name='out_proj')
                    outputs = tf.matmul(tf.reshape(outputs, [-1, nmaps]), output_w)
                    outputs = tf.reshape(outputs, [length, batch_size, noclass])
                    gpu_out_idx.append(tf.argmax(outputs, 2))
                else:
                    enc_length = length
                    step = enc_step(step)
                    i = tf.constant(1)
                    c = lambda i, _s: tf.less(i, enc_length)

                    def enc_step_lambda(i, step):
                        if False:
                            for i in range(10):
                                print('nop')
                        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                            new_step = enc_step(step)
                        return (i + 1, new_step)
                    (_, step) = tf.while_loop(c, enc_step_lambda, [i, step], parallel_iterations=1, swap_memory=True)
                    self.after_enc_step.append(step)
                    if beam_size > 0:
                        output_ta = tf.TensorArray(dtype=tf.float32, size=length, dynamic_size=False, infer_shape=False, name='outputs')
                        out_idx = tf.zeros([beam_size * batch_size, length, 1], dtype=tf.int32)
                        decided_t = tf.zeros([beam_size * batch_size, length, height, vec_size])
                        tgts = tf.concat(axis=1, values=[embedded_targets_tn] * beam_size)
                        beam_cost = tf.zeros([batch_size, beam_size])
                        step = tf.concat(axis=0, values=[step] * beam_size)
                        (step, decided_t, output_ta, mem_loss, nupd, oi, bc) = dec_step(step, 0, 0, decided_t, output_ta, tgts, 0.0, 0, out_idx, beam_cost)
                        tf.get_variable_scope().reuse_variables()

                        def step_lambda(i, step, dec_t, out_ta, ml, nu, oi, bc):
                            if False:
                                for i in range(10):
                                    print('nop')
                            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                                (s, d, t, nml, nu, oi, bc) = dec_step(step, i, 1, dec_t, out_ta, tgts, ml, nu, oi, bc)
                            return (i + 1, s, d, t, nml, nu, oi, bc)
                        i = tf.constant(1)
                        c = lambda i, _s, _d, _o, _ml, _nu, _oi, _bc: tf.less(i, length)
                        (_, step, _, output_ta, mem_loss, nupd, out_idx, _) = tf.while_loop(c, step_lambda, [i, step, decided_t, output_ta, mem_loss, nupd, oi, bc], parallel_iterations=1, swap_memory=True)
                        gpu_out_idx.append(tf.squeeze(out_idx, [2]))
                        outputs = output_ta.stack()
                        outputs = tf.squeeze(outputs, [2, 3])
                    else:
                        mem_loss = 0.0
                        outputs = tf.transpose(step[:, :, 1, :], [1, 0, 2])
                        gpu_out_idx.append(tf.argmax(outputs, 2))
                    outputs = tf.matmul(tf.reshape(outputs, [-1, nmaps]), output_w)
                    outputs = tf.reshape(outputs, [length, batch_size, noclass])
                gpu_outputs[gpu] = tf.nn.softmax(outputs)
                targets_soft = make_dense(tf.squeeze(gpu_target[gpu], [1]), noclass, 0.1)
                targets_soft = tf.reshape(targets_soft, [-1, noclass])
                targets_hard = make_dense(tf.squeeze(gpu_target[gpu], [1]), noclass, 0.0)
                targets_hard = tf.reshape(targets_hard, [-1, noclass])
                output = tf.transpose(outputs, [1, 0, 2])
                xent_soft = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(output, [-1, noclass]), labels=targets_soft), [batch_size, length])
                xent_hard = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(output, [-1, noclass]), labels=targets_hard), [batch_size, length])
                (low, high) = (0.1 / float(noclass - 1), 0.9)
                const = high * tf.log(high) + float(noclass - 1) * low * tf.log(low)
                weight_sum = tf.reduce_sum(weights) + 1e-20
                true_perp = tf.reduce_sum(xent_hard * weights) / weight_sum
                soft_loss = tf.reduce_sum(xent_soft * weights) / weight_sum
                perp_loss = soft_loss + const
                mem_loss = 0.5 * tf.reduce_mean(mem_loss) / length_float
                total_loss = perp_loss + mem_loss
                gpu_losses[gpu].append(true_perp)
                if backward:
                    data.print_out('Creating backward pass for the model.')
                    grads = tf.gradients(total_loss, tf.trainable_variables(), colocate_gradients_with_ops=True)
                    for (g_i, g) in enumerate(grads):
                        if isinstance(g, tf.IndexedSlices):
                            grads[g_i] = tf.convert_to_tensor(g)
                    (grads, norm) = tf.clip_by_global_norm(grads, max_grad_norm)
                    gpu_grad_norms[gpu].append(norm)
                    for g in grads:
                        if grad_noise_scale > 0.001:
                            g += tf.truncated_normal(tf.shape(g)) * self.noise_param
                    grads_list.append(grads)
                else:
                    gpu_grad_norms[gpu].append(0.0)
                data.print_out('Created model for gpu %d in %.2f s.' % (gpu, time.time() - start_time))
        self.updates = []
        self.after_enc_step = tf.concat(axis=0, values=self.after_enc_step)
        if backward:
            tf.get_variable_scope()._reuse = False
            tf.get_variable_scope().set_caching_device(None)
            grads = [gpu_avg([grads_list[g][i] for g in xrange(num_gpus)]) for i in xrange(len(grads_list[0]))]
            update = adam_update(grads)
            self.updates.append(update)
        else:
            self.updates.append(tf.no_op())
        self.losses = [gpu_avg([gpu_losses[g][i] for g in xrange(num_gpus)]) for i in xrange(len(gpu_losses[0]))]
        self.out_idx = tf.concat(axis=0, values=gpu_out_idx)
        self.grad_norms = [gpu_avg([gpu_grad_norms[g][i] for g in xrange(num_gpus)]) for i in xrange(len(gpu_grad_norms[0]))]
        self.outputs = [tf.concat(axis=1, values=[gpu_outputs[g] for g in xrange(num_gpus)])]
        self.quantize_op = quantize_weights_op(512, 8)
        if backward:
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    def step(self, sess, inp, target, do_backward_in, noise_param=None, beam_size=2, eos_id=2, eos_cost=0.0, update_mem=None, state=None):
        if False:
            i = 10
            return i + 15
        'Run a step of the network.'
        (batch_size, height, length) = (inp.shape[0], inp.shape[1], inp.shape[2])
        do_backward = do_backward_in
        train_mode = True
        if do_backward_in is None:
            do_backward = False
            train_mode = False
        if update_mem is None:
            update_mem = do_backward
        feed_in = {}
        if state is None:
            state = np.zeros([batch_size, length, height, self.nmaps])
        feed_in[self.prev_step.name] = state
        feed_in[self.length_tensor.name] = length
        feed_in[self.noise_param.name] = noise_param if noise_param else 0.0
        feed_in[self.do_training.name] = 1.0 if do_backward else 0.0
        feed_in[self.update_mem.name] = 1 if update_mem else 0
        if do_backward_in is False:
            feed_in[self.sampling.name] = 0.0
        index = 0
        feed_out = []
        if do_backward:
            feed_out.append(self.updates[index])
            feed_out.append(self.grad_norms[index])
        if train_mode:
            feed_out.append(self.losses[index])
        feed_in[self.input.name] = inp
        feed_in[self.target.name] = target
        feed_out.append(self.outputs[index])
        if train_mode:
            res = sess.run([self.after_enc_step] + feed_out, feed_in)
            (after_enc_state, res) = (res[0], res[1:])
        else:
            feed_in[self.sampling.name] = 1.1
            res = sess.run([self.after_enc_step, self.out_idx] + feed_out, feed_in)
            (after_enc_state, out_idx) = (res[0], res[1])
            res = [res[2][l] for l in xrange(length)]
            outputs = [out_idx[:, i] for i in xrange(length)]
            cost = [0.0 for _ in xrange(beam_size * batch_size)]
            seen_eos = [0 for _ in xrange(beam_size * batch_size)]
            for (idx, logit) in enumerate(res):
                best = outputs[idx]
                for b in xrange(batch_size):
                    if seen_eos[b] > 1:
                        cost[b] -= eos_cost
                    else:
                        cost[b] += np.log(logit[b][best[b]])
                    if best[b] in [eos_id]:
                        seen_eos[b] += 1
            res = [[-c for c in cost]] + outputs
        offset = 0
        norm = None
        if do_backward:
            offset = 2
            norm = res[1]
        if train_mode:
            outputs = res[offset + 1]
            outputs = [outputs[l] for l in xrange(length)]
        return (res[offset], outputs, norm, after_enc_state)