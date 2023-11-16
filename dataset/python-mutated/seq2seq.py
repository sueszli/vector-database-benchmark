from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
from six.moves import zip
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import init_ops
import third_party.rnn as rnn
import third_party.rnn_cell as rnn_cell
import GlobalParams
import memoryModule_decoder
linear = rnn_cell._linear

def _extract_argmax_and_embed(embedding, output_projection=None, update_embedding=True):
    if False:
        print('Hello World!')
    'Get a loop_function that extracts the previous symbol and embeds it.\n\n    Args:\n      embedding: embedding tensor for symbols.\n      output_projection: None or a pair (W, B). If provided, each fed previous\n        output will first be multiplied by W and added B.\n      update_embedding: Boolean; if False, the gradients will not propagate\n        through the embeddings.\n\n    Returns:\n      A loop function.\n    '

    def loop_function(prev, _):
        if False:
            while True:
                i = 10
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
        prev_symbol = math_ops.argmax(prev, 1)
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev
    return loop_function

def attention_decoder(decoder_inputs, decoder_weights, initial_state, initial_mem_state, attention_states, cell, encoder_weights, output_size=None, num_heads=1, loop_function=None, memory_weight=1.0, dtype=dtypes.float32, scope=None, initial_state_attention=False, cell_initializer=None):
    if False:
        for i in range(10):
            print('nop')
    'RNN decoder with attention for the sequence-to-sequence model.\n\n    In this context "attention" means that, during decoding, the RNN can look up\n    information in the additional tensor attention_states, and it does this by\n    focusing on a few entries from the tensor. This model has proven to yield\n    especially good results in a number of sequence-to-sequence tasks. This\n    implementation is based on http://arxiv.org/abs/1412.7449 (see below for\n    details). It is recommended for complex sequence-to-sequence tasks.\n\n    Args:\n      decoder_inputs: A list of 2D Tensors [batch_size x input_size].\n      decoder_weights: A list of 1D int32 Tensors of shape [batch_size].\n      initial_state: 2D Tensor [batch_size x cell.state_size].\n      initial_mem_state: the value of the initial memory state.\n      attention_states: 3D Tensor [batch_size x attn_length x attn_size].\n      cell: rnn_cell.RNNCell defining the cell function and size.\n      encoder_weights:A list of 1D int32 Tensors of shape [batch_size].\n      output_size: Size of the output vectors; if None, we use cell.output_size.\n      num_heads: Number of attention heads that read from attention_states.\n      loop_function: If not None, this function will be applied to i-th output\n        in order to generate i+1-th input, and decoder_inputs will be ignored,\n        except for the first element ("GO" symbol). This can be used for decoding,\n        but also for training to emulate http://arxiv.org/abs/1506.03099.\n        Signature -- loop_function(prev, i) = next\n          * prev is a 2D Tensor of shape [batch_size x output_size],\n          * i is an integer, the step number (when advanced control is needed),\n          * next is a 2D Tensor of shape [batch_size x input_size].\n\n      memory_weight: the weight of the memory model.\n      dtype: The dtype to use for the RNN initial state (default: tf.float32).\n      scope: VariableScope for the created subgraph; default: "attention_decoder".\n      initial_state_attention: If False (default), initial attentions are zero.\n        If True, initialize the attentions from the initial state and attention\n        states -- useful when we wish to resume decoding from a previously\n        stored decoder state and attention states.\n      cell_initializer: the initial value of the word embedding.\n\n    Returns:\n      A tuple of the form (outputs, state, state_not_atten), where:\n        outputs: A list of the same length as decoder_inputs of 2D Tensors of\n          shape [batch_size x output_size]. These represent the generated outputs.\n          Output i is computed from input i (which is either the i-th element\n          of decoder_inputs or loop_function(output {i-1}, i)) as follows.\n          First, we run the cell on a combination of the input and previous\n          attention masks:\n            cell_output, new_state = cell(linear(input, prev_attn), prev_state).\n          Then, we calculate new attention masks:\n            new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))\n          and then we calculate the output:\n            output = linear(cell_output, new_attn).\n        state: The state of each decoder cell the final time-step.\n          It is a 2D Tensor of shape [batch_size x cell.state_size].\n        state_not_atten: The state of each paralell decoder cell the final time-step.\n          It is a 2D Tensor of shape [batch_size x cell.state_size].\n\n    Raises:\n      ValueError: when num_heads is not positive, there are no inputs, shapes\n        of attention_states are not set, or input size cannot be inferred\n        from the input.\n    '
    if not decoder_inputs:
        raise ValueError('Must provide at least 1 input to attention decoder.')
    if num_heads < 1:
        raise ValueError('With less than 1 heads, use a non-attention decoder.')
    if not attention_states.get_shape()[1:2].is_fully_defined():
        raise ValueError('Shape[1] and [2] of attention_states must be known: %s' % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size
    with variable_scope.variable_scope(scope or 'attention_decoder'):
        batch_size = array_ops.shape(decoder_inputs[0])[0]
        batch_size_not_tensor = decoder_inputs[0].get_shape()[0].value
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value
        embedding = variable_scope.get_variable('embedding', [4777, 200], initializer=cell_initializer, trainable=False)
        if not os.path.exists('../resource/memory_resource/npy/' + sys.argv[1] + '_' + sys.argv[2] + '_memory.npy'):
            memoryModule_decoder.main()
        memoryTemp = np.load('../resource/memory_resource/npy/' + sys.argv[1] + '_' + sys.argv[2] + '_memory.npy')
        memoryWordVectorTemp = np.load('../resource/memory_resource/npy/' + sys.argv[1] + '_' + sys.argv[2] + '_memoryWordVector.npy')
        memory = variable_scope.get_variable('memory', memoryTemp.shape, initializer=init_ops.constant_initializer(memoryTemp), trainable=False)
        memoryWordVector = variable_scope.get_variable('memoryWordVector', memoryWordVectorTemp.shape, initializer=init_ops.constant_initializer(memoryWordVectorTemp), trainable=False)
        memory_length = memory.get_shape()[1].value
        hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = 500
        for a in xrange(num_heads):
            k = variable_scope.get_variable('AttnW_%d' % a, [1, 1, attn_size, attention_vec_size], initializer=init_ops.constant_initializer(GlobalParams.params['U_A'].reshape(1, 1, attn_size, attention_vec_size)))
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], 'SAME'))
            v.append(variable_scope.get_variable('AttnV_%d' % a, [attention_vec_size], initializer=init_ops.constant_initializer(GlobalParams.params['V_A'].T[0])))
        state = initial_state
        attens = []

        def attention(query):
            if False:
                for i in range(10):
                    print('nop')
            'Put attention masks on hidden using hidden_features and query.'
            ds = []
            for a in xrange(num_heads):
                with variable_scope.variable_scope('Attention_%d' % a):
                    y = linear(query, attention_vec_size, False, 'atten_hidden_W')
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    atten_weights = array_ops.transpose(encoder_weights)
                    s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                    s_softmax_weights = math_ops.exp(s) * atten_weights
                    a = s_softmax_weights / math_ops.reduce_sum(s_softmax_weights, 1, keep_dims=True)
                    d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds
        outputs = []
        state_outputs = []
        prev = None
        batch_attn_size = array_ops.pack([batch_size, attn_size])
        attns = [array_ops.zeros(batch_attn_size, dtype=dtype) for _ in xrange(num_heads)]
        for a in attns:
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state[1])
        for (i, inp) in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope('loop_function', reuse=True):
                    inp = loop_function(prev, i)
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError('Could not infer input size from input: %s' % inp.name)
            atten_weight_zero = array_ops.zeros(array_ops.pack([batch_size_not_tensor, attn_size]), dtype=dtype)
            (cell_output_not_atten, state_not_atten) = cell(array_ops.concat(1, [inp, atten_weight_zero]), initial_mem_state)
            with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
                (cell_output, state) = cell(array_ops.concat(1, [inp, attns[0]]), state)
            cell_output_reshape = array_ops.reshape(cell_output_not_atten, [batch_size_not_tensor, 1, cell.output_size])
            norm_l2 = tf.sqrt(tf.reduce_sum(tf.square(memory), reduction_indices=2, keep_dims=True)) * tf.sqrt(tf.reduce_sum(tf.square(cell_output_reshape), reduction_indices=2, keep_dims=True))
            cos_dist = tf.reduce_sum(cell_output_reshape * memory, reduction_indices=2, keep_dims=True) / norm_l2
            normalized_cos_dist = 0.5 + 0.5 * cos_dist
            memory_hidden = tf.matmul(array_ops.reshape(normalized_cos_dist / tf.reduce_sum(normalized_cos_dist, reduction_indices=1, keep_dims=True), [batch_size_not_tensor, memory_length]), memoryWordVector)
            maxsize = 500
            with variable_scope.variable_scope('U_O'):
                u0 = linear([cell_output], maxsize, False, 'U_O')
            with variable_scope.variable_scope('V_O'):
                v0 = linear([inp], maxsize, False, 'V_O')
            with variable_scope.variable_scope('C_O'):
                c0 = linear([attns[0]], maxsize, False, 'C_O')
            y_temp = u0 + v0 + c0
            y_temp_mask = array_ops.transpose(array_ops.pack([decoder_weights[i] for i_mask in range(maxsize)]))
            y_temp_T = array_ops.transpose(y_temp)
            t1 = array_ops.gather(y_temp_T, range(0, y_temp_T.get_shape()[0], 2))
            t2 = array_ops.gather(y_temp_T, range(1, y_temp_T.get_shape()[0], 2))
            max_out = math_ops.maximum(t1, t2)
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
                    attns = attention(state[1])
            else:
                attns = attention(state[1])
            with variable_scope.variable_scope('AttnOutputProjection'):
                memory_output = tf.matmul(memory_hidden, embedding, transpose_b=True)
                atten_output = linear([array_ops.transpose(max_out)], output_size, True, 'W_O')
                output = memory_output * memory_weight + atten_output
            if loop_function is not None:
                prev = output
            if i == 1:
                attens.append(nn_ops.softmax(memory_output))
                attens.append(nn_ops.softmax(atten_output))
            if i == 0:
                outputs.append(output)
            else:
                outputs.append(output)
    return (outputs[0], array_ops.pack(state), array_ops.pack(state_not_atten))

def embedding_attention_decoder(decoder_inputs, decoder_weights, initial_state, initial_mem_state, attention_states, cell, encoder_weights, num_symbols, embedding_size, num_heads=1, output_size=None, output_projection=None, memory_weight=1.0, feed_previous=False, update_embedding_for_previous=True, dtype=dtypes.float32, scope=None, initial_state_attention=False, cell_initializer=None):
    if False:
        for i in range(10):
            print('nop')
    'RNN decoder with embedding and attention and a pure-decoding option.\n\n    Args:\n      decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).\n      decoder_weights: A list of 1D int32 Tensors of shape [batch_size].\n      initial_state: 2D Tensor [batch_size x cell.state_size].\n      initial_mem_state: the value of the initial memory state.\n      attention_states: 3D Tensor [batch_size x attn_length x attn_size].\n      cell: rnn_cell.RNNCell defining the cell function.\n      encoder_weights: the weights of the encoder_input.\n      num_symbols: Integer, how many symbols come into the embedding.\n      embedding_size: Integer, the length of the embedding vector for each symbol.\n      num_heads: Number of attention heads that read from attention_states.\n      output_size: Size of the output vectors; if None, use output_size.\n      output_projection: None or a pair (W, B) of output projection weights and\n        biases; W has shape [output_size x num_symbols] and B has shape\n        [num_symbols]; if provided and feed_previous=True, each fed previous\n        output will first be multiplied by W and added B.\n      memory_weight: the weight of the memory model.\n      feed_previous: Boolean; if True, only the first of decoder_inputs will be\n        used (the "GO" symbol), and all other decoder inputs will be generated by:\n          next = embedding_lookup(embedding, argmax(previous_output)),\n        In effect, this implements a greedy decoder. It can also be used\n        during training to emulate http://arxiv.org/abs/1506.03099.\n        If False, decoder_inputs are used as given (the standard decoder case).\n      update_embedding_for_previous: Boolean; if False and feed_previous=True,\n        only the embedding for the first symbol of decoder_inputs (the "GO"\n        symbol) will be updated by back propagation. Embeddings for the symbols\n        generated from the decoder itself remain unchanged. This parameter has\n        no effect if feed_previous=False.\n      dtype: The dtype to use for the RNN initial states (default: tf.float32).\n      scope: VariableScope for the created subgraph; defaults to\n        "embedding_attention_decoder".\n      initial_state_attention: If False (default), initial attentions are zero.\n        If True, initialize the attentions from the initial state and attention\n        states -- useful when we wish to resume decoding from a previously\n        stored decoder state and attention states.\n      cell_initializer: the initial value of the word embedding.\n\n    Returns:\n      A tuple of the form (outputs, state, state_not_atten), where:\n        outputs: A list of the same length as decoder_inputs of 2D Tensors with\n          shape [batch_size x output_size] containing the generated outputs.\n        state: The state of each decoder cell at the final time-step.\n          It is a 2D Tensor of shape [batch_size x cell.state_size].\n        state_not_atten: The state of each paralell decoder cell the final time-step.\n          It is a 2D Tensor of shape [batch_size x cell.state_size].\n\n    Raises:\n      ValueError: When output_projection has the wrong shape.\n    '
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])
    with variable_scope.variable_scope(scope or 'embedding_attention_decoder'):
        embedding = variable_scope.get_variable('embedding', [num_symbols, embedding_size], initializer=cell_initializer, trainable=False)
        loop_function = _extract_argmax_and_embed(embedding, output_projection, update_embedding_for_previous) if feed_previous else None
        emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
        return attention_decoder(emb_inp, decoder_weights, initial_state, initial_mem_state, attention_states, cell, encoder_weights, output_size=output_size, num_heads=num_heads, loop_function=loop_function, memory_weight=memory_weight, initial_state_attention=initial_state_attention, cell_initializer=cell_initializer)

def embedding_attention_seq2seq(encoder_inputs, reverse_encoder_inputs, decoder_inputs, cell, encoder_weights, decoder_weights, num_encoder_symbols, num_decoder_symbols, embedding_size, num_heads=1, sig_weight=None, output_projection=None, sequence_length=None, output_keep_prob=1.0, memory_weight=1.0, feed_previous=False, dtype=dtypes.float32, scope=None, initial_state_attention=False, cell_initializer=None):
    if False:
        return 10
    'Embedding sequence-to-sequence model with attention.\n\n    This model first embeds encoder_inputs by a newly created embedding (of shape\n    [num_encoder_symbols x input_size]). Then it runs an RNN to encode\n    embedded encoder_inputs into a state vector. It keeps the outputs of this\n    RNN at every step to use for attention later. Next, it embeds decoder_inputs\n    by another newly created embedding (of shape [num_decoder_symbols x\n    input_size]). Then it runs attention decoder, initialized with the last\n    encoder state, on embedded decoder_inputs and attending to encoder outputs.\n\n    Args:\n      encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].\n      reverse_encoder_inputs: the reverse of encoder_inputs.\n      decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].\n      cell: rnn_cell.RNNCell defining the cell function and size.\n      encoder_weights: the weights of the encoder_input.\n      decoder_weights: A list of 1D int32 Tensors of shape [batch_size].\n      num_encoder_symbols: Integer; number of symbols on the encoder side.\n      num_decoder_symbols: Integer; number of symbols on the decoder side.\n      embedding_size: Integer, the length of the embedding vector for each symbol.\n      num_heads: Number of attention heads that read from attention_states.\n      output_projection: None or a pair (W, B) of output projection weights and\n        biases; W has shape [output_size x num_decoder_symbols] and B has\n        shape [num_decoder_symbols]; if provided and feed_previous=True, each\n        fed previous output will first be multiplied by W and added B.\n      feed_previous: Boolean or scalar Boolean Tensor; if True, only the first\n        of decoder_inputs will be used (the "GO" symbol), and all other decoder\n        inputs will be taken from previous outputs (as in embedding_rnn_decoder).\n        If False, decoder_inputs are used as given (the standard decoder case).\n      dtype: The dtype of the initial RNN state (default: tf.float32).\n      scope: VariableScope for the created subgraph; defaults to\n        "embedding_attention_seq2seq".\n      initial_state_attention: If False (default), initial attentions are zero.\n        If True, initialize the attentions from the initial state and attention\n        states.\n      cell_initializer: the initial value of the word embedding.\n\n    Returns:\n      A tuple of the form (outputs, state,state_not_atten), where:\n        outputs: A list of the same length as decoder_inputs of 2D Tensors with\n          shape [batch_size x num_decoder_symbols] containing the generated\n          outputs.\n        state: The state of each decoder cell at the final time-step.\n          It is a 2D Tensor of shape [batch_size x cell.state_size].\n                state_not_atten: The state of each paralell decoder cell the final time-step.\n          It is a 2D Tensor of shape [batch_size x cell.state_size].\n    '
    with variable_scope.variable_scope(scope or 'embedding_attention_seq2seq'):
        embedding = variable_scope.get_variable('embedding', [num_encoder_symbols, embedding_size], initializer=cell_initializer, trainable=False)
        embed_encoder_inputs = [embedding_ops.embedding_lookup(embedding, i) * array_ops.transpose(array_ops.pack([encoder_weights[index] for i in range(embedding_size)])) for (index, i) in enumerate(encoder_inputs)]
        atten_inputs = array_ops.concat(1, [array_ops.reshape(e, [-1, 1, embedding_size]) for e in embed_encoder_inputs])
        attn_length = atten_inputs.get_shape()[1].value
        attn_size = atten_inputs.get_shape()[2].value
        reshape_atten_inputs = array_ops.reshape(atten_inputs, [-1, attn_length, 1, attn_size])
        attention_vec_size = cell.output_size
        atten_inputs_W = variable_scope.get_variable('atten_inputs_W', [1, 1, attn_size, attention_vec_size], initializer=init_ops.constant_initializer(GlobalParams.params['lstm_W_i'].reshape(1, 1, attn_size, attention_vec_size)))
        atten_inputs_hidden = nn_ops.dropout(array_ops.reshape(nn_ops.conv2d(reshape_atten_inputs, atten_inputs_W, [1, 1, 1, 1], 'SAME'), [-1, attn_length, attention_vec_size]), output_keep_prob)
        drop_out_cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=output_keep_prob)
        (outputs, state_fw, state_bw, reverse_embed_encoder_inputs, tmp) = rnn.bidirectional_rnn(drop_out_cell, drop_out_cell, embed_encoder_inputs, sequence_length=sequence_length, dtype=dtype)
        if sig_weight != None:
            W_c_0 = variable_scope.get_variable('W_c_0', [cell.output_size + embedding_size, cell.output_size], initializer=init_ops.constant_initializer(GlobalParams.params['C_lstm_end_once']))
            concat_c_sig = array_ops.concat(1, [state_fw[1], sig_weight])
            W_h_0 = variable_scope.get_variable('W_h_0', [cell.output_size + embedding_size, cell.output_size], initializer=init_ops.constant_initializer(GlobalParams.params['W_S']))
            concat_h_sig = array_ops.concat(1, [state_bw[1], sig_weight])
            initial_state = rnn_cell.LSTMStateTuple(tanh(math_ops.matmul(concat_c_sig, W_c_0)), tanh(math_ops.matmul(concat_h_sig, W_h_0)))
        top_states = [array_ops.reshape(e, [-1, 1, 2 * cell.output_size]) for e in outputs]
        attention_states = array_ops.concat(2, [array_ops.concat(1, top_states), atten_inputs_hidden])
        output_size = None
        if output_projection is None:
            output_size = num_decoder_symbols
        if isinstance(feed_previous, bool):
            return embedding_attention_decoder(decoder_inputs, initial_state, attention_states, drop_out_cell, encoder_weights, num_decoder_symbols, embedding_size, num_heads=num_heads, output_size=output_size, output_projection=output_projection, feed_previous=feed_previous, initial_state_attention=True, cell_initializer=cell_initializer)

        def decoder(feed_previous_bool):
            if False:
                i = 10
                return i + 15
            reuse = None
            with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=reuse):
                return [None] + [None] + [None]
        outputs_and_state = decoder(False)
        return (outputs_and_state[:-2], outputs_and_state[-2], attention_states, atten_inputs_hidden, array_ops.pack(initial_state), array_ops.pack(embed_encoder_inputs), state_fw[1], state_bw[1], array_ops.pack(reverse_embed_encoder_inputs), array_ops.pack(tmp))

def predict_decoder(feed_previous_bool, decoder_inputs, decoder_weights, initial_state, initial_mem_state, attention_states, drop_out_cell, encoder_weights, num_decoder_symbols, embedding_size, memory_weight, cell_initializer):
    if False:
        print('Hello World!')
    '\n      Accelerating the decoder.\n\n      Args:\n        feed_previous_bool: if True, only the first of decoder_inputs will be used (the "GO" symbol), and all other decoder\n        inputs will be taken from previous outputs (as in embedding_rnn_decoder).\n        If False, decoder_inputs are used as given (the standard decoder case).\n        decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].\n        decoder_weights: A list of 1D int32 Tensors of shape [batch_size].\n        initial_state: the initial of the state.\n        initial_mem_state: the initial of the memory state.\n        attention_states: the current attention state.\n        cell: rnn_cell.RNNCell defining the cell function and size.\n        encoder_weights: the weight of encoder inputs.\n        num_decoder_symbols: the number of decoder symbols.\n        embedding_size: the size of the word embedding.\n        memory_weight: the weight of memory model.\n        cell_initializer: the initial value of the word embedding.\n\n      Returns:\n        A tuple of the form (outputs,state,mem_state), where:\n          outputs: A list of the same length as decoder_inputs of 2D Tensors with\n            shape [batch_size x num_decoder_symbols] containing the generated\n            outputs.\n          state: The state of each decoder cell at the final time-step.\n            It is a 2D Tensor of shape [batch_size x cell.state_size].\n          mem_state: The state of each paralell decoder cell the final time-step.\n            It is a 2D Tensor of shape [batch_size x cell.state_size].\n      '

    def spAndRe(tensor):
        if False:
            for i in range(10):
                print('nop')
        return [array_ops.reshape(t, [1, 500]) for t in array_ops.split(0, 2, tensor)]
    initial_state = rnn_cell.LSTMStateTuple(spAndRe(initial_state)[0], spAndRe(initial_state)[1])
    initial_mem_state = rnn_cell.LSTMStateTuple(spAndRe(initial_mem_state)[0], spAndRe(initial_mem_state)[1])
    reuse = None
    with variable_scope.variable_scope('embedding_attention_seq2seq', reuse=reuse):
        (outputs, state, mem_state) = embedding_attention_decoder(decoder_inputs, decoder_weights, initial_state, initial_mem_state, attention_states, drop_out_cell, encoder_weights, num_decoder_symbols, embedding_size, num_heads=1, output_size=num_decoder_symbols, output_projection=None, memory_weight=memory_weight, feed_previous=feed_previous_bool, update_embedding_for_previous=False, initial_state_attention=True, cell_initializer=cell_initializer)
        return (outputs, state, mem_state)

def sequence_loss_by_example(logits, targets, weights, average_across_timesteps=True, softmax_loss_function=None, name=None):
    if False:
        return 10
    'Weighted cross-entropy loss for a sequence of logits (per example).\n\n    Args:\n      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].\n      targets: List of 1D batch-sized int32 Tensors of the same length as logits.\n      weights: List of 1D batch-sized float-Tensors of the same length as logits.\n      average_across_timesteps: If set, divide the returned cost by the total\n        label weight.\n      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch\n        to be used instead of the standard softmax (the default if this is None).\n      name: Optional name for this operation, default: "sequence_loss_by_example".\n\n    Returns:\n      1D batch-sized float Tensor: The log-perplexity for each sequence.\n\n    Raises:\n      ValueError: If len(logits) is different from len(targets) or len(weights).\n    '
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError('Lengths of logits, weights, and targets must be the same %d, %d, %d.' % (len(logits), len(weights), len(targets)))
    with ops.op_scope(logits + targets + weights, name, 'sequence_loss_by_example'):
        log_perp_list = []
        for (logit, target, weight) in zip(logits, targets, weights):
            if softmax_loss_function is None:
                target = array_ops.reshape(target, [-1])
                crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(logit, target)
            else:
                crossent = softmax_loss_function(logit, target)
            log_perp_list.append(crossent * weight)
        log_perps = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            total_size += 1e-12
            log_perps /= total_size
    return log_perps

def sequence_loss(logits, targets, weights, average_across_timesteps=False, average_across_batch=True, softmax_loss_function=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Weighted cross-entropy loss for a sequence of logits, batch-collapsed.\n\n    Args:\n      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].\n      targets: List of 1D batch-sized int32 Tensors of the same length as logits.\n      weights: List of 1D batch-sized float-Tensors of the same length as logits.\n      average_across_timesteps: If set, divide the returned cost by the total\n        label weight.\n      average_across_batch: If set, divide the returned cost by the batch size.\n      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch\n        to be used instead of the standard softmax (the default if this is None).\n      name: Optional name for this operation, defaults to "sequence_loss".\n\n    Returns:\n      A scalar float Tensor: The average log-perplexity per symbol (weighted).\n\n    Raises:\n      ValueError: If len(logits) is different from len(targets) or len(weights).\n    '
    with ops.op_scope(logits + targets + weights, name, 'sequence_loss'):
        cost = math_ops.reduce_sum(sequence_loss_by_example(logits, targets, weights, average_across_timesteps=average_across_timesteps, softmax_loss_function=softmax_loss_function))
        if average_across_batch:
            batch_size = array_ops.shape(targets[0])[0]
            return cost / math_ops.cast(batch_size, dtypes.float32)
        else:
            return cost

def model_with_buckets(encoder_inputs, reverse_encoder_inputs, decoder_inputs, targets, encoder_weights, weights, buckets, keep_prob, is_feed, memory_weight, seq2seq, softmax_loss_function=None, batch_size=None, per_example_loss=False, name=None, sequence_length=None, sig_weight=None):
    if False:
        for i in range(10):
            print('nop')
    'Create a sequence-to-sequence model with support for bucketing.\n\n    The seq2seq argument is a function that defines a sequence-to-sequence model,\n    e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))\n\n    Args:\n      encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.\n      reverse_encoder_inputs: the reverse of encoder_inputs.\n      decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.\n      targets: A list of 1D batch-sized int32 Tensors (desired output sequence).\n      encoder_weights: the weight of encoder inputs.\n      weights: List of 1D batch-sized float-Tensors to weight the targets.\n      buckets: A list of pairs of (input size, output size) for each bucket.\n      keep_prob: the dropout rate.\n      is_feed: if True, only the first of decoder_inputs will be used (the "GO" symbol), and all other decoder\n        inputs will be taken from previous outputs (as in embedding_rnn_decoder).\n        If False, decoder_inputs are used as given (the standard decoder case).\n      memory_weight: the weight of the memory model.\n      seq2seq: A sequence-to-sequence model function; it takes 2 input that\n        agree with encoder_inputs and decoder_inputs, and returns a pair\n        consisting of outputs and states (as, e.g., basic_rnn_seq2seq).\n      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch\n        to be used instead of the standard softmax (the default if this is None).\n      per_example_loss: Boolean. If set, the returned loss will be a batch-sized\n        tensor of losses for each sequence in the batch. If unset, it will be\n        a scalar with the averaged loss from all examples.\n      name: Optional name for this operation, defaults to "model_with_buckets".\n      sequence_length: the length of sequence.\n      sig_weight: the weight of signature.\n\n    Returns:\n      A tuple of the form (outputs, losses), where:\n        outputs: The outputs for each bucket. Its j\'th element consists of a list\n          of 2D Tensors of shape [batch_size x num_decoder_symbols] (jth outputs).\n        losses: List of scalar Tensors, representing losses for each bucket, or,\n          if per_example_loss is set, a list of 1D batch-sized float Tensors.\n\n    Raises:\n      ValueError: If length of encoder_inputsut, targets, or weights is smaller\n        than the largest (last) bucket.\n    '
    if len(encoder_inputs) < buckets[-1][0]:
        raise ValueError('Length of encoder_inputs (%d) must be at least that of last bucket (%d).' % (len(encoder_inputs), buckets[-1][0]))
    if len(targets) < buckets[-1][1]:
        raise ValueError('Length of targets (%d) must be at least that of lastbucket (%d).' % (len(targets), buckets[-1][1]))
    if len(weights) < buckets[-1][1]:
        raise ValueError('Length of weights (%d) must be at least that of lastbucket (%d).' % (len(weights), buckets[-1][1]))
    all_inputs = encoder_inputs + reverse_encoder_inputs + decoder_inputs + targets + weights + encoder_weights
    all_inputs.append(sequence_length)
    all_inputs.append(keep_prob)
    all_inputs.append(is_feed)
    losses = []
    outputs = []
    state_outputs = []
    attens = []
    atten_inputs = []
    hiddens = []
    embed_inputs = []
    state_fw = []
    state_bw = []
    reverse_embed_encoder_inputs = []
    tmp = []
    with ops.op_scope(all_inputs, name, 'model_with_buckets'):
        for (j, bucket) in enumerate(buckets):
            with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
                (bucket_outputs, bucket_state_outputs, bucket_attens, bucket_atten_inputs, bucket_hiddens, bucket_embed_inputs, bucket_state_fw, bucket_state_bw, bucket_reverse_embed_encoder_inputs, bucket_tmp) = seq2seq(encoder_inputs[:bucket[0]], reverse_encoder_inputs[:bucket[0]], decoder_inputs[:bucket[1] - 1], sequence_length, encoder_weights[:bucket[0]], weights[:bucket[1] - 1], keep_prob, sig_weight, is_feed, memory_weight)
                outputs.append(bucket_outputs)
                state_outputs.append(bucket_state_outputs)
                attens.append(bucket_attens)
                atten_inputs.append(bucket_atten_inputs)
                hiddens.append(bucket_hiddens)
                embed_inputs.append(bucket_embed_inputs)
                state_fw.append(bucket_state_fw)
                state_bw.append(bucket_state_bw)
                reverse_embed_encoder_inputs.append(bucket_reverse_embed_encoder_inputs)
                tmp.append(bucket_tmp)
    return (outputs, None, attens, state_outputs, atten_inputs, hiddens, embed_inputs, state_fw, state_bw, reverse_embed_encoder_inputs, tmp)