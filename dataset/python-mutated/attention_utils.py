"""Attention-based decoder functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import function
__all__ = ['prepare_attention', 'attention_decoder_fn_train', 'attention_decoder_fn_inference']

def attention_decoder_fn_train(encoder_state, attention_keys, attention_values, attention_score_fn, attention_construct_fn, name=None):
    if False:
        return 10
    'Attentional decoder function for `dynamic_rnn_decoder` during training.\n\n  The `attention_decoder_fn_train` is a training function for an\n  attention-based sequence-to-sequence model. It should be used when\n  `dynamic_rnn_decoder` is in the training mode.\n\n  The `attention_decoder_fn_train` is called with a set of the user arguments\n  and returns the `decoder_fn`, which can be passed to the\n  `dynamic_rnn_decoder`, such that\n\n  ```\n  dynamic_fn_train = attention_decoder_fn_train(encoder_state)\n  outputs_train, state_train = dynamic_rnn_decoder(\n      decoder_fn=dynamic_fn_train, ...)\n  ```\n\n  Further usage can be found in the `kernel_tests/seq2seq_test.py`.\n\n  Args:\n    encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.\n    attention_keys: to be compared with target states.\n    attention_values: to be used to construct context vectors.\n    attention_score_fn: to compute similarity between key and target states.\n    attention_construct_fn: to build attention states.\n    name: (default: `None`) NameScope for the decoder function;\n      defaults to "simple_decoder_fn_train"\n\n  Returns:\n    A decoder function with the required interface of `dynamic_rnn_decoder`\n    intended for training.\n  '
    with tf.name_scope(name, 'attention_decoder_fn_train', [encoder_state, attention_keys, attention_values, attention_score_fn, attention_construct_fn]):
        pass

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        if False:
            return 10
        'Decoder function used in the `dynamic_rnn_decoder` for training.\n\n    Args:\n      time: positive integer constant reflecting the current timestep.\n      cell_state: state of RNNCell.\n      cell_input: input provided by `dynamic_rnn_decoder`.\n      cell_output: output of RNNCell.\n      context_state: context state provided by `dynamic_rnn_decoder`.\n\n    Returns:\n      A tuple (done, next state, next input, emit output, next context state)\n      where:\n\n      done: `None`, which is used by the `dynamic_rnn_decoder` to indicate\n      that `sequence_lengths` in `dynamic_rnn_decoder` should be used.\n\n      next state: `cell_state`, this decoder function does not modify the\n      given state.\n\n      next input: `cell_input`, this decoder function does not modify the\n      given input. The input could be modified when applying e.g. attention.\n\n      emit output: `cell_output`, this decoder function does not modify the\n      given output.\n\n      next context state: `context_state`, this decoder function does not\n      modify the given context state. The context state could be modified when\n      applying e.g. beam search.\n    '
        with tf.name_scope(name, 'attention_decoder_fn_train', [time, cell_state, cell_input, cell_output, context_state]):
            if cell_state is None:
                cell_state = encoder_state
                attention = _init_attention(encoder_state)
            else:
                attention = attention_construct_fn(cell_output, attention_keys, attention_values)
                cell_output = attention
            next_input = tf.concat([cell_input, attention], 1)
            return (None, cell_state, next_input, cell_output, context_state)
    return decoder_fn

def attention_decoder_fn_inference(output_fn, encoder_state, attention_keys, attention_values, attention_score_fn, attention_construct_fn, embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, num_decoder_symbols, dtype=tf.int32, name=None):
    if False:
        return 10
    'Attentional decoder function for `dynamic_rnn_decoder` during inference.\n\n  The `attention_decoder_fn_inference` is a simple inference function for a\n  sequence-to-sequence model. It should be used when `dynamic_rnn_decoder` is\n  in the inference mode.\n\n  The `attention_decoder_fn_inference` is called with user arguments\n  and returns the `decoder_fn`, which can be passed to the\n  `dynamic_rnn_decoder`, such that\n\n  ```\n  dynamic_fn_inference = attention_decoder_fn_inference(...)\n  outputs_inference, state_inference = dynamic_rnn_decoder(\n      decoder_fn=dynamic_fn_inference, ...)\n  ```\n\n  Further usage can be found in the `kernel_tests/seq2seq_test.py`.\n\n  Args:\n    output_fn: An output function to project your `cell_output` onto class\n    logits.\n\n    An example of an output function;\n\n    ```\n      tf.variable_scope("decoder") as varscope\n        output_fn = lambda x: tf.contrib.layers.linear(x, num_decoder_symbols,\n                                            scope=varscope)\n\n        outputs_train, state_train = seq2seq.dynamic_rnn_decoder(...)\n        logits_train = output_fn(outputs_train)\n\n        varscope.reuse_variables()\n        logits_inference, state_inference = seq2seq.dynamic_rnn_decoder(\n            output_fn=output_fn, ...)\n    ```\n\n    If `None` is supplied it will act as an identity function, which\n    might be wanted when using the RNNCell `OutputProjectionWrapper`.\n\n    encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.\n    attention_keys: to be compared with target states.\n    attention_values: to be used to construct context vectors.\n    attention_score_fn: to compute similarity between key and target states.\n    attention_construct_fn: to build attention states.\n    embeddings: The embeddings matrix used for the decoder sized\n    `[num_decoder_symbols, embedding_size]`.\n    start_of_sequence_id: The start of sequence ID in the decoder embeddings.\n    end_of_sequence_id: The end of sequence ID in the decoder embeddings.\n    maximum_length: The maximum allowed of time steps to decode.\n    num_decoder_symbols: The number of classes to decode at each time step.\n    dtype: (default: `tf.int32`) The default data type to use when\n    handling integer objects.\n    name: (default: `None`) NameScope for the decoder function;\n      defaults to "attention_decoder_fn_inference"\n\n  Returns:\n    A decoder function with the required interface of `dynamic_rnn_decoder`\n    intended for inference.\n  '
    with tf.name_scope(name, 'attention_decoder_fn_inference', [output_fn, encoder_state, attention_keys, attention_values, attention_score_fn, attention_construct_fn, embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, num_decoder_symbols, dtype]):
        start_of_sequence_id = tf.convert_to_tensor(start_of_sequence_id, dtype)
        end_of_sequence_id = tf.convert_to_tensor(end_of_sequence_id, dtype)
        maximum_length = tf.convert_to_tensor(maximum_length, dtype)
        num_decoder_symbols = tf.convert_to_tensor(num_decoder_symbols, dtype)
        encoder_info = tf.contrib.framework.nest.flatten(encoder_state)[0]
        batch_size = encoder_info.get_shape()[0].value
        if output_fn is None:
            output_fn = lambda x: x
        if batch_size is None:
            batch_size = tf.shape(encoder_info)[0]

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        if False:
            i = 10
            return i + 15
        'Decoder function used in the `dynamic_rnn_decoder` for inference.\n\n    The main difference between this decoder function and the `decoder_fn` in\n    `attention_decoder_fn_train` is how `next_cell_input` is calculated. In\n    decoder function we calculate the next input by applying an argmax across\n    the feature dimension of the output from the decoder. This is a\n    greedy-search approach. (Bahdanau et al., 2014) & (Sutskever et al., 2014)\n    use beam-search instead.\n\n    Args:\n      time: positive integer constant reflecting the current timestep.\n      cell_state: state of RNNCell.\n      cell_input: input provided by `dynamic_rnn_decoder`.\n      cell_output: output of RNNCell.\n      context_state: context state provided by `dynamic_rnn_decoder`.\n\n    Returns:\n      A tuple (done, next state, next input, emit output, next context state)\n      where:\n\n      done: A boolean vector to indicate which sentences has reached a\n      `end_of_sequence_id`. This is used for early stopping by the\n      `dynamic_rnn_decoder`. When `time>=maximum_length` a boolean vector with\n      all elements as `true` is returned.\n\n      next state: `cell_state`, this decoder function does not modify the\n      given state.\n\n      next input: The embedding from argmax of the `cell_output` is used as\n      `next_input`.\n\n      emit output: If `output_fn is None` the supplied `cell_output` is\n      returned, else the `output_fn` is used to update the `cell_output`\n      before calculating `next_input` and returning `cell_output`.\n\n      next context state: `context_state`, this decoder function does not\n      modify the given context state. The context state could be modified when\n      applying e.g. beam search.\n\n    Raises:\n      ValueError: if cell_input is not None.\n\n    '
        with tf.name_scope(name, 'attention_decoder_fn_inference', [time, cell_state, cell_input, cell_output, context_state]):
            if cell_input is not None:
                raise ValueError('Expected cell_input to be None, but saw: %s' % cell_input)
            if cell_output is None:
                next_input_id = tf.ones([batch_size], dtype=dtype) * start_of_sequence_id
                done = tf.zeros([batch_size], dtype=tf.bool)
                cell_state = encoder_state
                cell_output = tf.zeros([num_decoder_symbols], dtype=tf.float32)
                cell_input = tf.gather(embeddings, next_input_id)
                attention = _init_attention(encoder_state)
            else:
                attention = attention_construct_fn(cell_output, attention_keys, attention_values)
                cell_output = attention
                cell_output = output_fn(cell_output)
                next_input_id = tf.cast(tf.argmax(cell_output, 1), dtype=dtype)
                done = tf.equal(next_input_id, end_of_sequence_id)
                cell_input = tf.gather(embeddings, next_input_id)
            next_input = tf.concat([cell_input, attention], 1)
            done = tf.cond(tf.greater(time, maximum_length), lambda : tf.ones([batch_size], dtype=tf.bool), lambda : done)
            return (done, cell_state, next_input, cell_output, context_state)
    return decoder_fn

def prepare_attention(attention_states, attention_option, num_units, reuse=None):
    if False:
        while True:
            i = 10
    'Prepare keys/values/functions for attention.\n\n  Args:\n    attention_states: hidden states to attend over.\n    attention_option: how to compute attention, either "luong" or "bahdanau".\n    num_units: hidden state dimension.\n    reuse: whether to reuse variable scope.\n\n  Returns:\n    attention_keys: to be compared with target states.\n    attention_values: to be used to construct context vectors.\n    attention_score_fn: to compute similarity between key and target states.\n    attention_construct_fn: to build attention states.\n  '
    with tf.variable_scope('attention_keys', reuse=reuse) as scope:
        attention_keys = tf.contrib.layers.linear(attention_states, num_units, biases_initializer=None, scope=scope)
    attention_values = attention_states
    attention_score_fn = _create_attention_score_fn('attention_score', num_units, attention_option, reuse)
    attention_construct_fn = _create_attention_construct_fn('attention_construct', num_units, attention_score_fn, reuse)
    return (attention_keys, attention_values, attention_score_fn, attention_construct_fn)

def _init_attention(encoder_state):
    if False:
        i = 10
        return i + 15
    'Initialize attention. Handling both LSTM and GRU.\n\n  Args:\n    encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.\n\n  Returns:\n    attn: initial zero attention vector.\n  '
    if isinstance(encoder_state, tuple):
        top_state = encoder_state[-1]
    else:
        top_state = encoder_state
    if isinstance(top_state, tf.contrib.rnn.LSTMStateTuple):
        attn = tf.zeros_like(top_state.h)
    else:
        attn = tf.zeros_like(top_state)
    return attn

def _create_attention_construct_fn(name, num_units, attention_score_fn, reuse):
    if False:
        for i in range(10):
            print('nop')
    'Function to compute attention vectors.\n\n  Args:\n    name: to label variables.\n    num_units: hidden state dimension.\n    attention_score_fn: to compute similarity between key and target states.\n    reuse: whether to reuse variable scope.\n\n  Returns:\n    attention_construct_fn: to build attention states.\n  '

    def construct_fn(attention_query, attention_keys, attention_values):
        if False:
            for i in range(10):
                print('nop')
        with tf.variable_scope(name, reuse=reuse) as scope:
            context = attention_score_fn(attention_query, attention_keys, attention_values)
            concat_input = tf.concat([attention_query, context], 1)
            attention = tf.contrib.layers.linear(concat_input, num_units, biases_initializer=None, scope=scope)
            return attention
    return construct_fn

@function.Defun(func_name='attn_add_fun', noinline=True)
def _attn_add_fun(v, keys, query):
    if False:
        for i in range(10):
            print('nop')
    return tf.reduce_sum(v * tf.tanh(keys + query), [2])

@function.Defun(func_name='attn_mul_fun', noinline=True)
def _attn_mul_fun(keys, query):
    if False:
        print('Hello World!')
    return tf.reduce_sum(keys * query, [2])

def _create_attention_score_fn(name, num_units, attention_option, reuse, dtype=tf.float32):
    if False:
        return 10
    'Different ways to compute attention scores.\n\n  Args:\n    name: to label variables.\n    num_units: hidden state dimension.\n    attention_option: how to compute attention, either "luong" or "bahdanau".\n      "bahdanau": additive (Bahdanau et al., ICLR\'2015)\n      "luong": multiplicative (Luong et al., EMNLP\'2015)\n    reuse: whether to reuse variable scope.\n    dtype: (default: `tf.float32`) data type to use.\n\n  Returns:\n    attention_score_fn: to compute similarity between key and target states.\n  '
    with tf.variable_scope(name, reuse=reuse):
        if attention_option == 'bahdanau':
            query_w = tf.get_variable('attnW', [num_units, num_units], dtype=dtype)
            score_v = tf.get_variable('attnV', [num_units], dtype=dtype)

        def attention_score_fn(query, keys, values):
            if False:
                for i in range(10):
                    print('nop')
            'Put attention masks on attention_values using attention_keys and query.\n\n      Args:\n        query: A Tensor of shape [batch_size, num_units].\n        keys: A Tensor of shape [batch_size, attention_length, num_units].\n        values: A Tensor of shape [batch_size, attention_length, num_units].\n\n      Returns:\n        context_vector: A Tensor of shape [batch_size, num_units].\n\n      Raises:\n        ValueError: if attention_option is neither "luong" or "bahdanau".\n\n\n      '
            if attention_option == 'bahdanau':
                query = tf.matmul(query, query_w)
                query = tf.reshape(query, [-1, 1, num_units])
                scores = _attn_add_fun(score_v, keys, query)
            elif attention_option == 'luong':
                query = tf.reshape(query, [-1, 1, num_units])
                scores = _attn_mul_fun(keys, query)
            else:
                raise ValueError('Unknown attention option %s!' % attention_option)
            alignments = tf.nn.softmax(scores)
            alignments = tf.expand_dims(alignments, 2)
            context_vector = tf.reduce_sum(alignments * values, [1])
            context_vector.set_shape([None, num_units])
            return context_vector
        return attention_score_fn