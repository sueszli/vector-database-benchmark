"""Custom RNN decoder."""
import tensorflow as tf

def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None, scope=None):
    if False:
        for i in range(10):
            print('nop')
    'RNN decoder for the LSTM-SSD model.\n\n  This decoder returns a list of all states, rather than only the final state.\n  Args:\n    decoder_inputs: A list of 4D Tensors with shape [batch_size x input_size].\n    initial_state: 2D Tensor with shape [batch_size x cell.state_size].\n    cell: rnn_cell.RNNCell defining the cell function and size.\n    loop_function: If not None, this function will be applied to the i-th output\n      in order to generate the i+1-st input, and decoder_inputs will be ignored,\n      except for the first element ("GO" symbol). This can be used for decoding,\n      but also for training to emulate http://arxiv.org/abs/1506.03099.\n      Signature -- loop_function(prev, i) = next\n        * prev is a 2D Tensor of shape [batch_size x output_size],\n        * i is an integer, the step number (when advanced control is needed),\n        * next is a 2D Tensor of shape [batch_size x input_size].\n    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".\n  Returns:\n    A tuple of the form (outputs, state), where:\n      outputs: A list of the same length as decoder_inputs of 4D Tensors with\n        shape [batch_size x output_size] containing generated outputs.\n      states: A list of the same length as decoder_inputs of the state of each\n        cell at each time-step. It is a 2D Tensor of shape\n        [batch_size x cell.state_size].\n  '
    with tf.variable_scope(scope or 'rnn_decoder'):
        state_tuple = initial_state
        outputs = []
        states = []
        prev = None
        for (local_step, decoder_input) in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with tf.variable_scope('loop_function', reuse=True):
                    decoder_input = loop_function(prev, local_step)
            (output, state_tuple) = cell(decoder_input, state_tuple)
            outputs.append(output)
            states.append(state_tuple)
            if loop_function is not None:
                prev = output
    return (outputs, states)

def multi_input_rnn_decoder(decoder_inputs, initial_state, cell, sequence_step, selection_strategy='RANDOM', is_training=None, is_quantized=False, preprocess_fn_list=None, pre_bottleneck=False, flatten_state=False, scope=None):
    if False:
        i = 10
        return i + 15
    'RNN decoder for the Interleaved LSTM-SSD model.\n\n  This decoder takes multiple sequences of inputs and selects the input to feed\n  to the rnn at each timestep using its selection_strategy, which can be random,\n  learned, or deterministic.\n  This decoder returns a list of all states, rather than only the final state.\n  Args:\n    decoder_inputs: A list of lists of 2D Tensors [batch_size x input_size].\n    initial_state: 2D Tensor with shape [batch_size x cell.state_size].\n    cell: rnn_cell.RNNCell defining the cell function and size.\n    sequence_step: Tensor [batch_size] of the step number of the first elements\n      in the sequence.\n    selection_strategy: Method for picking the decoder_input to use at each\n      timestep. Must be \'RANDOM\', \'SKIPX\' for integer X,  where X is the number\n      of times to use the second input before using the first.\n    is_training: boolean, whether the network is training. When using learned\n      selection, attempts exploration if training.\n    is_quantized: flag to enable/disable quantization mode.\n    preprocess_fn_list: List of functions accepting two tensor arguments: one\n      timestep of decoder_inputs and the lstm state. If not None,\n      decoder_inputs[i] will be updated with preprocess_fn[i] at the start of\n      each timestep.\n    pre_bottleneck: if True, use separate bottleneck weights for each sequence.\n      Useful when input sequences have differing numbers of channels. Final\n      bottlenecks will have the same dimension.\n    flatten_state: Whether the LSTM state is flattened.\n    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".\n  Returns:\n    A tuple of the form (outputs, state), where:\n      outputs: A list of the same length as decoder_inputs of 2D Tensors with\n        shape [batch_size x output_size] containing generated outputs.\n      states: A list of the same length as decoder_inputs of the state of each\n        cell at each time-step. It is a 2D Tensor of shape\n        [batch_size x cell.state_size].\n  Raises:\n    ValueError: If selection_strategy is not recognized or unexpected unroll\n      length.\n  '
    if flatten_state and len(decoder_inputs[0]) > 1:
        raise ValueError('In export mode, unroll length should not be more than 1')
    with tf.variable_scope(scope or 'rnn_decoder'):
        state_tuple = initial_state
        outputs = []
        states = []
        batch_size = decoder_inputs[0][0].shape[0].value
        num_sequences = len(decoder_inputs)
        sequence_length = len(decoder_inputs[0])
        for local_step in range(sequence_length):
            for sequence_index in range(num_sequences):
                if preprocess_fn_list is not None:
                    decoder_inputs[sequence_index][local_step] = preprocess_fn_list[sequence_index](decoder_inputs[sequence_index][local_step], state_tuple[0])
                if pre_bottleneck:
                    decoder_inputs[sequence_index][local_step] = cell.pre_bottleneck(inputs=decoder_inputs[sequence_index][local_step], state=state_tuple[1], input_index=sequence_index)
            action = generate_action(selection_strategy, local_step, sequence_step, [batch_size, 1, 1, 1])
            (inputs, _) = select_inputs(decoder_inputs, action, local_step)
            with tf.name_scope(None):
                inputs = tf.identity(inputs, 'raw_inputs/base_endpoint')
            (output, state_tuple_out) = cell(inputs, state_tuple)
            state_tuple = select_state(state_tuple, state_tuple_out, action)
            outputs.append(output)
            states.append(state_tuple)
    return (outputs, states)

def generate_action(selection_strategy, local_step, sequence_step, action_shape):
    if False:
        print('Hello World!')
    "Generate current (binary) action based on selection strategy.\n\n  Args:\n    selection_strategy: Method for picking the decoder_input to use at each\n      timestep. Must be 'RANDOM', 'SKIPX' for integer X,  where X is the number\n      of times to use the second input before using the first.\n    local_step: Tensor [batch_size] of the step number within the current\n      unrolled batch.\n    sequence_step: Tensor [batch_size] of the step number of the first elements\n      in the sequence.\n    action_shape: The shape of action tensor to be generated.\n\n  Returns:\n    A tensor of shape action_shape, each element is an individual action.\n\n  Raises:\n    ValueError: if selection_strategy is not supported or if 'SKIP' is not\n      followed by numerics.\n  "
    if selection_strategy.startswith('RANDOM'):
        action = tf.random.uniform(action_shape, maxval=2, dtype=tf.int32)
        action = tf.minimum(action, 1)
        if local_step == 0 and sequence_step is not None:
            action *= tf.minimum(tf.reshape(tf.cast(sequence_step, tf.int32), action_shape), 1)
    elif selection_strategy.startswith('SKIP'):
        inter_count = int(selection_strategy[4:])
        if local_step % (inter_count + 1) == 0:
            action = tf.zeros(action_shape)
        else:
            action = tf.ones(action_shape)
    else:
        raise ValueError('Selection strategy %s not recognized' % selection_strategy)
    return tf.cast(action, tf.int32)

def select_inputs(decoder_inputs, action, local_step, get_alt_inputs=False):
    if False:
        print('Hello World!')
    'Selects sequence from decoder_inputs based on 1D actions.\n\n  Given multiple input batches, creates a single output batch by\n  selecting from the action[i]-ith input for the i-th batch element.\n\n  Args:\n    decoder_inputs: A 2-D list of tensor inputs.\n    action: A tensor of shape [batch_size]. Each element corresponds to an index\n      of decoder_inputs to choose.\n    step: The current timestep.\n    get_alt_inputs: Whether the non-chosen inputs should also be returned.\n\n  Returns:\n    The constructed output. Also outputs the elements that were not chosen\n    if get_alt_inputs is True, otherwise None.\n\n  Raises:\n    ValueError: if the decoder inputs contains other than two sequences.\n  '
    num_seqs = len(decoder_inputs)
    if not num_seqs == 2:
        raise ValueError('Currently only supports two sets of inputs.')
    stacked_inputs = tf.stack([decoder_inputs[seq_index][local_step] for seq_index in range(num_seqs)], axis=-1)
    action_index = tf.one_hot(action, num_seqs)
    inputs = tf.reduce_sum(stacked_inputs * action_index, axis=-1)
    inputs_alt = None
    if get_alt_inputs:
        action_index_alt = tf.one_hot(action, num_seqs, on_value=0.0, off_value=1.0)
        inputs_alt = tf.reduce_sum(stacked_inputs * action_index_alt, axis=-1)
    return (inputs, inputs_alt)

def select_state(previous_state, new_state, action):
    if False:
        while True:
            i = 10
    'Select state given action.\n\n  Currently only supports binary action. If action is 0, it means the state is\n  generated from the large model, and thus we will update the state. Otherwise,\n  if the action is 1, it means the state is generated from the small model, and\n  in interleaved model, we skip this state update.\n\n  Args:\n    previous_state: A state tuple representing state from previous step.\n    new_state: A state tuple representing newly computed state.\n    action: A tensor the same shape as state.\n\n  Returns:\n    A state tuple selected based on the given action.\n  '
    action = tf.cast(action, tf.float32)
    state_c = previous_state[0] * action + new_state[0] * (1 - action)
    state_h = previous_state[1] * action + new_state[1] * (1 - action)
    return (state_c, state_h)