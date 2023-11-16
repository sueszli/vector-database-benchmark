"""Network units wrapping TensorFlows' tf.contrib.rnn cells.

Please put all wrapping logic for tf.contrib.rnn in this module; this will help
collect common subroutines that prove useful.
"""
import abc
import tensorflow as tf
from dragnn.python import network_units as dragnn
from syntaxnet.util import check

def capture_variables(function, scope_name):
    if False:
        for i in range(10):
            print('nop')
    'Captures and returns variables created by a function.\n\n  Runs |function| in a scope of name |scope_name| and returns the list of\n  variables created by |function|.\n\n  Args:\n    function: Function whose variables should be captured.  The function should\n        take one argument, its enclosing variable scope.\n    scope_name: Variable scope in which the |function| is evaluated.\n\n  Returns:\n    List of created variables.\n  '
    created_vars = {}

    def _custom_getter(getter, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Calls the real getter and captures its result in |created_vars|.'
        real_variable = getter(*args, **kwargs)
        created_vars[real_variable.name] = real_variable
        return real_variable
    with tf.variable_scope(scope_name, reuse=None, custom_getter=_custom_getter) as scope:
        function(scope)
    return created_vars.values()

def apply_with_captured_variables(function, scope_name, component):
    if False:
        return 10
    'Applies a function using previously-captured variables.\n\n  The counterpart to capture_variables(); invokes |function| in a scope of name\n  |scope_name|, extracting captured variables from the |component|.\n\n  Args:\n    function: Function to apply using captured variables.  The function should\n        take one argument, its enclosing variable scope.\n    scope_name: Variable scope in which the |function| is evaluated.  Must match\n        the scope passed to capture_variables().\n    component: Component from which to extract captured variables.\n\n  Returns:\n    Results of function application.\n  '

    def _custom_getter(getter, *args, **kwargs):
        if False:
            print('Hello World!')
        'Retrieves the normal or moving-average variables.'
        return component.get_variable(var_params=getter(*args, **kwargs))
    with tf.variable_scope(scope_name, reuse=True, custom_getter=_custom_getter) as scope:
        return function(scope)

class BaseLSTMNetwork(dragnn.NetworkUnitInterface):
    """Base class for wrapped LSTM networks.

  This LSTM network unit supports multiple layers with layer normalization.
  Because it is imported from tf.contrib.rnn, we need to capture the created
  variables during initialization time.

  Layers:
    ...subclass-specific layers...
    last_layer: Alias for the activations of the last hidden layer.
    logits: Logits associated with component actions.
  """

    def __init__(self, component, additional_attr_defaults=None):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the LSTM base class.\n\n    Parameters used:\n      hidden_layer_sizes: Comma-delimited number of hidden units for each layer.\n      input_dropout_rate (-1.0): Input dropout rate for each layer.  If < 0.0,\n          use the global |dropout_rate| hyperparameter.\n      recurrent_dropout_rate (0.8): Recurrent dropout rate.  If < 0.0, use the\n          global |recurrent_dropout_rate| hyperparameter.\n      layer_norm (True): Whether or not to use layer norm.\n\n    Hyperparameters used:\n      dropout_rate: Input dropout rate.\n      recurrent_dropout_rate: Recurrent dropout rate.\n\n    Args:\n      component: parent ComponentBuilderBase object.\n      additional_attr_defaults: Additional attributes for use by derived class.\n    '
        attr_defaults = additional_attr_defaults or {}
        attr_defaults.update({'layer_norm': True, 'input_dropout_rate': -1.0, 'recurrent_dropout_rate': 0.8, 'hidden_layer_sizes': '256'})
        self._attrs = dragnn.get_attrs_with_defaults(component.spec.network_unit.parameters, defaults=attr_defaults)
        self._hidden_layer_sizes = map(int, self._attrs['hidden_layer_sizes'].split(','))
        self._input_dropout_rate = self._attrs['input_dropout_rate']
        if self._input_dropout_rate < 0.0:
            self._input_dropout_rate = component.master.hyperparams.dropout_rate
        self._recurrent_dropout_rate = self._attrs['recurrent_dropout_rate']
        if self._recurrent_dropout_rate < 0.0:
            self._recurrent_dropout_rate = component.master.hyperparams.recurrent_dropout_rate
        if self._recurrent_dropout_rate < 0.0:
            self._recurrent_dropout_rate = component.master.hyperparams.dropout_rate
        tf.logging.info('[%s] input_dropout_rate=%s recurrent_dropout_rate=%s', component.name, self._input_dropout_rate, self._recurrent_dropout_rate)
        (layers, context_layers) = self.create_hidden_layers(component, self._hidden_layer_sizes)
        last_layer_dim = layers[-1].dim
        layers.append(dragnn.Layer(component, name='last_layer', dim=last_layer_dim))
        layers.append(dragnn.Layer(component, name='logits', dim=component.num_actions))
        super(BaseLSTMNetwork, self).__init__(component, init_layers=layers, init_context_layers=context_layers)
        self._params.append(tf.get_variable('weights_softmax', [last_layer_dim, component.num_actions], initializer=tf.random_normal_initializer(stddev=0.0001)))
        self._params.append(tf.get_variable('bias_softmax', [component.num_actions], initializer=tf.zeros_initializer()))

    def get_logits(self, network_tensors):
        if False:
            print('Hello World!')
        'Returns the logits for prediction.'
        return network_tensors[self.get_layer_index('logits')]

    @abc.abstractmethod
    def create_hidden_layers(self, component, hidden_layer_sizes):
        if False:
            for i in range(10):
                print('nop')
        'Creates hidden network layers.\n\n    Args:\n      component: Parent ComponentBuilderBase object.\n      hidden_layer_sizes: List of requested hidden layer activation sizes.\n\n    Returns:\n      layers: List of layers created by this network.\n      context_layers: List of context layers created by this network.\n    '
        pass

    def _append_base_layers(self, hidden_layers):
        if False:
            while True:
                i = 10
        'Appends layers defined by the base class to the |hidden_layers|.'
        last_layer = hidden_layers[-1]
        logits = tf.nn.xw_plus_b(last_layer, self._component.get_variable('weights_softmax'), self._component.get_variable('bias_softmax'))
        return hidden_layers + [last_layer, logits]

    def _create_cell(self, num_units, during_training):
        if False:
            i = 10
            return i + 15
        'Creates a single LSTM cell, possibly with dropout.\n\n    Requires that BaseLSTMNetwork.__init__() was called.\n\n    Args:\n      num_units: Number of hidden units in the cell.\n      during_training: Whether to create a cell for training (vs inference).\n\n    Returns:\n      A RNNCell of the requested size, possibly with dropout.\n    '
        if not during_training:
            return tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, layer_norm=self._attrs['layer_norm'], reuse=True)
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, dropout_keep_prob=self._recurrent_dropout_rate, layer_norm=self._attrs['layer_norm'])
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self._input_dropout_rate)
        return cell

    def _create_train_cells(self):
        if False:
            return 10
        'Creates a list of LSTM cells for training.'
        return [self._create_cell(num_units, during_training=True) for num_units in self._hidden_layer_sizes]

    def _create_inference_cells(self):
        if False:
            while True:
                i = 10
        'Creates a list of LSTM cells for inference.'
        return [self._create_cell(num_units, during_training=False) for num_units in self._hidden_layer_sizes]

    def _capture_variables_as_params(self, function):
        if False:
            while True:
                i = 10
        'Captures variables created by a function in |self._params|.'
        self._params.extend(capture_variables(function, 'cell'))

    def _apply_with_captured_variables(self, function):
        if False:
            while True:
                i = 10
        'Applies a function using previously-captured variables.'
        return apply_with_captured_variables(function, 'cell', self._component)

class LayerNormBasicLSTMNetwork(BaseLSTMNetwork):
    """Wrapper around tf.contrib.rnn.LayerNormBasicLSTMCell.

  Features:
    All inputs are concatenated.

  Subclass-specific layers:
    state_c_<n>: Cell states for the <n>'th LSTM layer (0-origin).
    state_h_<n>: Hidden states for the <n>'th LSTM layer (0-origin).
  """

    def __init__(self, component):
        if False:
            while True:
                i = 10
        'Sets up context and output layers, as well as a final softmax.'
        super(LayerNormBasicLSTMNetwork, self).__init__(component)
        self._train_cell = tf.contrib.rnn.MultiRNNCell(self._create_train_cells())
        self._inference_cell = tf.contrib.rnn.MultiRNNCell(self._create_inference_cells())

        def _cell_closure(scope):
            if False:
                print('Hello World!')
            'Applies the LSTM cell to placeholder inputs and state.'
            placeholder_inputs = tf.placeholder(dtype=tf.float32, shape=(1, self._concatenated_input_dim))
            placeholder_substates = []
            for num_units in self._hidden_layer_sizes:
                placeholder_substate = tf.contrib.rnn.LSTMStateTuple(tf.placeholder(dtype=tf.float32, shape=(1, num_units)), tf.placeholder(dtype=tf.float32, shape=(1, num_units)))
                placeholder_substates.append(placeholder_substate)
            placeholder_state = tuple(placeholder_substates)
            self._train_cell(inputs=placeholder_inputs, state=placeholder_state, scope=scope)
        self._capture_variables_as_params(_cell_closure)

    def create_hidden_layers(self, component, hidden_layer_sizes):
        if False:
            for i in range(10):
                print('nop')
        'See base class.'
        layers = []
        for (index, num_units) in enumerate(hidden_layer_sizes):
            layers.append(dragnn.Layer(component, name='state_c_%d' % index, dim=num_units))
            layers.append(dragnn.Layer(component, name='state_h_%d' % index, dim=num_units))
        context_layers = list(layers)
        return (layers, context_layers)

    def create(self, fixed_embeddings, linked_embeddings, context_tensor_arrays, attention_tensor, during_training, stride=None):
        if False:
            return 10
        'See base class.'
        check.Eq(len(context_tensor_arrays), 2 * len(self._hidden_layer_sizes), 'require two context tensors per hidden layer')
        length = context_tensor_arrays[0].size()
        substates = []
        for (index, num_units) in enumerate(self._hidden_layer_sizes):
            state_c = context_tensor_arrays[2 * index].read(length - 1)
            state_h = context_tensor_arrays[2 * index + 1].read(length - 1)
            state_c.set_shape([tf.Dimension(None), num_units])
            state_h.set_shape([tf.Dimension(None), num_units])
            substates.append(tf.contrib.rnn.LSTMStateTuple(state_c, state_h))
        state = tuple(substates)
        input_tensor = dragnn.get_input_tensor(fixed_embeddings, linked_embeddings)
        cell = self._train_cell if during_training else self._inference_cell

        def _cell_closure(scope):
            if False:
                i = 10
                return i + 15
            'Applies the LSTM cell to the current inputs and state.'
            return cell(input_tensor, state, scope=scope)
        (unused_h, state) = self._apply_with_captured_variables(_cell_closure)
        output_tensors = []
        for new_substate in state:
            (new_c, new_h) = new_substate
            output_tensors.append(new_c)
            output_tensors.append(new_h)
        return self._append_base_layers(output_tensors)

class BulkBiLSTMNetwork(BaseLSTMNetwork):
    """Bulk wrapper around tf.contrib.rnn.stack_bidirectional_dynamic_rnn().

  Features:
    lengths: [stride, 1] sequence lengths per batch item.
    All other features are concatenated into input activations.

  Subclass-specific layers:
    outputs: [stride * num_steps, self._output_dim] bi-LSTM activations.
  """

    def __init__(self, component):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the bulk bi-LSTM.\n\n    Parameters used:\n      parallel_iterations (1): Parallelism of the underlying tf.while_loop().\n        Defaults to 1 thread to encourage deterministic behavior, but can be\n        increased to trade memory for speed.\n\n    Args:\n      component: parent ComponentBuilderBase object.\n    '
        super(BulkBiLSTMNetwork, self).__init__(component, additional_attr_defaults={'parallel_iterations': 1})
        check.In('lengths', self._linked_feature_dims, 'Missing required linked feature')
        check.Eq(self._linked_feature_dims['lengths'], 1, 'Wrong dimension for "lengths" feature')
        self._input_dim = self._concatenated_input_dim - 1
        self._output_dim = self.get_layer_size('outputs')
        tf.logging.info('[%s] Bulk bi-LSTM with input_dim=%d output_dim=%d', component.name, self._input_dim, self._output_dim)
        self._train_cells_forward = self._create_train_cells()
        self._train_cells_backward = self._create_train_cells()
        self._inference_cells_forward = self._create_inference_cells()
        self._inference_cells_backward = self._create_inference_cells()

        def _bilstm_closure(scope):
            if False:
                for i in range(10):
                    print('nop')
            'Applies the bi-LSTM to placeholder inputs and lengths.'
            (stride, steps) = (1, 1)
            placeholder_inputs = tf.placeholder(dtype=tf.float32, shape=[stride, steps, self._input_dim])
            placeholder_lengths = tf.placeholder(dtype=tf.int64, shape=[stride])
            tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self._train_cells_forward, self._train_cells_backward, placeholder_inputs, dtype=tf.float32, sequence_length=placeholder_lengths, scope=scope)
        self._capture_variables_as_params(_bilstm_closure)
        for (index, num_units) in enumerate(self._hidden_layer_sizes):
            for direction in ['forward', 'backward']:
                for substate in ['c', 'h']:
                    self._params.append(tf.get_variable('initial_state_%s_%s_%d' % (direction, substate, index), [1, num_units], dtype=tf.float32, initializer=tf.constant_initializer(0.0)))

    def create_hidden_layers(self, component, hidden_layer_sizes):
        if False:
            print('Hello World!')
        'See base class.'
        dim = 2 * hidden_layer_sizes[-1]
        return ([dragnn.Layer(component, name='outputs', dim=dim)], [])

    def create(self, fixed_embeddings, linked_embeddings, context_tensor_arrays, attention_tensor, during_training, stride=None):
        if False:
            for i in range(10):
                print('nop')
        'Requires |stride|; otherwise see base class.'
        check.NotNone(stride, 'BulkBiLSTMNetwork requires "stride" and must be called in the bulk feature extractor component.')
        lengths = dragnn.lookup_named_tensor('lengths', linked_embeddings)
        lengths_s = tf.squeeze(lengths.tensor, [1])
        linked_embeddings = [named_tensor for named_tensor in linked_embeddings if named_tensor.name != 'lengths']
        inputs_sxnxd = dragnn.get_input_tensor_with_stride(fixed_embeddings, linked_embeddings, stride)
        inputs_sxnxd.set_shape([tf.Dimension(None), tf.Dimension(None), self._input_dim])
        (initial_states_forward, initial_states_backward) = self._create_initial_states(stride)
        if during_training:
            cells_forward = self._train_cells_forward
            cells_backward = self._train_cells_backward
        else:
            cells_forward = self._inference_cells_forward
            cells_backward = self._inference_cells_backward

        def _bilstm_closure(scope):
            if False:
                print('Hello World!')
            'Applies the bi-LSTM to the current inputs.'
            (outputs_sxnxd, _, _) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_forward, cells_backward, inputs_sxnxd, initial_states_fw=initial_states_forward, initial_states_bw=initial_states_backward, sequence_length=lengths_s, parallel_iterations=self._attrs['parallel_iterations'], scope=scope)
            return outputs_sxnxd
        outputs_sxnxd = self._apply_with_captured_variables(_bilstm_closure)
        outputs_snxd = tf.reshape(outputs_sxnxd, [-1, self._output_dim])
        return self._append_base_layers([outputs_snxd])

    def _create_initial_states(self, stride):
        if False:
            print('Hello World!')
        'Returns stacked and batched initial states for the bi-LSTM.'
        initial_states_forward = []
        initial_states_backward = []
        for index in range(len(self._hidden_layer_sizes)):
            states_sxd = []
            for direction in ['forward', 'backward']:
                for substate in ['c', 'h']:
                    state_1xd = self._component.get_variable('initial_state_%s_%s_%d' % (direction, substate, index))
                    state_sxd = tf.tile(state_1xd, [stride, 1])
                    states_sxd.append(state_sxd)
            initial_states_forward.append(tf.contrib.rnn.LSTMStateTuple(states_sxd[0], states_sxd[1]))
            initial_states_backward.append(tf.contrib.rnn.LSTMStateTuple(states_sxd[2], states_sxd[3]))
        return (initial_states_forward, initial_states_backward)