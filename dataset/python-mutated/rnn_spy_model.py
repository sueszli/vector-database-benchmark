import numpy as np
import pickle
import ray
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
(tf1, tf, tfv) = try_import_tf()

class SpyLayer(tf.keras.layers.Layer):
    """A keras Layer, which intercepts its inputs and stored them as pickled."""
    output = np.array(0, dtype=np.int64)

    def __init__(self, num_outputs, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=num_outputs, kernel_initializer=normc_initializer(0.01))

    def call(self, inputs, **kwargs):
        if False:
            while True:
                i = 10
        'Does a forward pass through our Dense, but also intercepts inputs.'
        del kwargs
        spy_fn = tf1.py_func(self.spy, [inputs[0], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]], tf.int64, stateful=True)
        with tf1.control_dependencies([spy_fn]):
            return self.dense(inputs[1])

    @staticmethod
    def spy(inputs, seq_lens, h_in, c_in, h_out, c_out):
        if False:
            while True:
                i = 10
        'The actual spy operation: Store inputs in internal_kv.'
        if len(inputs) == 1:
            return SpyLayer.output
        ray.experimental.internal_kv._internal_kv_put('rnn_spy_in_{}'.format(RNNSpyModel.capture_index), pickle.dumps({'sequences': inputs, 'seq_lens': seq_lens, 'state_in': [h_in, c_in], 'state_out': [h_out, c_out]}), overwrite=True)
        RNNSpyModel.capture_index += 1
        return SpyLayer.output

class RNNSpyModel(RecurrentNetwork):
    capture_index = 0
    cell_size = 3

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        if False:
            return 10
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = RNNSpyModel.cell_size
        inputs = tf.keras.layers.Input(shape=(None,) + obs_space.shape, name='input')
        state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name='h')
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name='c')
        seq_lens = tf.keras.layers.Input(shape=(), name='seq_lens', dtype=tf.int32)
        (lstm_out, state_out_h, state_out_c) = tf.keras.layers.LSTM(self.cell_size, return_sequences=True, return_state=True, name='lstm')(inputs=inputs, mask=tf.sequence_mask(seq_lens), initial_state=[state_in_h, state_in_c])
        logits = SpyLayer(num_outputs=self.num_outputs)([inputs, lstm_out, seq_lens, state_in_h, state_in_c, state_out_h, state_out_c])
        value_out = tf.keras.layers.Dense(units=1, kernel_initializer=normc_initializer(1.0))(lstm_out)
        self.base_model = tf.keras.Model([inputs, seq_lens, state_in_h, state_in_c], [logits, value_out, state_out_h, state_out_c])
        self.base_model.summary()

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        if False:
            print('Hello World!')
        RNNSpyModel.capture_index = 0
        (model_out, value_out, h, c) = self.base_model([inputs, seq_lens, state[0], state[1]])
        self._value_out = value_out
        return (model_out, [h, c])

    @override(ModelV2)
    def value_function(self):
        if False:
            i = 10
            return i + 15
        return tf.reshape(self._value_out, [-1])

    @override(ModelV2)
    def get_initial_state(self):
        if False:
            print('Hello World!')
        return [np.zeros(self.cell_size, np.float32), np.zeros(self.cell_size, np.float32)]