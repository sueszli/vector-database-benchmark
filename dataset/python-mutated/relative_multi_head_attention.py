from typing import Optional
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.deprecation import deprecation_warning
from ray.util import log_once
(tf1, tf, tfv) = try_import_tf()

class RelativeMultiHeadAttention(tf.keras.layers.Layer if tf else object):
    """A RelativeMultiHeadAttention layer as described in [3].

    Uses segment level recurrence with state reuse.
    """

    def __init__(self, out_dim: int, num_heads: int, head_dim: int, input_layernorm: bool=False, output_activation: Optional['tf.nn.activation']=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a RelativeMultiHeadAttention keras Layer object.\n\n        Args:\n            out_dim: The output dimensions of the multi-head attention\n                unit.\n            num_heads: The number of attention heads to use.\n                Denoted `H` in [2].\n            head_dim: The dimension of a single(!) attention head within\n                a multi-head attention unit. Denoted as `d` in [3].\n            input_layernorm: Whether to prepend a LayerNorm before\n                everything else. Should be True for building a GTrXL.\n            output_activation (Optional[tf.nn.activation]): Optional tf.nn\n                activation function. Should be relu for GTrXL.\n            **kwargs:\n        '
        if log_once('relative_multi_head_attention'):
            deprecation_warning(old='rllib.models.tf.layers.RelativeMultiHeadAttention')
        super().__init__(**kwargs)
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._qkv_layer = tf.keras.layers.Dense(3 * num_heads * head_dim, use_bias=False)
        self._linear_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(out_dim, use_bias=False, activation=output_activation))
        self._uvar = self.add_weight(shape=(num_heads, head_dim))
        self._vvar = self.add_weight(shape=(num_heads, head_dim))
        self._pos_embedding = PositionalEmbedding(out_dim)
        self._pos_proj = tf.keras.layers.Dense(num_heads * head_dim, use_bias=False)
        self._input_layernorm = None
        if input_layernorm:
            self._input_layernorm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, inputs: TensorType, memory: Optional[TensorType]=None) -> TensorType:
        if False:
            i = 10
            return i + 15
        T = tf.shape(inputs)[1]
        H = self._num_heads
        d = self._head_dim
        Tau = tf.shape(memory)[1]
        inputs = tf.concat([tf.stop_gradient(memory), inputs], axis=1)
        if self._input_layernorm is not None:
            inputs = self._input_layernorm(inputs)
        qkv = self._qkv_layer(inputs)
        (queries, keys, values) = tf.split(qkv, 3, -1)
        queries = queries[:, -T:]
        queries = tf.reshape(queries, [-1, T, H, d])
        keys = tf.reshape(keys, [-1, Tau + T, H, d])
        values = tf.reshape(values, [-1, Tau + T, H, d])
        R = self._pos_embedding(Tau + T)
        R = self._pos_proj(R)
        R = tf.reshape(R, [Tau + T, H, d])
        score = tf.einsum('bihd,bjhd->bijh', queries + self._uvar, keys)
        pos_score = tf.einsum('bihd,jhd->bijh', queries + self._vvar, R)
        score = score + self.rel_shift(pos_score)
        score = score / d ** 0.5
        mask = tf.sequence_mask(tf.range(Tau + 1, Tau + T + 1), dtype=score.dtype)
        mask = mask[None, :, :, None]
        masked_score = score * mask + 1e+30 * (mask - 1.0)
        wmat = tf.nn.softmax(masked_score, axis=2)
        out = tf.einsum('bijh,bjhd->bihd', wmat, values)
        out = tf.reshape(out, tf.concat((tf.shape(out)[:2], [H * d]), axis=0))
        return self._linear_layer(out)

    @staticmethod
    def rel_shift(x: TensorType) -> TensorType:
        if False:
            i = 10
            return i + 15
        x_size = tf.shape(x)
        x = tf.pad(x, [[0, 0], [0, 0], [1, 0], [0, 0]])
        x = tf.reshape(x, [x_size[0], x_size[2] + 1, x_size[1], x_size[3]])
        x = x[:, 1:, :, :]
        x = tf.reshape(x, x_size)
        return x

class PositionalEmbedding(tf.keras.layers.Layer if tf else object):

    def __init__(self, out_dim, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.inverse_freq = 1 / 10000 ** (tf.range(0, out_dim, 2.0) / out_dim)

    def call(self, seq_length):
        if False:
            for i in range(10):
                print('nop')
        pos_offsets = tf.cast(tf.range(seq_length - 1, -1, -1), tf.float32)
        inputs = pos_offsets[:, None] * self.inverse_freq[None, :]
        return tf.concat((tf.sin(inputs), tf.cos(inputs)), axis=-1)