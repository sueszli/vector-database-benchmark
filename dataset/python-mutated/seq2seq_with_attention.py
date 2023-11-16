import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Dense, Dropout, Input
from tensorlayer.layers.core import Layer
from tensorlayer.models import Model
__all__ = ['Seq2seqLuongAttention']

class Encoder(Layer):

    def __init__(self, hidden_size, cell, embedding_layer, name=None):
        if False:
            return 10
        super(Encoder, self).__init__(name)
        self.cell = cell(hidden_size)
        self.hidden_size = hidden_size
        self.embedding_layer = embedding_layer
        self.build((None, None, self.embedding_layer.embedding_size))
        self._built = True

    def build(self, inputs_shape):
        if False:
            for i in range(10):
                print('nop')
        self.cell.build(input_shape=tuple(inputs_shape))
        self._built = True
        if self._trainable_weights is None:
            self._trainable_weights = list()
        for var in self.cell.trainable_variables:
            self._trainable_weights.append(var)

    def forward(self, src_seq, initial_state=None):
        if False:
            while True:
                i = 10
        states = initial_state if initial_state is not None else self.cell.get_initial_state(src_seq)
        encoding_hidden_states = list()
        total_steps = src_seq.get_shape().as_list()[1]
        for time_step in range(total_steps):
            if not isinstance(states, list):
                states = [states]
            (output, states) = self.cell.call(src_seq[:, time_step, :], states, training=self.is_train)
            encoding_hidden_states.append(states[0])
        return (output, encoding_hidden_states, states[0])

class Decoder_Attention(Layer):

    def __init__(self, hidden_size, cell, embedding_layer, method, name=None):
        if False:
            for i in range(10):
                print('nop')
        super(Decoder_Attention, self).__init__(name)
        self.cell = cell(hidden_size)
        self.hidden_size = hidden_size
        self.embedding_layer = embedding_layer
        self.method = method
        self.build((None, hidden_size + self.embedding_layer.embedding_size))
        self._built = True

    def build(self, inputs_shape):
        if False:
            while True:
                i = 10
        self.cell.build(input_shape=tuple(inputs_shape))
        self._built = True
        if self.method is 'concat':
            self.W = self._get_weights('W', shape=(2 * self.hidden_size, self.hidden_size))
            self.V = self._get_weights('V', shape=(self.hidden_size, 1))
        elif self.method is 'general':
            self.W = self._get_weights('W', shape=(self.hidden_size, self.hidden_size))
        if self._trainable_weights is None:
            self._trainable_weights = list()
        for var in self.cell.trainable_variables:
            self._trainable_weights.append(var)

    def score(self, encoding_hidden, hidden, method):
        if False:
            i = 10
            return i + 15
        if method is 'concat':
            hidden = tf.expand_dims(hidden, 1)
            hidden = tf.tile(hidden, [1, encoding_hidden.shape[1], 1])
            combined = tf.concat([hidden, encoding_hidden], 2)
            combined = tf.cast(combined, tf.float32)
            score = tf.tensordot(combined, self.W, axes=[[2], [0]])
            score = tf.nn.tanh(score)
            score = tf.tensordot(self.V, score, axes=[[0], [2]])
            score = tf.squeeze(score, axis=0)
        elif method is 'dot':
            hidden = tf.expand_dims(hidden, 2)
            score = tf.matmul(encoding_hidden, hidden)
            score = tf.squeeze(score, axis=2)
        elif method is 'general':
            score = tf.matmul(hidden, self.W)
            score = tf.expand_dims(score, 2)
            score = tf.matmul(encoding_hidden, score)
            score = tf.squeeze(score, axis=2)
        score = tf.nn.softmax(score, axis=-1)
        return score

    def forward(self, dec_seq, enc_hiddens, last_hidden, method, return_last_state=False):
        if False:
            print('Hello World!')
        total_steps = dec_seq.get_shape().as_list()[1]
        states = last_hidden
        cell_outputs = list()
        for time_step in range(total_steps):
            attention_weights = self.score(enc_hiddens, last_hidden, method)
            attention_weights = tf.expand_dims(attention_weights, 1)
            context = tf.matmul(attention_weights, enc_hiddens)
            context = tf.squeeze(context, 1)
            inputs = tf.concat([dec_seq[:, time_step, :], context], 1)
            if not isinstance(states, list):
                states = [states]
            (cell_output, states) = self.cell.call(inputs, states, training=self.is_train)
            cell_outputs.append(cell_output)
            last_hidden = states[0]
        cell_outputs = tf.convert_to_tensor(cell_outputs)
        cell_outputs = tf.transpose(cell_outputs, perm=[1, 0, 2])
        if return_last_state:
            return (cell_outputs, last_hidden)
        return cell_outputs

class Seq2seqLuongAttention(Model):
    """Luong Attention-based Seq2Seq model. Implementation based on https://arxiv.org/pdf/1508.04025.pdf.

    Parameters
    ----------
    hidden_size: int
        The hidden size of both encoder and decoder RNN cells
    cell : TensorFlow cell function
        The RNN function cell for your encoder and decoder stack, e.g. tf.keras.layers.GRUCell
    embedding_layer : tl.Layer
        A embedding layer, e.g. tl.layers.Embedding(vocabulary_size=voc_size, embedding_size=emb_dim)
    method : str
        The three alternatives to calculate the attention scores, e.g. "dot", "general" and "concat"
    name : str
        The model name
    

    Returns
    -------
        static single layer attention-based Seq2Seq model.
    """

    def __init__(self, hidden_size, embedding_layer, cell, method, name=None):
        if False:
            for i in range(10):
                print('nop')
        super(Seq2seqLuongAttention, self).__init__(name)
        self.enc_layer = Encoder(hidden_size, cell, embedding_layer)
        self.dec_layer = Decoder_Attention(hidden_size, cell, embedding_layer, method=method)
        self.embedding_layer = embedding_layer
        self.dense_layer = tl.layers.Dense(n_units=self.embedding_layer.vocabulary_size, in_channels=hidden_size)
        self.method = method

    def inference(self, src_seq, encoding_hidden_states, last_hidden_states, seq_length, sos):
        if False:
            for i in range(10):
                print('nop')
        'Inference mode'
        '\n        Parameters\n        ----------\n        src_seq : input tensor\n            The source sequences \n        encoding_hidden_states : a list of tensor\n            The list of encoder\'s hidden states at each time step\n        last_hidden_states: tensor\n            The last hidden_state from encoder\n        seq_length : int\n            The expected length of your predicted sequence.\n        sos : int\n            <SOS> : The token of "start of sequence"\n        '
        batch_size = src_seq.shape[0]
        decoding = [[sos] for i in range(batch_size)]
        dec_output = self.embedding_layer(decoding)
        outputs = [[0] for i in range(batch_size)]
        for step in range(seq_length):
            (dec_output, last_hidden_states) = self.dec_layer(dec_output, encoding_hidden_states, last_hidden_states, method=self.method, return_last_state=True)
            dec_output = tf.reshape(dec_output, [-1, dec_output.shape[-1]])
            dec_output = self.dense_layer(dec_output)
            dec_output = tf.reshape(dec_output, [batch_size, -1, dec_output.shape[-1]])
            dec_output = tf.argmax(dec_output, -1)
            outputs = tf.concat([outputs, dec_output], 1)
            dec_output = self.embedding_layer(dec_output)
        return outputs[:, 1:]

    def forward(self, inputs, seq_length=20, sos=None):
        if False:
            while True:
                i = 10
        src_seq = inputs[0]
        src_seq = self.embedding_layer(src_seq)
        (enc_output, encoding_hidden_states, last_hidden_states) = self.enc_layer(src_seq)
        encoding_hidden_states = tf.convert_to_tensor(encoding_hidden_states)
        encoding_hidden_states = tf.transpose(encoding_hidden_states, perm=[1, 0, 2])
        last_hidden_states = tf.convert_to_tensor(last_hidden_states)
        if self.is_train:
            dec_seq = inputs[1]
            dec_seq = self.embedding_layer(dec_seq)
            dec_output = self.dec_layer(dec_seq, encoding_hidden_states, last_hidden_states, method=self.method)
            batch_size = dec_output.shape[0]
            dec_output = tf.reshape(dec_output, [-1, dec_output.shape[-1]])
            dec_output = self.dense_layer(dec_output)
            dec_output = tf.reshape(dec_output, [batch_size, -1, dec_output.shape[-1]])
        else:
            dec_output = self.inference(src_seq, encoding_hidden_states, last_hidden_states, seq_length, sos)
        return dec_output