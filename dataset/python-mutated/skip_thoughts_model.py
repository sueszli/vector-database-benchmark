"""Skip-Thoughts model for learning sentence vectors.

The model is based on the paper:

  "Skip-Thought Vectors"
  Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel,
  Antonio Torralba, Raquel Urtasun, Sanja Fidler.
  https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf

Layer normalization is applied based on the paper:

  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
  https://arxiv.org/abs/1607.06450
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from skip_thoughts.ops import gru_cell
from skip_thoughts.ops import input_ops

def random_orthonormal_initializer(shape, dtype=tf.float32, partition_info=None):
    if False:
        i = 10
        return i + 15
    'Variable initializer that produces a random orthonormal matrix.'
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError('Expecting square shape, got %s' % shape)
    (_, u, _) = tf.svd(tf.random_normal(shape, dtype=dtype), full_matrices=True)
    return u

class SkipThoughtsModel(object):
    """Skip-thoughts model."""

    def __init__(self, config, mode='train', input_reader=None):
        if False:
            for i in range(10):
                print('nop')
        'Basic setup. The actual TensorFlow graph is constructed in build().\n\n    Args:\n      config: Object containing configuration parameters.\n      mode: "train", "eval" or "encode".\n      input_reader: Subclass of tf.ReaderBase for reading the input serialized\n        tf.Example protocol buffers. Defaults to TFRecordReader.\n\n    Raises:\n      ValueError: If mode is invalid.\n    '
        if mode not in ['train', 'eval', 'encode']:
            raise ValueError('Unrecognized mode: %s' % mode)
        self.config = config
        self.mode = mode
        self.reader = input_reader if input_reader else tf.TFRecordReader()
        self.uniform_initializer = tf.random_uniform_initializer(minval=-self.config.uniform_init_scale, maxval=self.config.uniform_init_scale)
        self.encode_ids = None
        self.decode_pre_ids = None
        self.decode_post_ids = None
        self.encode_mask = None
        self.decode_pre_mask = None
        self.decode_post_mask = None
        self.encode_emb = None
        self.decode_pre_emb = None
        self.decode_post_emb = None
        self.thought_vectors = None
        self.target_cross_entropy_losses = []
        self.target_cross_entropy_loss_weights = []
        self.total_loss = None

    def build_inputs(self):
        if False:
            while True:
                i = 10
        'Builds the ops for reading input data.\n\n    Outputs:\n      self.encode_ids\n      self.decode_pre_ids\n      self.decode_post_ids\n      self.encode_mask\n      self.decode_pre_mask\n      self.decode_post_mask\n    '
        if self.mode == 'encode':
            encode_ids = None
            decode_pre_ids = None
            decode_post_ids = None
            encode_mask = tf.placeholder(tf.int8, (None, None), name='encode_mask')
            decode_pre_mask = None
            decode_post_mask = None
        else:
            input_queue = input_ops.prefetch_input_data(self.reader, self.config.input_file_pattern, shuffle=self.config.shuffle_input_data, capacity=self.config.input_queue_capacity, num_reader_threads=self.config.num_input_reader_threads)
            serialized = input_queue.dequeue_many(self.config.batch_size)
            (encode, decode_pre, decode_post) = input_ops.parse_example_batch(serialized)
            encode_ids = encode.ids
            decode_pre_ids = decode_pre.ids
            decode_post_ids = decode_post.ids
            encode_mask = encode.mask
            decode_pre_mask = decode_pre.mask
            decode_post_mask = decode_post.mask
        self.encode_ids = encode_ids
        self.decode_pre_ids = decode_pre_ids
        self.decode_post_ids = decode_post_ids
        self.encode_mask = encode_mask
        self.decode_pre_mask = decode_pre_mask
        self.decode_post_mask = decode_post_mask

    def build_word_embeddings(self):
        if False:
            while True:
                i = 10
        'Builds the word embeddings.\n\n    Inputs:\n      self.encode_ids\n      self.decode_pre_ids\n      self.decode_post_ids\n\n    Outputs:\n      self.encode_emb\n      self.decode_pre_emb\n      self.decode_post_emb\n    '
        if self.mode == 'encode':
            encode_emb = tf.placeholder(tf.float32, (None, None, self.config.word_embedding_dim), 'encode_emb')
            decode_pre_emb = None
            decode_post_emb = None
        else:
            word_emb = tf.get_variable(name='word_embedding', shape=[self.config.vocab_size, self.config.word_embedding_dim], initializer=self.uniform_initializer)
            encode_emb = tf.nn.embedding_lookup(word_emb, self.encode_ids)
            decode_pre_emb = tf.nn.embedding_lookup(word_emb, self.decode_pre_ids)
            decode_post_emb = tf.nn.embedding_lookup(word_emb, self.decode_post_ids)
        self.encode_emb = encode_emb
        self.decode_pre_emb = decode_pre_emb
        self.decode_post_emb = decode_post_emb

    def _initialize_gru_cell(self, num_units):
        if False:
            print('Hello World!')
        'Initializes a GRU cell.\n\n    The Variables of the GRU cell are initialized in a way that exactly matches\n    the skip-thoughts paper: recurrent weights are initialized from random\n    orthonormal matrices and non-recurrent weights are initialized from random\n    uniform matrices.\n\n    Args:\n      num_units: Number of output units.\n\n    Returns:\n      cell: An instance of RNNCell with variable initializers that match the\n        skip-thoughts paper.\n    '
        return gru_cell.LayerNormGRUCell(num_units, w_initializer=self.uniform_initializer, u_initializer=random_orthonormal_initializer, b_initializer=tf.constant_initializer(0.0))

    def build_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        'Builds the sentence encoder.\n\n    Inputs:\n      self.encode_emb\n      self.encode_mask\n\n    Outputs:\n      self.thought_vectors\n\n    Raises:\n      ValueError: if config.bidirectional_encoder is True and config.encoder_dim\n        is odd.\n    '
        with tf.variable_scope('encoder') as scope:
            length = tf.to_int32(tf.reduce_sum(self.encode_mask, 1), name='length')
            if self.config.bidirectional_encoder:
                if self.config.encoder_dim % 2:
                    raise ValueError('encoder_dim must be even when using a bidirectional encoder.')
                num_units = self.config.encoder_dim // 2
                cell_fw = self._initialize_gru_cell(num_units)
                cell_bw = self._initialize_gru_cell(num_units)
                (_, states) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.encode_emb, sequence_length=length, dtype=tf.float32, scope=scope)
                thought_vectors = tf.concat(states, 1, name='thought_vectors')
            else:
                cell = self._initialize_gru_cell(self.config.encoder_dim)
                (_, state) = tf.nn.dynamic_rnn(cell=cell, inputs=self.encode_emb, sequence_length=length, dtype=tf.float32, scope=scope)
                thought_vectors = tf.identity(state, name='thought_vectors')
        self.thought_vectors = thought_vectors

    def _build_decoder(self, name, embeddings, targets, mask, initial_state, reuse_logits):
        if False:
            i = 10
            return i + 15
        'Builds a sentence decoder.\n\n    Args:\n      name: Decoder name.\n      embeddings: Batch of sentences to decode; a float32 Tensor with shape\n        [batch_size, padded_length, emb_dim].\n      targets: Batch of target word ids; an int64 Tensor with shape\n        [batch_size, padded_length].\n      mask: A 0/1 Tensor with shape [batch_size, padded_length].\n      initial_state: Initial state of the GRU. A float32 Tensor with shape\n        [batch_size, num_gru_cells].\n      reuse_logits: Whether to reuse the logits weights.\n    '
        cell = self._initialize_gru_cell(self.config.encoder_dim)
        with tf.variable_scope(name) as scope:
            decoder_input = tf.pad(embeddings[:, :-1, :], [[0, 0], [1, 0], [0, 0]], name='input')
            length = tf.reduce_sum(mask, 1, name='length')
            (decoder_output, _) = tf.nn.dynamic_rnn(cell=cell, inputs=decoder_input, sequence_length=length, initial_state=initial_state, scope=scope)
        decoder_output = tf.reshape(decoder_output, [-1, self.config.encoder_dim])
        targets = tf.reshape(targets, [-1])
        weights = tf.to_float(tf.reshape(mask, [-1]))
        with tf.variable_scope('logits', reuse=reuse_logits) as scope:
            logits = tf.contrib.layers.fully_connected(inputs=decoder_output, num_outputs=self.config.vocab_size, activation_fn=None, weights_initializer=self.uniform_initializer, scope=scope)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        batch_loss = tf.reduce_sum(losses * weights)
        tf.losses.add_loss(batch_loss)
        tf.summary.scalar('losses/' + name, batch_loss)
        self.target_cross_entropy_losses.append(losses)
        self.target_cross_entropy_loss_weights.append(weights)

    def build_decoders(self):
        if False:
            for i in range(10):
                print('nop')
        'Builds the sentence decoders.\n\n    Inputs:\n      self.decode_pre_emb\n      self.decode_post_emb\n      self.decode_pre_ids\n      self.decode_post_ids\n      self.decode_pre_mask\n      self.decode_post_mask\n      self.thought_vectors\n\n    Outputs:\n      self.target_cross_entropy_losses\n      self.target_cross_entropy_loss_weights\n    '
        if self.mode != 'encode':
            self._build_decoder('decoder_pre', self.decode_pre_emb, self.decode_pre_ids, self.decode_pre_mask, self.thought_vectors, False)
            self._build_decoder('decoder_post', self.decode_post_emb, self.decode_post_ids, self.decode_post_mask, self.thought_vectors, True)

    def build_loss(self):
        if False:
            print('Hello World!')
        'Builds the loss Tensor.\n\n    Outputs:\n      self.total_loss\n    '
        if self.mode != 'encode':
            total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('losses/total', total_loss)
            self.total_loss = total_loss

    def build_global_step(self):
        if False:
            for i in range(10):
                print('nop')
        'Builds the global step Tensor.\n\n    Outputs:\n      self.global_step\n    '
        self.global_step = tf.contrib.framework.create_global_step()

    def build(self):
        if False:
            return 10
        'Creates all ops for training, evaluation or encoding.'
        self.build_inputs()
        self.build_word_embeddings()
        self.build_encoder()
        self.build_decoders()
        self.build_loss()
        self.build_global_step()