"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from im2txt.ops import image_embedding
from im2txt.ops import image_processing
from im2txt.ops import inputs as input_ops

class ShowAndTellModel(object):
    """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

    def __init__(self, config, mode, train_inception=False):
        if False:
            print('Hello World!')
        'Basic setup.\n\n    Args:\n      config: Object containing configuration parameters.\n      mode: "train", "eval" or "inference".\n      train_inception: Whether the inception submodel variables are trainable.\n    '
        assert mode in ['train', 'eval', 'inference']
        self.config = config
        self.mode = mode
        self.train_inception = train_inception
        self.reader = tf.TFRecordReader()
        self.initializer = tf.random_uniform_initializer(minval=-self.config.initializer_scale, maxval=self.config.initializer_scale)
        self.images = None
        self.input_seqs = None
        self.target_seqs = None
        self.input_mask = None
        self.image_embeddings = None
        self.seq_embeddings = None
        self.total_loss = None
        self.target_cross_entropy_losses = None
        self.target_cross_entropy_loss_weights = None
        self.inception_variables = []
        self.init_fn = None
        self.global_step = None

    def is_training(self):
        if False:
            i = 10
            return i + 15
        'Returns true if the model is built for training mode.'
        return self.mode == 'train'

    def process_image(self, encoded_image, thread_id=0):
        if False:
            print('Hello World!')
        'Decodes and processes an image string.\n\n    Args:\n      encoded_image: A scalar string Tensor; the encoded image.\n      thread_id: Preprocessing thread id used to select the ordering of color\n        distortions.\n\n    Returns:\n      A float32 Tensor of shape [height, width, 3]; the processed image.\n    '
        return image_processing.process_image(encoded_image, is_training=self.is_training(), height=self.config.image_height, width=self.config.image_width, thread_id=thread_id, image_format=self.config.image_format)

    def build_inputs(self):
        if False:
            while True:
                i = 10
        'Input prefetching, preprocessing and batching.\n\n    Outputs:\n      self.images\n      self.input_seqs\n      self.target_seqs (training and eval only)\n      self.input_mask (training and eval only)\n    '
        if self.mode == 'inference':
            image_feed = tf.placeholder(dtype=tf.string, shape=[], name='image_feed')
            input_feed = tf.placeholder(dtype=tf.int64, shape=[None], name='input_feed')
            images = tf.expand_dims(self.process_image(image_feed), 0)
            input_seqs = tf.expand_dims(input_feed, 1)
            target_seqs = None
            input_mask = None
        else:
            input_queue = input_ops.prefetch_input_data(self.reader, self.config.input_file_pattern, is_training=self.is_training(), batch_size=self.config.batch_size, values_per_shard=self.config.values_per_input_shard, input_queue_capacity_factor=self.config.input_queue_capacity_factor, num_reader_threads=self.config.num_input_reader_threads)
            assert self.config.num_preprocess_threads % 2 == 0
            images_and_captions = []
            for thread_id in range(self.config.num_preprocess_threads):
                serialized_sequence_example = input_queue.dequeue()
                (encoded_image, caption) = input_ops.parse_sequence_example(serialized_sequence_example, image_feature=self.config.image_feature_name, caption_feature=self.config.caption_feature_name)
                image = self.process_image(encoded_image, thread_id=thread_id)
                images_and_captions.append([image, caption])
            queue_capacity = 2 * self.config.num_preprocess_threads * self.config.batch_size
            (images, input_seqs, target_seqs, input_mask) = input_ops.batch_with_dynamic_pad(images_and_captions, batch_size=self.config.batch_size, queue_capacity=queue_capacity)
        self.images = images
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.input_mask = input_mask

    def build_image_embeddings(self):
        if False:
            return 10
        'Builds the image model subgraph and generates image embeddings.\n\n    Inputs:\n      self.images\n\n    Outputs:\n      self.image_embeddings\n    '
        inception_output = image_embedding.inception_v3(self.images, trainable=self.train_inception, is_training=self.is_training())
        self.inception_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionV3')
        with tf.variable_scope('image_embedding') as scope:
            image_embeddings = tf.contrib.layers.fully_connected(inputs=inception_output, num_outputs=self.config.embedding_size, activation_fn=None, weights_initializer=self.initializer, biases_initializer=None, scope=scope)
        tf.constant(self.config.embedding_size, name='embedding_size')
        self.image_embeddings = image_embeddings

    def build_seq_embeddings(self):
        if False:
            while True:
                i = 10
        'Builds the input sequence embeddings.\n\n    Inputs:\n      self.input_seqs\n\n    Outputs:\n      self.seq_embeddings\n    '
        with tf.variable_scope('seq_embedding'), tf.device('/cpu:0'):
            embedding_map = tf.get_variable(name='map', shape=[self.config.vocab_size, self.config.embedding_size], initializer=self.initializer)
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)
        self.seq_embeddings = seq_embeddings

    def build_model(self):
        if False:
            i = 10
            return i + 15
        'Builds the model.\n\n    Inputs:\n      self.image_embeddings\n      self.seq_embeddings\n      self.target_seqs (training and eval only)\n      self.input_mask (training and eval only)\n\n    Outputs:\n      self.total_loss (training and eval only)\n      self.target_cross_entropy_losses (training and eval only)\n      self.target_cross_entropy_loss_weights (training and eval only)\n    '
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.config.num_lstm_units, state_is_tuple=True)
        if self.mode == 'train':
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.config.lstm_dropout_keep_prob, output_keep_prob=self.config.lstm_dropout_keep_prob)
        with tf.variable_scope('lstm', initializer=self.initializer) as lstm_scope:
            zero_state = lstm_cell.zero_state(batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
            (_, initial_state) = lstm_cell(self.image_embeddings, zero_state)
            lstm_scope.reuse_variables()
            if self.mode == 'inference':
                tf.concat(axis=1, values=initial_state, name='initial_state')
                state_feed = tf.placeholder(dtype=tf.float32, shape=[None, sum(lstm_cell.state_size)], name='state_feed')
                state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)
                (lstm_outputs, state_tuple) = lstm_cell(inputs=tf.squeeze(self.seq_embeddings, axis=[1]), state=state_tuple)
                tf.concat(axis=1, values=state_tuple, name='state')
            else:
                sequence_length = tf.reduce_sum(self.input_mask, 1)
                (lstm_outputs, _) = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=self.seq_embeddings, sequence_length=sequence_length, initial_state=initial_state, dtype=tf.float32, scope=lstm_scope)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])
        with tf.variable_scope('logits') as logits_scope:
            logits = tf.contrib.layers.fully_connected(inputs=lstm_outputs, num_outputs=self.config.vocab_size, activation_fn=None, weights_initializer=self.initializer, scope=logits_scope)
        if self.mode == 'inference':
            tf.nn.softmax(logits, name='softmax')
        else:
            targets = tf.reshape(self.target_seqs, [-1])
            weights = tf.to_float(tf.reshape(self.input_mask, [-1]))
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
            batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)), tf.reduce_sum(weights), name='batch_loss')
            tf.losses.add_loss(batch_loss)
            total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('losses/batch_loss', batch_loss)
            tf.summary.scalar('losses/total_loss', total_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram('parameters/' + var.op.name, var)
            self.total_loss = total_loss
            self.target_cross_entropy_losses = losses
            self.target_cross_entropy_loss_weights = weights

    def setup_inception_initializer(self):
        if False:
            for i in range(10):
                print('nop')
        'Sets up the function to restore inception variables from checkpoint.'
        if self.mode != 'inference':
            saver = tf.train.Saver(self.inception_variables)

            def restore_fn(sess):
                if False:
                    return 10
                tf.logging.info('Restoring Inception variables from checkpoint file %s', self.config.inception_checkpoint_file)
                saver.restore(sess, self.config.inception_checkpoint_file)
            self.init_fn = restore_fn

    def setup_global_step(self):
        if False:
            return 10
        'Sets up the global step Tensor.'
        global_step = tf.Variable(initial_value=0, name='global_step', trainable=False, collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step = global_step

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        'Creates all ops for training and evaluation.'
        self.build_inputs()
        self.build_image_embeddings()
        self.build_seq_embeddings()
        self.build_model()
        self.setup_inception_initializer()
        self.setup_global_step()