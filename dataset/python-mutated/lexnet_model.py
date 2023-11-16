"""The integrated LexNET model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import lexnet_common
import numpy as np
import tensorflow as tf
from six.moves import xrange

class LexNETModel(object):
    """The LexNET model for classifying relationships between noun compounds."""

    @classmethod
    def default_hparams(cls):
        if False:
            return 10
        'Returns the default hyper-parameters.'
        return tf.contrib.training.HParams(batch_size=10, num_classes=37, num_epochs=30, input_keep_prob=0.9, input='integrated', learn_relata=False, corpus='wiki_gigawords', random_seed=133, relata_embeddings_file='glove/glove.6B.300d.bin', nc_embeddings_file='nc_glove/vecs.6B.300d.bin', path_embeddings_file='path_embeddings/tratz/fine_grained/wiki', hidden_layers=1, path_dim=60)

    def __init__(self, hparams, relata_embeddings, path_embeddings, nc_embeddings, path_to_index):
        if False:
            i = 10
            return i + 15
        'Initialize the LexNET classifier.\n\n    Args:\n      hparams: the hyper-parameters.\n      relata_embeddings: word embeddings for the distributional component.\n      path_embeddings: embeddings for the paths.\n      nc_embeddings: noun compound embeddings.\n      path_to_index: a mapping from string path to an index in the path\n      embeddings matrix.\n    '
        self.hparams = hparams
        self.path_embeddings = path_embeddings
        self.relata_embeddings = relata_embeddings
        self.nc_embeddings = nc_embeddings
        (self.vocab_size, self.relata_dim) = (0, 0)
        self.path_to_index = None
        self.path_dim = 0
        if hparams.random_seed > 0:
            tf.set_random_seed(hparams.random_seed)
        if self.hparams.input in ['dist', 'dist-nc', 'integrated', 'integrated-nc']:
            (self.vocab_size, self.relata_dim) = self.relata_embeddings.shape
        if self.hparams.input in ['path', 'integrated', 'integrated-nc']:
            self.path_to_index = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(path_to_index.keys()), tf.constant(path_to_index.values()), key_dtype=tf.string, value_dtype=tf.int32), 0)
            self.path_dim = self.path_embeddings.shape[1]
        self.__create_computation_graph__()

    def __create_computation_graph__(self):
        if False:
            while True:
                i = 10
        'Initialize the model and define the graph.'
        network_input = 0
        if self.hparams.input in ['dist', 'dist-nc', 'integrated', 'integrated-nc']:
            network_input += 2 * self.relata_dim
            self.relata_lookup = tf.get_variable('relata_lookup', initializer=self.relata_embeddings, dtype=tf.float32, trainable=self.hparams.learn_relata)
        if self.hparams.input in ['path', 'integrated', 'integrated-nc']:
            network_input += self.path_dim
            self.path_initial_value_t = tf.placeholder(tf.float32, None)
            self.path_lookup = tf.get_variable(name='path_lookup', dtype=tf.float32, trainable=False, shape=self.path_embeddings.shape)
            self.initialize_path_op = tf.assign(self.path_lookup, self.path_initial_value_t, validate_shape=False)
        if self.hparams.input in ['dist-nc', 'integrated-nc']:
            network_input += self.relata_dim
            self.nc_initial_value_t = tf.placeholder(tf.float32, None)
            self.nc_lookup = tf.get_variable(name='nc_lookup', dtype=tf.float32, trainable=False, shape=self.nc_embeddings.shape)
            self.initialize_nc_op = tf.assign(self.nc_lookup, self.nc_initial_value_t, validate_shape=False)
        hidden_dim = network_input // 2
        if self.hparams.hidden_layers == 0:
            self.weights1 = tf.get_variable('W1', shape=[network_input, self.hparams.num_classes], dtype=tf.float32)
            self.bias1 = tf.get_variable('b1', shape=[self.hparams.num_classes], dtype=tf.float32)
        elif self.hparams.hidden_layers == 1:
            self.weights1 = tf.get_variable('W1', shape=[network_input, hidden_dim], dtype=tf.float32)
            self.bias1 = tf.get_variable('b1', shape=[hidden_dim], dtype=tf.float32)
            self.weights2 = tf.get_variable('W2', shape=[hidden_dim, self.hparams.num_classes], dtype=tf.float32)
            self.bias2 = tf.get_variable('b2', shape=[self.hparams.num_classes], dtype=tf.float32)
        else:
            raise ValueError('Only 0 or 1 hidden layers are supported')
        self.instances = tf.placeholder(dtype=tf.string, shape=[self.hparams.batch_size])
        (self.x_embedding_id, self.y_embedding_id, self.nc_embedding_id, self.path_embedding_id, self.path_counts, self.labels) = parse_tensorflow_examples(self.instances, self.hparams.batch_size, self.path_to_index)
        self.__mlp__()
        self.instances_to_load = tf.placeholder(dtype=tf.string, shape=[None])
        self.labels_to_load = lexnet_common.load_all_labels(self.instances_to_load)
        self.pairs_to_load = lexnet_common.load_all_pairs(self.instances_to_load)

    def load_labels(self, session, instances):
        if False:
            while True:
                i = 10
        'Loads the labels for these instances.\n\n    Args:\n      session: The current TensorFlow session,\n      instances: The instances for which to load the labels.\n\n    Returns:\n      the labels of these instances.\n    '
        return session.run(self.labels_to_load, feed_dict={self.instances_to_load: instances})

    def load_pairs(self, session, instances):
        if False:
            i = 10
            return i + 15
        'Loads the word pairs for these instances.\n\n    Args:\n      session: The current TensorFlow session,\n      instances: The instances for which to load the labels.\n\n    Returns:\n      the word pairs of these instances.\n    '
        word_pairs = session.run(self.pairs_to_load, feed_dict={self.instances_to_load: instances})
        return [pair[0].split('::') for pair in word_pairs]

    def __train_single_batch__(self, session, batch_instances):
        if False:
            i = 10
            return i + 15
        'Train a single batch.\n\n    Args:\n      session: The current TensorFlow session.\n      batch_instances: TensorFlow examples containing the training intances\n\n    Returns:\n      The cost for the current batch.\n    '
        (cost, _) = session.run([self.cost, self.train_op], feed_dict={self.instances: batch_instances})
        return cost

    def fit(self, session, inputs, on_epoch_completed, val_instances, val_labels, save_path):
        if False:
            i = 10
            return i + 15
        'Train the model.\n\n    Args:\n      session: The current TensorFlow session.\n      inputs:\n      on_epoch_completed: A method to call after each epoch.\n      val_instances: The validation set instances (evaluation between epochs).\n      val_labels: The validation set labels (for evaluation between epochs).\n      save_path: Where to save the model.\n    '
        for epoch in range(self.hparams.num_epochs):
            losses = []
            epoch_indices = list(np.random.permutation(len(inputs)))
            mod = len(epoch_indices) % self.hparams.batch_size
            if mod > 0:
                epoch_indices.extend([np.random.randint(0, high=len(inputs))] * mod)
            n_batches = len(epoch_indices) // self.hparams.batch_size
            for minibatch in range(n_batches):
                batch_indices = epoch_indices[minibatch * self.hparams.batch_size:(minibatch + 1) * self.hparams.batch_size]
                batch_instances = [inputs[i] for i in batch_indices]
                loss = self.__train_single_batch__(session, batch_instances)
                losses.append(loss)
            epoch_loss = np.nanmean(losses)
            if on_epoch_completed:
                should_stop = on_epoch_completed(self, session, epoch, epoch_loss, val_instances, val_labels, save_path)
                if should_stop:
                    print('Stopping training after %d epochs.' % epoch)
                    return

    def predict(self, session, inputs):
        if False:
            while True:
                i = 10
        'Predict the classification of the test set.\n\n    Args:\n      session: The current TensorFlow session.\n      inputs: the train paths, x, y and/or nc vectors\n\n    Returns:\n      The test predictions.\n    '
        (predictions, _) = zip(*self.predict_with_score(session, inputs))
        return np.array(predictions)

    def predict_with_score(self, session, inputs):
        if False:
            print('Hello World!')
        'Predict the classification of the test set.\n\n    Args:\n      session: The current TensorFlow session.\n      inputs: the test paths, x, y and/or nc vectors\n\n    Returns:\n      The test predictions along with their scores.\n    '
        test_pred = [0] * len(inputs)
        for chunk in xrange(0, len(test_pred), self.hparams.batch_size):
            batch_indices = list(range(chunk, min(chunk + self.hparams.batch_size, len(test_pred))))
            if len(batch_indices) < self.hparams.batch_size:
                batch_indices += [0] * (self.hparams.batch_size - len(batch_indices))
            batch_instances = [inputs[i] for i in batch_indices]
            (predictions, scores) = session.run([self.predictions, self.scores], feed_dict={self.instances: batch_instances})
            for (index_in_batch, index_in_dataset) in enumerate(batch_indices):
                prediction = predictions[index_in_batch]
                score = scores[index_in_batch][prediction]
                test_pred[index_in_dataset] = (prediction, score)
        return test_pred

    def __mlp__(self):
        if False:
            i = 10
            return i + 15
        'Performs the MLP operations.\n\n    Returns: the prediction object to be computed in a Session\n    '
        vec_inputs = []
        if self.hparams.input in ['dist', 'dist-nc', 'integrated', 'integrated-nc']:
            for emb_id in [self.x_embedding_id, self.y_embedding_id]:
                vec_inputs.append(tf.nn.embedding_lookup(self.relata_lookup, emb_id))
        if self.hparams.input in ['dist-nc', 'integrated-nc']:
            vec = tf.nn.embedding_lookup(self.nc_lookup, self.nc_embedding_id)
            vec_inputs.append(vec)
        if self.hparams.input in ['path', 'integrated', 'integrated-nc']:
            self.path_embeddings = tf.nn.embedding_lookup(self.path_lookup, self.path_embedding_id)
            self.path_freq = tf.tile(tf.expand_dims(self.path_counts, -1), [1, 1, self.path_dim])
            self.weighted = tf.multiply(self.path_freq, self.path_embeddings)
            self.pair_path_embeddings = tf.reduce_sum(self.weighted, 1)
            self.num_paths = tf.clip_by_value(tf.reduce_sum(self.path_counts, 1), 1, np.inf)
            self.num_paths = tf.tile(tf.expand_dims(self.num_paths, -1), [1, self.path_dim])
            self.pair_path_embeddings = tf.div(self.pair_path_embeddings, self.num_paths)
            vec_inputs.append(self.pair_path_embeddings)
        self.input_vec = tf.nn.dropout(tf.concat(vec_inputs, 1), keep_prob=self.hparams.input_keep_prob)
        h = tf.matmul(self.input_vec, self.weights1)
        self.output = h
        if self.hparams.hidden_layers == 1:
            self.output = tf.matmul(tf.nn.tanh(h), self.weights2)
        self.scores = self.output
        self.predictions = tf.argmax(self.scores, axis=1)
        self.cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.labels)
        self.cost = tf.reduce_sum(self.cross_entropies, name='cost')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.cost, global_step=self.global_step)

def parse_tensorflow_examples(record, batch_size, path_to_index):
    if False:
        for i in range(10):
            print('nop')
    'Reads TensorFlow examples from a RecordReader.\n\n  Args:\n    record: a record with TensorFlow examples.\n    batch_size: the number of instances in a minibatch\n    path_to_index: mapping from string path to index in the embeddings matrix.\n\n  Returns:\n    The word embeddings IDs, paths and counts\n  '
    features = tf.parse_example(record, {'x_embedding_id': tf.FixedLenFeature([1], dtype=tf.int64), 'y_embedding_id': tf.FixedLenFeature([1], dtype=tf.int64), 'nc_embedding_id': tf.FixedLenFeature([1], dtype=tf.int64), 'reprs': tf.FixedLenSequenceFeature(shape=(), dtype=tf.string, allow_missing=True), 'counts': tf.FixedLenSequenceFeature(shape=(), dtype=tf.int64, allow_missing=True), 'rel_id': tf.FixedLenFeature([1], dtype=tf.int64)})
    x_embedding_id = tf.squeeze(features['x_embedding_id'], [-1])
    y_embedding_id = tf.squeeze(features['y_embedding_id'], [-1])
    nc_embedding_id = tf.squeeze(features['nc_embedding_id'], [-1])
    labels = tf.squeeze(features['rel_id'], [-1])
    path_counts = tf.to_float(tf.reshape(features['counts'], [batch_size, -1]))
    path_embedding_id = None
    if path_to_index:
        path_embedding_id = path_to_index.lookup(features['reprs'])
    return (x_embedding_id, y_embedding_id, nc_embedding_id, path_embedding_id, path_counts, labels)