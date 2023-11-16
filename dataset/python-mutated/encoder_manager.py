"""Manager class for loading and encoding with multiple skip-thoughts models.

If multiple models are loaded at once then the encode() function returns the
concatenation of the outputs of each model.

Example usage:
  manager = EncoderManager()
  manager.load_model(model_config_1, vocabulary_file_1, embedding_matrix_file_1,
                     checkpoint_path_1)
  manager.load_model(model_config_2, vocabulary_file_2, embedding_matrix_file_2,
                     checkpoint_path_2)
  encodings = manager.encode(data)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
import tensorflow as tf
from skip_thoughts import skip_thoughts_encoder

class EncoderManager(object):
    """Manager class for loading and encoding with skip-thoughts models."""

    def __init__(self):
        if False:
            return 10
        self.encoders = []
        self.sessions = []

    def load_model(self, model_config, vocabulary_file, embedding_matrix_file, checkpoint_path):
        if False:
            while True:
                i = 10
        'Loads a skip-thoughts model.\n\n    Args:\n      model_config: Object containing parameters for building the model.\n      vocabulary_file: Path to vocabulary file containing a list of newline-\n        separated words where the word id is the corresponding 0-based index in\n        the file.\n      embedding_matrix_file: Path to a serialized numpy array of shape\n        [vocab_size, embedding_dim].\n      checkpoint_path: SkipThoughtsModel checkpoint file or a directory\n        containing a checkpoint file.\n    '
        tf.logging.info('Reading vocabulary from %s', vocabulary_file)
        with tf.gfile.GFile(vocabulary_file, mode='r') as f:
            lines = list(f.readlines())
        reverse_vocab = [line.decode('utf-8').strip() for line in lines]
        tf.logging.info('Loaded vocabulary with %d words.', len(reverse_vocab))
        tf.logging.info('Loading embedding matrix from %s', embedding_matrix_file)
        embedding_matrix = np.load(embedding_matrix_file)
        tf.logging.info('Loaded embedding matrix with shape %s', embedding_matrix.shape)
        word_embeddings = collections.OrderedDict(zip(reverse_vocab, embedding_matrix))
        g = tf.Graph()
        with g.as_default():
            encoder = skip_thoughts_encoder.SkipThoughtsEncoder(word_embeddings)
            restore_model = encoder.build_graph_from_config(model_config, checkpoint_path)
        sess = tf.Session(graph=g)
        restore_model(sess)
        self.encoders.append(encoder)
        self.sessions.append(sess)

    def encode(self, data, use_norm=True, verbose=False, batch_size=128, use_eos=False):
        if False:
            print('Hello World!')
        "Encodes a sequence of sentences as skip-thought vectors.\n\n    Args:\n      data: A list of input strings.\n      use_norm: If True, normalize output skip-thought vectors to unit L2 norm.\n      verbose: Whether to log every batch.\n      batch_size: Batch size for the RNN encoders.\n      use_eos: If True, append the end-of-sentence word to each input sentence.\n\n    Returns:\n      thought_vectors: A list of numpy arrays corresponding to 'data'.\n\n    Raises:\n      ValueError: If called before calling load_encoder.\n    "
        if not self.encoders:
            raise ValueError('Must call load_model at least once before calling encode.')
        encoded = []
        for (encoder, sess) in zip(self.encoders, self.sessions):
            encoded.append(np.array(encoder.encode(sess, data, use_norm=use_norm, verbose=verbose, batch_size=batch_size, use_eos=use_eos)))
        return np.concatenate(encoded, axis=1)

    def close(self):
        if False:
            return 10
        'Closes the active TensorFlow Sessions.'
        for sess in self.sessions:
            sess.close()