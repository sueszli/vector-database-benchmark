"""Class for encoding text using a trained SkipThoughtsModel.

Example usage:
  g = tf.Graph()
  with g.as_default():
    encoder = SkipThoughtsEncoder(embeddings)
    restore_fn = encoder.build_graph_from_config(model_config, checkpoint_path)

  with tf.Session(graph=g) as sess:
    restore_fn(sess)
    skip_thought_vectors = encoder.encode(sess, data)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import nltk
import nltk.tokenize
import numpy as np
import tensorflow as tf
from skip_thoughts import skip_thoughts_model
from skip_thoughts.data import special_words

def _pad(seq, target_len):
    if False:
        i = 10
        return i + 15
    'Pads a sequence of word embeddings up to the target length.\n\n  Args:\n    seq: Sequence of word embeddings.\n    target_len: Desired padded sequence length.\n\n  Returns:\n    embeddings: Input sequence padded with zero embeddings up to the target\n      length.\n    mask: A 0/1 vector with zeros corresponding to padded embeddings.\n\n  Raises:\n    ValueError: If len(seq) is not in the interval (0, target_len].\n  '
    seq_len = len(seq)
    if seq_len <= 0 or seq_len > target_len:
        raise ValueError('Expected 0 < len(seq) <= %d, got %d' % (target_len, seq_len))
    emb_dim = seq[0].shape[0]
    padded_seq = np.zeros(shape=(target_len, emb_dim), dtype=seq[0].dtype)
    mask = np.zeros(shape=(target_len,), dtype=np.int8)
    for i in range(seq_len):
        padded_seq[i] = seq[i]
        mask[i] = 1
    return (padded_seq, mask)

def _batch_and_pad(sequences):
    if False:
        print('Hello World!')
    'Batches and pads sequences of word embeddings into a 2D array.\n\n  Args:\n    sequences: A list of batch_size sequences of word embeddings.\n\n  Returns:\n    embeddings: A numpy array with shape [batch_size, padded_length, emb_dim].\n    mask: A numpy 0/1 array with shape [batch_size, padded_length] with zeros\n      corresponding to padded elements.\n  '
    batch_embeddings = []
    batch_mask = []
    batch_len = max([len(seq) for seq in sequences])
    for seq in sequences:
        (embeddings, mask) = _pad(seq, batch_len)
        batch_embeddings.append(embeddings)
        batch_mask.append(mask)
    return (np.array(batch_embeddings), np.array(batch_mask))

class SkipThoughtsEncoder(object):
    """Skip-thoughts sentence encoder."""

    def __init__(self, embeddings):
        if False:
            i = 10
            return i + 15
        'Initializes the encoder.\n\n    Args:\n      embeddings: Dictionary of word to embedding vector (1D numpy array).\n    '
        self._sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self._embeddings = embeddings

    def _create_restore_fn(self, checkpoint_path, saver):
        if False:
            i = 10
            return i + 15
        'Creates a function that restores a model from checkpoint.\n\n    Args:\n      checkpoint_path: Checkpoint file or a directory containing a checkpoint\n        file.\n      saver: Saver for restoring variables from the checkpoint file.\n\n    Returns:\n      restore_fn: A function such that restore_fn(sess) loads model variables\n        from the checkpoint file.\n\n    Raises:\n      ValueError: If checkpoint_path does not refer to a checkpoint file or a\n        directory containing a checkpoint file.\n    '
        if tf.gfile.IsDirectory(checkpoint_path):
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
            if not latest_checkpoint:
                raise ValueError('No checkpoint file found in: %s' % checkpoint_path)
            checkpoint_path = latest_checkpoint

        def _restore_fn(sess):
            if False:
                return 10
            tf.logging.info('Loading model from checkpoint: %s', checkpoint_path)
            saver.restore(sess, checkpoint_path)
            tf.logging.info('Successfully loaded checkpoint: %s', os.path.basename(checkpoint_path))
        return _restore_fn

    def build_graph_from_config(self, model_config, checkpoint_path):
        if False:
            return 10
        'Builds the inference graph from a configuration object.\n\n    Args:\n      model_config: Object containing configuration for building the model.\n      checkpoint_path: Checkpoint file or a directory containing a checkpoint\n        file.\n\n    Returns:\n      restore_fn: A function such that restore_fn(sess) loads model variables\n        from the checkpoint file.\n    '
        tf.logging.info('Building model.')
        model = skip_thoughts_model.SkipThoughtsModel(model_config, mode='encode')
        model.build()
        saver = tf.train.Saver()
        return self._create_restore_fn(checkpoint_path, saver)

    def build_graph_from_proto(self, graph_def_file, saver_def_file, checkpoint_path):
        if False:
            print('Hello World!')
        'Builds the inference graph from serialized GraphDef and SaverDef protos.\n\n    Args:\n      graph_def_file: File containing a serialized GraphDef proto.\n      saver_def_file: File containing a serialized SaverDef proto.\n      checkpoint_path: Checkpoint file or a directory containing a checkpoint\n        file.\n\n    Returns:\n      restore_fn: A function such that restore_fn(sess) loads model variables\n        from the checkpoint file.\n    '
        tf.logging.info('Loading GraphDef from file: %s', graph_def_file)
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(graph_def_file, 'rb') as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.logging.info('Loading SaverDef from file: %s', saver_def_file)
        saver_def = tf.train.SaverDef()
        with tf.gfile.FastGFile(saver_def_file, 'rb') as f:
            saver_def.ParseFromString(f.read())
        saver = tf.train.Saver(saver_def=saver_def)
        return self._create_restore_fn(checkpoint_path, saver)

    def _tokenize(self, item):
        if False:
            i = 10
            return i + 15
        'Tokenizes an input string into a list of words.'
        tokenized = []
        for s in self._sentence_detector.tokenize(item):
            tokenized.extend(nltk.tokenize.word_tokenize(s))
        return tokenized

    def _word_to_embedding(self, w):
        if False:
            while True:
                i = 10
        'Returns the embedding of a word.'
        return self._embeddings.get(w, self._embeddings[special_words.UNK])

    def _preprocess(self, data, use_eos):
        if False:
            i = 10
            return i + 15
        'Preprocesses text for the encoder.\n\n    Args:\n      data: A list of input strings.\n      use_eos: Whether to append the end-of-sentence word to each sentence.\n\n    Returns:\n      embeddings: A list of word embedding sequences corresponding to the input\n        strings.\n    '
        preprocessed_data = []
        for item in data:
            tokenized = self._tokenize(item)
            if use_eos:
                tokenized.append(special_words.EOS)
            preprocessed_data.append([self._word_to_embedding(w) for w in tokenized])
        return preprocessed_data

    def encode(self, sess, data, use_norm=True, verbose=True, batch_size=128, use_eos=False):
        if False:
            for i in range(10):
                print('nop')
        "Encodes a sequence of sentences as skip-thought vectors.\n\n    Args:\n      sess: TensorFlow Session.\n      data: A list of input strings.\n      use_norm: Whether to normalize skip-thought vectors to unit L2 norm.\n      verbose: Whether to log every batch.\n      batch_size: Batch size for the encoder.\n      use_eos: Whether to append the end-of-sentence word to each input\n        sentence.\n\n    Returns:\n      thought_vectors: A list of numpy arrays corresponding to the skip-thought\n        encodings of sentences in 'data'.\n    "
        data = self._preprocess(data, use_eos)
        thought_vectors = []
        batch_indices = np.arange(0, len(data), batch_size)
        for (batch, start_index) in enumerate(batch_indices):
            if verbose:
                tf.logging.info('Batch %d / %d.', batch, len(batch_indices))
            (embeddings, mask) = _batch_and_pad(data[start_index:start_index + batch_size])
            feed_dict = {'encode_emb:0': embeddings, 'encode_mask:0': mask}
            thought_vectors.extend(sess.run('encoder/thought_vectors:0', feed_dict=feed_dict))
        if use_norm:
            thought_vectors = [v / np.linalg.norm(v) for v in thought_vectors]
        return thought_vectors