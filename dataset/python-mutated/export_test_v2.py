"""Tests for text embedding exporting tool v2."""
import logging
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from examples.text_embeddings_v2 import export_v2
_MOCK_EMBEDDING = '\n'.join(['cat 1.11 2.56 3.45', 'dog 1 2 3', 'mouse 0.5 0.1 0.6'])

class ExportTokenEmbeddingTest(tf.test.TestCase):
    """Test for text embedding exporter."""

    def setUp(self):
        if False:
            while True:
                i = 10
        self._embedding_file_path = os.path.join(self.get_temp_dir(), 'mock_embedding_file.txt')
        with tf.io.gfile.GFile(self._embedding_file_path, mode='w') as f:
            f.write(_MOCK_EMBEDDING)

    def testEmbeddingLoaded(self):
        if False:
            for i in range(10):
                print('nop')
        (vocabulary, embeddings) = export_v2.load(self._embedding_file_path, export_v2.parse_line, num_lines_to_ignore=0, num_lines_to_use=None)
        self.assertEqual((3,), np.shape(vocabulary))
        self.assertEqual((3, 3), np.shape(embeddings))

    def testExportTextEmbeddingModule(self):
        if False:
            i = 10
            return i + 15
        export_v2.export_module_from_file(embedding_file=self._embedding_file_path, export_path=self.get_temp_dir(), num_oov_buckets=1, num_lines_to_ignore=0, num_lines_to_use=None)
        hub_module = hub.load(self.get_temp_dir())
        tokens = tf.constant(['cat', 'cat cat', 'lizard. dog', 'cat? dog', ''])
        embeddings = hub_module(tokens)
        self.assertAllClose(embeddings.numpy(), [[1.11, 2.56, 3.45], [1.57, 3.62, 4.88], [0.7, 1.41, 2.12], [1.49, 3.22, 4.56], [0.0, 0.0, 0.0]], rtol=0.02)

    def testEmptyInput(self):
        if False:
            i = 10
            return i + 15
        export_v2.export_module_from_file(embedding_file=self._embedding_file_path, export_path=self.get_temp_dir(), num_oov_buckets=1, num_lines_to_ignore=0, num_lines_to_use=None)
        hub_module = hub.load(self.get_temp_dir())
        tokens = tf.constant(['', '', ''])
        embeddings = hub_module(tokens)
        self.assertAllClose(embeddings.numpy(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], rtol=0.02)

    def testEmptyLeading(self):
        if False:
            for i in range(10):
                print('nop')
        export_v2.export_module_from_file(embedding_file=self._embedding_file_path, export_path=self.get_temp_dir(), num_oov_buckets=1, num_lines_to_ignore=0, num_lines_to_use=None)
        hub_module = hub.load(self.get_temp_dir())
        tokens = tf.constant(['', 'cat dog'])
        embeddings = hub_module(tokens)
        self.assertAllClose(embeddings.numpy(), [[0.0, 0.0, 0.0], [1.49, 3.22, 4.56]], rtol=0.02)

    def testNumLinesIgnore(self):
        if False:
            print('Hello World!')
        export_v2.export_module_from_file(embedding_file=self._embedding_file_path, export_path=self.get_temp_dir(), num_oov_buckets=1, num_lines_to_ignore=1, num_lines_to_use=None)
        hub_module = hub.load(self.get_temp_dir())
        tokens = tf.constant(['cat', 'dog', 'mouse'])
        embeddings = hub_module(tokens)
        self.assertAllClose(embeddings.numpy(), [[0.0, 0.0, 0.0], [1, 2, 3], [0.5, 0.1, 0.6]], rtol=0.02)

    def testNumLinesUse(self):
        if False:
            for i in range(10):
                print('nop')
        export_v2.export_module_from_file(embedding_file=self._embedding_file_path, export_path=self.get_temp_dir(), num_oov_buckets=1, num_lines_to_ignore=0, num_lines_to_use=2)
        hub_module = hub.load(self.get_temp_dir())
        tokens = tf.constant(['cat', 'dog', 'mouse'])
        embeddings = hub_module(tokens)
        self.assertAllClose(embeddings.numpy(), [[1.1, 2.56, 3.45], [1, 2, 3], [0, 0, 0]], rtol=0.02)
if __name__ == '__main__':
    if tf.executing_eagerly():
        logging.info('Using TF version: %s', tf.__version__)
        tf.test.main()
    else:
        logging.warning('Skipping running tests for TF Version: %s', tf.__version__)