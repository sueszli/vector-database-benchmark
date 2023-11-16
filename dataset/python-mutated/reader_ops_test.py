"""Tests for reader_ops."""
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from syntaxnet import dictionary_pb2
from syntaxnet import graph_builder
from syntaxnet import sparse_pb2
from syntaxnet import test_flags
from syntaxnet.ops import gen_parser_ops

class ParsingReaderOpsTest(tf.test.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        initial_task_context = os.path.join(test_flags.source_root(), 'syntaxnet/testdata/context.pbtxt')
        self._task_context = os.path.join(test_flags.temp_dir(), 'context.pbtxt')
        with open(initial_task_context, 'r') as fin:
            with open(self._task_context, 'w') as fout:
                fout.write(fin.read().replace('SRCDIR', test_flags.source_root()).replace('OUTPATH', test_flags.temp_dir()))
        with self.test_session() as sess:
            gen_parser_ops.lexicon_builder(task_context=self._task_context, corpus_name='training-corpus').run()
            (self._num_features, self._num_feature_ids, _, self._num_actions) = sess.run(gen_parser_ops.feature_size(task_context=self._task_context, arg_prefix='brain_parser'))

    def GetMaxId(self, sparse_features):
        if False:
            return 10
        max_id = 0
        for x in sparse_features:
            for y in x:
                f = sparse_pb2.SparseFeatures()
                f.ParseFromString(y)
                for i in f.id:
                    max_id = max(i, max_id)
        return max_id

    def testParsingReaderOp(self):
        if False:
            i = 10
            return i + 15
        num_steps_a = 0
        num_actions = 0
        num_word_ids = 0
        num_tag_ids = 0
        num_label_ids = 0
        batch_size = 10
        with self.test_session() as sess:
            ((words, tags, labels), epochs, gold_actions) = gen_parser_ops.gold_parse_reader(self._task_context, 3, batch_size, corpus_name='training-corpus')
            while True:
                (tf_gold_actions, tf_epochs, tf_words, tf_tags, tf_labels) = sess.run([gold_actions, epochs, words, tags, labels])
                num_steps_a += 1
                num_actions = max(num_actions, max(tf_gold_actions) + 1)
                num_word_ids = max(num_word_ids, self.GetMaxId(tf_words) + 1)
                num_tag_ids = max(num_tag_ids, self.GetMaxId(tf_tags) + 1)
                num_label_ids = max(num_label_ids, self.GetMaxId(tf_labels) + 1)
                self.assertIn(tf_epochs, [0, 1, 2])
                if tf_epochs > 1:
                    break
        num_steps_b = 0
        with self.test_session() as sess:
            num_features = [6, 6, 4]
            num_feature_ids = [num_word_ids, num_tag_ids, num_label_ids]
            embedding_sizes = [8, 8, 8]
            hidden_layer_sizes = [32, 32]
            parser = graph_builder.GreedyParser(num_actions, num_features, num_feature_ids, embedding_sizes, hidden_layer_sizes)
            parser.AddTraining(self._task_context, batch_size, corpus_name='training-corpus')
            sess.run(parser.inits.values())
            while True:
                (tf_epochs, tf_cost, _) = sess.run([parser.training['epochs'], parser.training['cost'], parser.training['train_op']])
                num_steps_b += 1
                self.assertGreaterEqual(tf_cost, 0)
                self.assertIn(tf_epochs, [0, 1, 2])
                if tf_epochs > 1:
                    break
        logging.info('Number of steps in the two runs: %d, %d', num_steps_a, num_steps_b)
        self.assertEqual(num_steps_a, num_steps_b)

    def testParsingReaderOpWhileLoop(self):
        if False:
            for i in range(10):
                print('nop')
        feature_size = 3
        batch_size = 5

        def ParserEndpoints():
            if False:
                return 10
            return gen_parser_ops.gold_parse_reader(self._task_context, feature_size, batch_size, corpus_name='training-corpus')
        with self.test_session() as sess:

            def Condition(epoch, *unused_args):
                if False:
                    while True:
                        i = 10
                return tf.less(epoch, 2)

            def Body(epoch, num_actions, *feature_args):
                if False:
                    return 10
                with epoch.graph.control_dependencies([epoch]):
                    (features, epoch, gold_actions) = ParserEndpoints()
                num_actions = tf.maximum(num_actions, tf.reduce_max(gold_actions, [0], False) + 1)
                feature_ids = []
                for i in range(len(feature_args)):
                    feature_ids.append(features[i])
                return [epoch, num_actions] + feature_ids
            epoch = ParserEndpoints()[-2]
            num_actions = tf.constant(0)
            loop_vars = [epoch, num_actions]
            res = sess.run(tf.while_loop(Condition, Body, loop_vars, shape_invariants=[tf.TensorShape(None)] * 2, parallel_iterations=1))
            logging.info('Result: %s', res)
            self.assertEqual(res[0], 2)

    def _token_embedding(self, token, embedding):
        if False:
            return 10
        e = dictionary_pb2.TokenEmbedding()
        e.token = token
        e.vector.values.extend(embedding)
        return e.SerializeToString()

    def testWordEmbeddingInitializer(self):
        if False:
            return 10
        records_path = os.path.join(test_flags.temp_dir(), 'records1')
        writer = tf.python_io.TFRecordWriter(records_path)
        writer.write(self._token_embedding('.', [1, 2]))
        writer.write(self._token_embedding(',', [3, 4]))
        writer.write(self._token_embedding('the', [5, 6]))
        del writer
        with self.test_session():
            embeddings = gen_parser_ops.word_embedding_initializer(vectors=records_path, task_context=self._task_context).eval()
        self.assertAllClose(np.array([[1.0 / (1 + 4) ** 0.5, 2.0 / (1 + 4) ** 0.5], [3.0 / (9 + 16) ** 0.5, 4.0 / (9 + 16) ** 0.5], [5.0 / (25 + 36) ** 0.5, 6.0 / (25 + 36) ** 0.5]]), embeddings[:3,])

    def testWordEmbeddingInitializerRepeatability(self):
        if False:
            i = 10
            return i + 15
        records_path = os.path.join(test_flags.temp_dir(), 'records2')
        writer = tf.python_io.TFRecordWriter(records_path)
        writer.write(self._token_embedding('.', [1, 2, 3]))
        del writer
        for (seed1, seed2) in [(0, 1), (1, 0), (123, 456)]:
            with tf.Graph().as_default(), self.test_session():
                embeddings1 = gen_parser_ops.word_embedding_initializer(vectors=records_path, task_context=self._task_context, seed=seed1, seed2=seed2)
                embeddings2 = gen_parser_ops.word_embedding_initializer(vectors=records_path, task_context=self._task_context, seed=seed1, seed2=seed2)
                self.assertGreater(tf.shape(embeddings1)[0].eval(), 0)
                self.assertGreater(tf.shape(embeddings2)[0].eval(), 0)
                self.assertEqual(tf.shape(embeddings1)[1].eval(), 3)
                self.assertEqual(tf.shape(embeddings2)[1].eval(), 3)
                self.assertAllEqual(embeddings1.eval(), embeddings2.eval())

    def testWordEmbeddingInitializerFailIfNeitherTaskContextOrVocabulary(self):
        if False:
            while True:
                i = 10
        with self.test_session():
            with self.assertRaises(Exception):
                gen_parser_ops.word_embedding_initializer(vectors='/dev/null').eval()

    def testWordEmbeddingInitializerFailIfBothTaskContextAndVocabulary(self):
        if False:
            i = 10
            return i + 15
        with self.test_session():
            with self.assertRaises(Exception):
                gen_parser_ops.word_embedding_initializer(vectors='/dev/null', task_context='/dev/null', vocabulary='/dev/null').eval()

    def testWordEmbeddingInitializerVocabularyFile(self):
        if False:
            print('Hello World!')
        records_path = os.path.join(test_flags.temp_dir(), 'records3')
        writer = tf.python_io.TFRecordWriter(records_path)
        writer.write(self._token_embedding('a', [1, 2, 3]))
        writer.write(self._token_embedding('b', [2, 3, 4]))
        writer.write(self._token_embedding('c', [3, 4, 5]))
        writer.write(self._token_embedding('d', [4, 5, 6]))
        writer.write(self._token_embedding('e', [5, 6, 7]))
        del writer
        vocabulary_path = os.path.join(test_flags.temp_dir(), 'vocabulary3')
        with open(vocabulary_path, 'w') as vocabulary_file:
            vocabulary_file.write('a\nc\ne\nx\n')
        for cache_vectors_locally in [False, True]:
            for num_special_embeddings in [None, 1, 2, 5]:
                with self.test_session():
                    embeddings = gen_parser_ops.word_embedding_initializer(vectors=records_path, vocabulary=vocabulary_path, cache_vectors_locally=cache_vectors_locally, num_special_embeddings=num_special_embeddings)
                    expected_num_embeddings = 4 + (num_special_embeddings or 3)
                    self.assertAllEqual([expected_num_embeddings, 3], tf.shape(embeddings).eval())
                    norm_a = (1.0 + 4.0 + 9.0) ** 0.5
                    norm_c = (9.0 + 16.0 + 25.0) ** 0.5
                    norm_e = (25.0 + 36.0 + 49.0) ** 0.5
                    self.assertAllClose([[1.0 / norm_a, 2.0 / norm_a, 3.0 / norm_a], [3.0 / norm_c, 4.0 / norm_c, 5.0 / norm_c], [5.0 / norm_e, 6.0 / norm_e, 7.0 / norm_e]], embeddings[:3].eval())

    def testWordEmbeddingInitializerPresetRowNumber(self):
        if False:
            for i in range(10):
                print('nop')
        records_path = os.path.join(test_flags.temp_dir(), 'records3')
        writer = tf.python_io.TFRecordWriter(records_path)
        writer.write(self._token_embedding('a', [1, 2, 3]))
        writer.write(self._token_embedding('b', [2, 3, 4]))
        writer.write(self._token_embedding('c', [3, 4, 5]))
        writer.write(self._token_embedding('d', [4, 5, 6]))
        writer.write(self._token_embedding('e', [5, 6, 7]))
        del writer
        vocabulary_path = os.path.join(test_flags.temp_dir(), 'vocabulary3')
        with open(vocabulary_path, 'w') as vocabulary_file:
            vocabulary_file.write('a\nc\ne\nx\n')
        for cache_vectors_locally in [False, True]:
            for num_special_embeddings in [None, 1, 2, 5]:
                for override_num_embeddings in [-1, 8, 10]:
                    with self.test_session():
                        embeddings = gen_parser_ops.word_embedding_initializer(vectors=records_path, vocabulary=vocabulary_path, override_num_embeddings=override_num_embeddings, cache_vectors_locally=cache_vectors_locally, num_special_embeddings=num_special_embeddings)
                        expected_num_embeddings = 4 + (num_special_embeddings or 3)
                        if override_num_embeddings > 0:
                            expected_num_embeddings = override_num_embeddings
                        self.assertAllEqual([expected_num_embeddings, 3], tf.shape(embeddings).eval())
                        norm_a = (1.0 + 4.0 + 9.0) ** 0.5
                        norm_c = (9.0 + 16.0 + 25.0) ** 0.5
                        norm_e = (25.0 + 36.0 + 49.0) ** 0.5
                        self.assertAllClose([[1.0 / norm_a, 2.0 / norm_a, 3.0 / norm_a], [3.0 / norm_c, 4.0 / norm_c, 5.0 / norm_c], [5.0 / norm_e, 6.0 / norm_e, 7.0 / norm_e]], embeddings[:3].eval())

    def testWordEmbeddingInitializerVocabularyFileWithDuplicates(self):
        if False:
            for i in range(10):
                print('nop')
        records_path = os.path.join(test_flags.temp_dir(), 'records4')
        writer = tf.python_io.TFRecordWriter(records_path)
        writer.write(self._token_embedding('a', [1, 2, 3]))
        writer.write(self._token_embedding('b', [2, 3, 4]))
        writer.write(self._token_embedding('c', [3, 4, 5]))
        writer.write(self._token_embedding('d', [4, 5, 6]))
        writer.write(self._token_embedding('e', [5, 6, 7]))
        del writer
        vocabulary_path = os.path.join(test_flags.temp_dir(), 'vocabulary4')
        with open(vocabulary_path, 'w') as vocabulary_file:
            vocabulary_file.write('a\nc\ne\nx\ny\nx')
        with self.test_session():
            with self.assertRaises(Exception):
                gen_parser_ops.word_embedding_initializer(vectors=records_path, vocabulary=vocabulary_path).eval()
if __name__ == '__main__':
    tf.test.main()