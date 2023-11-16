"""Tests for graphs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict
import operator
import os
import random
import shutil
import string
import tempfile
import tensorflow as tf
import graphs
from data import data_utils
flags = tf.app.flags
FLAGS = flags.FLAGS
data = data_utils
flags.DEFINE_integer('task', 0, 'Task id; needed for SyncReplicas test')

def _build_random_vocabulary(vocab_size=100):
    if False:
        print('Hello World!')
    'Builds and returns a dict<term, id>.'
    vocab = set()
    while len(vocab) < vocab_size - 1:
        rand_word = ''.join((random.choice(string.ascii_lowercase) for _ in range(random.randint(1, 10))))
        vocab.add(rand_word)
    vocab_ids = dict([(word, i) for (i, word) in enumerate(vocab)])
    vocab_ids[data.EOS_TOKEN] = vocab_size - 1
    return vocab_ids

def _build_random_sequence(vocab_ids):
    if False:
        for i in range(10):
            print('nop')
    seq_len = random.randint(10, 200)
    ids = vocab_ids.values()
    seq = data.SequenceWrapper()
    for token_id in [random.choice(ids) for _ in range(seq_len)]:
        seq.add_timestep().set_token(token_id)
    return seq

def _build_vocab_frequencies(seqs, vocab_ids):
    if False:
        return 10
    vocab_freqs = defaultdict(int)
    ids_to_words = dict([(i, word) for (word, i) in vocab_ids.iteritems()])
    for seq in seqs:
        for timestep in seq:
            vocab_freqs[ids_to_words[timestep.token]] += 1
    vocab_freqs[data.EOS_TOKEN] = 0
    return vocab_freqs

class GraphsTest(tf.test.TestCase):
    """Test graph construction methods."""

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        FLAGS.batch_size = 2
        FLAGS.num_timesteps = 3
        FLAGS.embedding_dims = 4
        FLAGS.rnn_num_layers = 2
        FLAGS.rnn_cell_size = 4
        FLAGS.cl_num_layers = 2
        FLAGS.cl_hidden_size = 4
        FLAGS.vocab_size = 10
        FLAGS.data_dir = tempfile.mkdtemp()
        vocab_ids = _build_random_vocabulary(FLAGS.vocab_size)
        seqs = [_build_random_sequence(vocab_ids) for _ in range(5)]
        seqs_label = [data.build_labeled_sequence(seq, random.choice([True, False])) for seq in seqs]
        seqs_lm = [data.build_lm_sequence(seq) for seq in seqs]
        seqs_ae = [data.build_seq_ae_sequence(seq) for seq in seqs]
        seqs_rev = [data.build_reverse_sequence(seq) for seq in seqs]
        seqs_bidir = [data.build_bidirectional_seq(seq, rev) for (seq, rev) in zip(seqs, seqs_rev)]
        seqs_bidir_label = [data.build_labeled_sequence(bd_seq, random.choice([True, False])) for bd_seq in seqs_bidir]
        filenames = [data.TRAIN_CLASS, data.TRAIN_LM, data.TRAIN_SA, data.TEST_CLASS, data.TRAIN_REV_LM, data.TRAIN_BD_CLASS, data.TEST_BD_CLASS]
        seq_lists = [seqs_label, seqs_lm, seqs_ae, seqs_label, seqs_rev, seqs_bidir, seqs_bidir_label]
        for (fname, seq_list) in zip(filenames, seq_lists):
            with tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, fname)) as writer:
                for seq in seq_list:
                    writer.write(seq.seq.SerializeToString())
        vocab_freqs = _build_vocab_frequencies(seqs, vocab_ids)
        ordered_vocab_freqs = sorted(vocab_freqs.items(), key=operator.itemgetter(1), reverse=True)
        with open(os.path.join(FLAGS.data_dir, 'vocab.txt'), 'w') as vocab_f:
            with open(os.path.join(FLAGS.data_dir, 'vocab_freq.txt'), 'w') as freq_f:
                for (word, freq) in ordered_vocab_freqs:
                    vocab_f.write('{}\n'.format(word))
                    freq_f.write('{}\n'.format(freq))

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        shutil.rmtree(FLAGS.data_dir)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        FLAGS.rnn_num_layers = 1
        FLAGS.sync_replicas = False
        FLAGS.adv_training_method = None
        FLAGS.num_candidate_samples = -1
        FLAGS.num_classes = 2
        FLAGS.use_seq2seq_autoencoder = False
        tf.reset_default_graph()

    def testClassifierGraph(self):
        if False:
            i = 10
            return i + 15
        FLAGS.rnn_num_layers = 2
        model = graphs.VatxtModel()
        (train_op, _, _) = model.classifier_training()
        self.assertEqual(len(model.pretrained_variables), 1 + 2 * FLAGS.rnn_num_layers)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.start_queue_runners(sess)
            sess.run(train_op)

    def testLanguageModelGraph(self):
        if False:
            for i in range(10):
                print('nop')
        (train_op, _, _) = graphs.VatxtModel().language_model_training()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.start_queue_runners(sess)
            sess.run(train_op)

    def testMulticlass(self):
        if False:
            i = 10
            return i + 15
        FLAGS.num_classes = 10
        graphs.VatxtModel().classifier_graph()

    def testATMethods(self):
        if False:
            while True:
                i = 10
        at_methods = [None, 'rp', 'at', 'vat', 'atvat']
        for method in at_methods:
            FLAGS.adv_training_method = method
            with tf.Graph().as_default():
                graphs.VatxtModel().classifier_graph()
                expected_num_vars = 1 + 2 * FLAGS.rnn_num_layers + 2 * FLAGS.cl_num_layers + 2
                self.assertEqual(len(tf.trainable_variables()), expected_num_vars)

    def testSyncReplicas(self):
        if False:
            for i in range(10):
                print('nop')
        FLAGS.sync_replicas = True
        graphs.VatxtModel().language_model_training()

    def testCandidateSampling(self):
        if False:
            while True:
                i = 10
        FLAGS.num_candidate_samples = 10
        graphs.VatxtModel().language_model_training()

    def testSeqAE(self):
        if False:
            while True:
                i = 10
        FLAGS.use_seq2seq_autoencoder = True
        graphs.VatxtModel().language_model_training()

    def testBidirLM(self):
        if False:
            i = 10
            return i + 15
        graphs.VatxtBidirModel().language_model_graph()

    def testBidirClassifier(self):
        if False:
            for i in range(10):
                print('nop')
        at_methods = [None, 'rp', 'at', 'vat', 'atvat']
        for method in at_methods:
            FLAGS.adv_training_method = method
            with tf.Graph().as_default():
                graphs.VatxtBidirModel().classifier_graph()
                expected_num_vars = 1 + 2 * 2 * FLAGS.rnn_num_layers + 2 * FLAGS.cl_num_layers + 2
                self.assertEqual(len(tf.trainable_variables()), expected_num_vars)

    def testEvalGraph(self):
        if False:
            print('Hello World!')
        (_, _) = graphs.VatxtModel().eval_graph()

    def testBidirEvalGraph(self):
        if False:
            for i in range(10):
                print('nop')
        (_, _) = graphs.VatxtBidirModel().eval_graph()
if __name__ == '__main__':
    tf.test.main()