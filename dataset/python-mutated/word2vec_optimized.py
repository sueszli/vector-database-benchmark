"""Multi-threaded word2vec unbatched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does true SGD (i.e. no minibatching). To do this efficiently, custom
ops are used to sequentially process data within a 'batch'.

The key ops used are:
* skipgram custom op that does input processing.
* neg_train custom op that efficiently calculates and applies the gradient using
  true SGD.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import threading
import time
from six.moves import xrange
import numpy as np
import tensorflow as tf
word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))
flags = tf.app.flags
flags.DEFINE_string('save_path', None, 'Directory to write the model.')
flags.DEFINE_string('train_data', None, 'Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.')
flags.DEFINE_string('eval_data', None, "Analogy questions. See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_integer('embedding_size', 200, 'The embedding dimension size.')
flags.DEFINE_integer('epochs_to_train', 15, 'Number of epochs to train. Each epoch processes the training data once completely.')
flags.DEFINE_float('learning_rate', 0.025, 'Initial learning rate.')
flags.DEFINE_integer('num_neg_samples', 25, 'Negative samples per training example.')
flags.DEFINE_integer('batch_size', 500, 'Numbers of training examples each step processes (no minibatching).')
flags.DEFINE_integer('concurrent_steps', 12, 'The number of concurrent training steps.')
flags.DEFINE_integer('window_size', 5, 'The number of words to predict to the left and right of the target word.')
flags.DEFINE_integer('min_count', 5, 'The minimum number of word occurrences for it to be included in the vocabulary.')
flags.DEFINE_float('subsample', 0.001, 'Subsample threshold for word occurrence. Words that appear with higher frequency will be randomly down-sampled. Set to 0 to disable.')
flags.DEFINE_boolean('interactive', False, "If true, enters an IPython interactive session to play with the trained model. E.g., try model.analogy(b'france', b'paris', b'russia') and model.nearby([b'proton', b'elephant', b'maxwell'])")
FLAGS = flags.FLAGS

class Options(object):
    """Options used by our word2vec model."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.emb_dim = FLAGS.embedding_size
        self.train_data = FLAGS.train_data
        self.num_samples = FLAGS.num_neg_samples
        self.learning_rate = FLAGS.learning_rate
        self.epochs_to_train = FLAGS.epochs_to_train
        self.concurrent_steps = FLAGS.concurrent_steps
        self.batch_size = FLAGS.batch_size
        self.window_size = FLAGS.window_size
        self.min_count = FLAGS.min_count
        self.subsample = FLAGS.subsample
        self.save_path = FLAGS.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.eval_data = FLAGS.eval_data

class Word2Vec(object):
    """Word2Vec model (Skipgram)."""

    def __init__(self, options, session):
        if False:
            while True:
                i = 10
        self._options = options
        self._session = session
        self._word2id = {}
        self._id2word = []
        self.build_graph()
        self.build_eval_graph()
        self.save_vocab()

    def read_analogies(self):
        if False:
            print('Hello World!')
        "Reads through the analogy question file.\n\n    Returns:\n      questions: a [n, 4] numpy array containing the analogy question's\n                 word ids.\n      questions_skipped: questions skipped due to unknown words.\n    "
        questions = []
        questions_skipped = 0
        with open(self._options.eval_data, 'rb') as analogy_f:
            for line in analogy_f:
                if line.startswith(b':'):
                    continue
                words = line.strip().lower().split(b' ')
                ids = [self._word2id.get(w.strip()) for w in words]
                if None in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                    questions.append(np.array(ids))
        print('Eval analogy file: ', self._options.eval_data)
        print('Questions: ', len(questions))
        print('Skipped: ', questions_skipped)
        self._analogy_questions = np.array(questions, dtype=np.int32)

    def build_graph(self):
        if False:
            i = 10
            return i + 15
        'Build the model graph.'
        opts = self._options
        (words, counts, words_per_epoch, current_epoch, total_words_processed, examples, labels) = word2vec.skipgram_word2vec(filename=opts.train_data, batch_size=opts.batch_size, window_size=opts.window_size, min_count=opts.min_count, subsample=opts.subsample)
        (opts.vocab_words, opts.vocab_counts, opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
        opts.vocab_size = len(opts.vocab_words)
        print('Data file: ', opts.train_data)
        print('Vocab size: ', opts.vocab_size - 1, ' + UNK')
        print('Words per epoch: ', opts.words_per_epoch)
        self._id2word = opts.vocab_words
        for (i, w) in enumerate(self._id2word):
            self._word2id[w] = i
        w_in = tf.Variable(tf.random_uniform([opts.vocab_size, opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim), name='w_in')
        w_out = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim]), name='w_out')
        global_step = tf.Variable(0, name='global_step')
        words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
        lr = opts.learning_rate * tf.maximum(0.0001, 1.0 - tf.cast(total_words_processed, tf.float32) / words_to_train)
        inc = global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            train = word2vec.neg_train_word2vec(w_in, w_out, examples, labels, lr, vocab_count=opts.vocab_counts.tolist(), num_negative_samples=opts.num_samples)
        self._w_in = w_in
        self._examples = examples
        self._labels = labels
        self._lr = lr
        self._train = train
        self.global_step = global_step
        self._epoch = current_epoch
        self._words = total_words_processed

    def save_vocab(self):
        if False:
            for i in range(10):
                print('nop')
        'Save the vocabulary to a file so the model can be reloaded.'
        opts = self._options
        with open(os.path.join(opts.save_path, 'vocab.txt'), 'w') as f:
            for i in xrange(opts.vocab_size):
                vocab_word = tf.compat.as_text(opts.vocab_words[i]).encode('utf-8')
                f.write('%s %d\n' % (vocab_word, opts.vocab_counts[i]))

    def build_eval_graph(self):
        if False:
            i = 10
            return i + 15
        'Build the evaluation graph.'
        opts = self._options
        analogy_a = tf.placeholder(dtype=tf.int32)
        analogy_b = tf.placeholder(dtype=tf.int32)
        analogy_c = tf.placeholder(dtype=tf.int32)
        nemb = tf.nn.l2_normalize(self._w_in, 1)
        a_emb = tf.gather(nemb, analogy_a)
        b_emb = tf.gather(nemb, analogy_b)
        c_emb = tf.gather(nemb, analogy_c)
        target = c_emb + (b_emb - a_emb)
        dist = tf.matmul(target, nemb, transpose_b=True)
        (_, pred_idx) = tf.nn.top_k(dist, 4)
        nearby_word = tf.placeholder(dtype=tf.int32)
        nearby_emb = tf.gather(nemb, nearby_word)
        nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
        (nearby_val, nearby_idx) = tf.nn.top_k(nearby_dist, min(1000, opts.vocab_size))
        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def _train_thread_body(self):
        if False:
            while True:
                i = 10
        (initial_epoch,) = self._session.run([self._epoch])
        while True:
            (_, epoch) = self._session.run([self._train, self._epoch])
            if epoch != initial_epoch:
                break

    def train(self):
        if False:
            print('Hello World!')
        'Train the model.'
        opts = self._options
        (initial_epoch, initial_words) = self._session.run([self._epoch, self._words])
        workers = []
        for _ in xrange(opts.concurrent_steps):
            t = threading.Thread(target=self._train_thread_body)
            t.start()
            workers.append(t)
        (last_words, last_time) = (initial_words, time.time())
        while True:
            time.sleep(5)
            (epoch, step, words, lr) = self._session.run([self._epoch, self.global_step, self._words, self._lr])
            now = time.time()
            (last_words, last_time, rate) = (words, now, (words - last_words) / (now - last_time))
            print('Epoch %4d Step %8d: lr = %5.3f words/sec = %8.0f\r' % (epoch, step, lr, rate), end='')
            sys.stdout.flush()
            if epoch != initial_epoch:
                break
        for t in workers:
            t.join()

    def _predict(self, analogy):
        if False:
            return 10
        'Predict the top 4 answers for analogy questions.'
        (idx,) = self._session.run([self._analogy_pred_idx], {self._analogy_a: analogy[:, 0], self._analogy_b: analogy[:, 1], self._analogy_c: analogy[:, 2]})
        return idx

    def eval(self):
        if False:
            return 10
        'Evaluate analogy questions and reports accuracy.'
        correct = 0
        try:
            total = self._analogy_questions.shape[0]
        except AttributeError as e:
            raise AttributeError('Need to read analogy questions.')
        start = 0
        while start < total:
            limit = start + 2500
            sub = self._analogy_questions[start:limit, :]
            idx = self._predict(sub)
            start = limit
            for question in xrange(sub.shape[0]):
                for j in xrange(4):
                    if idx[question, j] == sub[question, 3]:
                        correct += 1
                        break
                    elif idx[question, j] in sub[question, :3]:
                        continue
                    else:
                        break
        print()
        print('Eval %4d/%d accuracy = %4.1f%%' % (correct, total, correct * 100.0 / total))

    def analogy(self, w0, w1, w2):
        if False:
            for i in range(10):
                print('nop')
        'Predict word w3 as in w0:w1 vs w2:w3.'
        wid = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
        idx = self._predict(wid)
        for c in [self._id2word[i] for i in idx[0, :]]:
            if c not in [w0, w1, w2]:
                print(c)
                break
        print('unknown')

    def nearby(self, words, num=20):
        if False:
            while True:
                i = 10
        'Prints out nearby words given a list of words.'
        ids = np.array([self._word2id.get(x, 0) for x in words])
        (vals, idx) = self._session.run([self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
        for i in xrange(len(words)):
            print('\n%s\n=====================================' % words[i])
            for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
                print('%-20s %6.4f' % (self._id2word[neighbor], distance))

def _start_shell(local_ns=None):
    if False:
        i = 10
        return i + 15
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)

def main(_):
    if False:
        i = 10
        return i + 15
    'Train a word2vec model.'
    if not FLAGS.train_data or not FLAGS.eval_data or (not FLAGS.save_path):
        print('--train_data --eval_data and --save_path must be specified.')
        sys.exit(1)
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device('/cpu:0'):
            model = Word2Vec(opts, session)
            model.read_analogies()
        for _ in xrange(opts.epochs_to_train):
            model.train()
            model.eval()
        model.saver.save(session, os.path.join(opts.save_path, 'model.ckpt'), global_step=model.global_step)
        if FLAGS.interactive:
            _start_shell(locals())
if __name__ == '__main__':
    tf.app.run()