"""
This demo implements FastText[1] for sentence classification. This demo should be run in eager mode and
can be slower than the corresponding demo in graph mode.

FastText is a simple model for text classification with performance often close
to state-of-the-art, and is useful as a solid baseline.

There are some important differences between this implementation and what
is described in the paper. Instead of Hogwild! SGD[2], we use Adam optimizer
with mini-batches. Hierarchical softmax is also not supported; if you have
a large label space, consider utilizing candidate sampling methods provided
by TensorFlow[3].

After 5 epochs, you should get test accuracy around 90.3%.

[1] Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2016).
    Bag of Tricks for Efficient Text Classification.
    http://arxiv.org/abs/1607.01759

[2] Recht, B., Re, C., Wright, S., & Niu, F. (2011).
    Hogwild: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent.
    In Advances in Neural Information Processing Systems 24 (pp. 693–701).

[3] https://www.tensorflow.org/api_guides/python/nn#Candidate_Sampling

"""
import array
import hashlib
import os
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import *
N_GRAM = 2
VOCAB_SIZE = 100000
N_BUCKETS = 1000000
EMBEDDING_SIZE = 50
N_EPOCH = 5
N_STEPS_TO_PRINT = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MODEL_FILE_PATH = 'model_dynamic.hdf5'

class FastTextModel(Model):
    """  Model structure and forwarding of FastText """

    def __init__(self, vocab_size, embedding_size, n_labels, name='fasttext'):
        if False:
            print('Hello World!')
        super(FastTextModel, self).__init__(name=name)
        self.avg_embed = AverageEmbedding(vocab_size, embedding_size)
        self.dense1 = Dense(n_units=10, in_channels=embedding_size)
        self.dense2 = Dense(n_units=n_labels, in_channels=10)

    def forward(self, x):
        if False:
            print('Hello World!')
        z = self.avg_embed(x)
        z = self.dense1(z)
        z = self.dense2(z)
        return z

def augment_with_ngrams(unigrams, unigram_vocab_size, n_buckets, n=2):
    if False:
        for i in range(10):
            print('nop')
    'Augment unigram features with hashed n-gram features.'

    def get_ngrams(n):
        if False:
            for i in range(10):
                print('nop')
        return list(zip(*[unigrams[i:] for i in range(n)]))

    def hash_ngram(ngram):
        if False:
            i = 10
            return i + 15
        bytes_ = array.array('L', ngram).tobytes()
        hash_ = int(hashlib.sha256(bytes_).hexdigest(), 16)
        return unigram_vocab_size + hash_ % n_buckets
    return unigrams + [hash_ngram(ngram) for i in range(2, n + 1) for ngram in get_ngrams(i)]

def load_and_preprocess_imdb_data(n_gram=None):
    if False:
        return 10
    'Load IMDb data and augment with hashed n-gram features.'
    tl.logging.info('Loading and preprocessing IMDB data.')
    (X_train, y_train, X_test, y_test) = tl.files.load_imdb_dataset(nb_words=VOCAB_SIZE)
    if n_gram is not None:
        X_train = np.array([augment_with_ngrams(x, VOCAB_SIZE, N_BUCKETS, n=n_gram) for x in X_train])
        X_test = np.array([augment_with_ngrams(x, VOCAB_SIZE, N_BUCKETS, n=n_gram) for x in X_test])
    return (X_train, y_train, X_test, y_test)

def train_test_and_save_model():
    if False:
        for i in range(10):
            print('nop')
    (X_train, y_train, X_test, y_test) = load_and_preprocess_imdb_data(N_GRAM)
    model = FastTextModel(vocab_size=VOCAB_SIZE + N_BUCKETS, embedding_size=EMBEDDING_SIZE, n_labels=2)
    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
    if os.path.exists(MODEL_FILE_PATH):
        model.load_weights(MODEL_FILE_PATH)
    else:
        model.train()
        for epoch in range(N_EPOCH):
            start_time = time.time()
            print('Epoch %d/%d' % (epoch + 1, N_EPOCH))
            train_accuracy = list()
            for (X_batch, y_batch) in tl.iterate.minibatches(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True):
                with tf.GradientTape() as tape:
                    y_pred = model(tl.prepro.pad_sequences(X_batch))
                    cost = tl.cost.cross_entropy(y_pred, y_batch, name='cost')
                grad = tape.gradient(cost, model.trainable_weights)
                optimizer.apply_gradients(zip(grad, model.trainable_weights))
                predictions = tf.argmax(y_pred, axis=1, output_type=tf.int32)
                are_predictions_correct = tf.equal(predictions, y_batch)
                accuracy = tf.reduce_mean(tf.cast(are_predictions_correct, tf.float32))
                train_accuracy.append(accuracy)
                if len(train_accuracy) % N_STEPS_TO_PRINT == 0:
                    print('\t[%d/%d][%d]accuracy ' % (epoch + 1, N_EPOCH, len(train_accuracy)), np.mean(train_accuracy[-N_STEPS_TO_PRINT:]))
            print('\tSummary: time %.5fs, overall accuracy' % (time.time() - start_time), np.mean(train_accuracy))
    model.eval()
    y_pred = model(tl.prepro.pad_sequences(X_test))
    predictions = tf.argmax(y_pred, axis=1, output_type=tf.int32)
    are_predictions_correct = tf.equal(predictions, y_test)
    test_accuracy = tf.reduce_mean(tf.cast(are_predictions_correct, tf.float32))
    print('Test accuracy: %.5f' % test_accuracy)
    model.save_weights(MODEL_FILE_PATH)
if __name__ == '__main__':
    train_test_and_save_model()