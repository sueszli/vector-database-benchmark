""" starter code for word2vec skip-gram model with NCE loss
CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 04
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import utils
import word2vec_utils
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128
SKIP_WINDOW = 1
NUM_SAMPLED = 64
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
VISUAL_FLD = 'visualization'
SKIP_STEP = 5000
DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
EXPECTED_BYTES = 31344016
NUM_VISUALIZE = 3000

def word2vec(dataset):
    if False:
        for i in range(10):
            print('nop')
    ' Build the graph for word2vec model and train it '
    with tf.name_scope('data'):
        iterator = dataset.make_initializable_iterator()
        (center_words, target_words) = iterator.get_next()
    " Step 2 + 3: define weights and embedding lookup.\n    In word2vec, it's actually the weights that we care about \n    "
    with tf.name_scope('embed'):
        embed_matrix = tf.get_variable('embed_matrix', shape=[VOCAB_SIZE, EMBED_SIZE], initializer=tf.random_uniform_initializer())
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embedding')
    with tf.name_scope('loss'):
        nce_weight = tf.get_variable('nce_weight', shape=[VOCAB_SIZE, EMBED_SIZE], initializer=tf.truncated_normal_initializer(stddev=1.0 / EMBED_SIZE ** 0.5))
        nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, labels=target_words, inputs=embed, num_sampled=NUM_SAMPLED, num_classes=VOCAB_SIZE), name='loss')
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    utils.safe_mkdir('checkpoints')
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())
        total_loss = 0.0
        writer = tf.summary.FileWriter('graphs/word2vec_simple', sess.graph)
        for index in range(NUM_TRAIN_STEPS):
            try:
                (loss_batch, _) = sess.run([loss, optimizer])
                total_loss += loss_batch
                if (index + 1) % SKIP_STEP == 0:
                    print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                    total_loss = 0.0
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
        writer.close()

def gen():
    if False:
        print('Hello World!')
    yield from word2vec_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES, VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW, VISUAL_FLD)

def main():
    if False:
        i = 10
        return i + 15
    dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32), (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
    word2vec(dataset)
if __name__ == '__main__':
    main()