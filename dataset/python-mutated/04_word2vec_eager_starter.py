""" starter code for word2vec skip-gram model with NCE loss
Eager execution
CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu) & Akshay Agrawal (akshayka@cs.stanford.edu)
Lecture 04
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
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

class Word2Vec(object):

    def __init__(self, vocab_size, embed_size, num_sampled=NUM_SAMPLED):
        if False:
            for i in range(10):
                print('nop')
        self.vocab_size = vocab_size
        self.num_sampled = num_sampled
        self.embed_matrix = None
        self.nce_weight = None
        self.nce_bias = None

    def compute_loss(self, center_words, target_words):
        if False:
            print('Hello World!')
        'Computes the forward pass of word2vec with the NCE loss.'
        embed = None
        loss = None
        return loss

def gen():
    if False:
        i = 10
        return i + 15
    yield from word2vec_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES, VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW, VISUAL_FLD)

def main():
    if False:
        for i in range(10):
            print('nop')
    dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32), (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    model = None
    grad_fn = None
    total_loss = 0.0
    num_train_steps = 0
    while num_train_steps < NUM_TRAIN_STEPS:
        for (center_words, target_words) in tfe.Iterator(dataset):
            if num_train_steps >= NUM_TRAIN_STEPS:
                break
            if (num_train_steps + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(num_train_steps, total_loss / SKIP_STEP))
                total_loss = 0.0
            num_train_steps += 1
if __name__ == '__main__':
    main()