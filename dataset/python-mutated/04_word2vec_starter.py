""" The mo frills implementation of word2vec skip-gram model using NCE loss. 
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from process_data import process_data
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128
SKIP_WINDOW = 1
NUM_SAMPLED = 64
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 20000
SKIP_STEP = 2000

def word2vec(batch_gen):
    if False:
        for i in range(10):
            print('nop')
    ' Build the graph for word2vec model and train it '
    with tf.Session() as sess:
        total_loss = 0.0
        writer = tf.summary.FileWriter('./graphs/no_frills/', sess.graph)
        for index in range(NUM_TRAIN_STEPS):
            (centers, targets) = next(batch_gen)
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
        writer.close()

def main():
    if False:
        while True:
            i = 10
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)
if __name__ == '__main__':
    main()