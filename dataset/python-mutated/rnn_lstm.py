import tensorflow as tf
import numpy as np
from sys import argv
from tensorflow.contrib import rnn
from util import run_model

def main():
    if False:
        while True:
            i = 10
    '\n    Run this command to generate the pb file\n    1. mkdir model\n    2. python rnn_lstm.py\n    '
    tf.set_random_seed(1)
    n_steps = 2
    n_input = 10
    n_hidden = 20
    n_output = 5
    xs = tf.Variable(tf.random_uniform([4, n_steps, n_input]) + 10, name='input', dtype=tf.float32)
    xs = tf.identity(xs, 'input_node')
    weight = tf.Variable(tf.random_uniform([n_hidden, n_output]) + 10, name='weight', dtype=tf.float32)
    bias = tf.Variable(tf.random_uniform([n_output]) + 10, name='bias', dtype=tf.float32)
    x = tf.unstack(xs, n_steps, 1)
    cell = rnn.BasicLSTMCell(n_hidden)
    (output, states) = rnn.static_rnn(cell, x, dtype=tf.float32)
    final = tf.nn.bias_add(tf.matmul(output[-1], weight), bias, name='output')
    net_outputs = map(lambda x: tf.get_default_graph().get_tensor_by_name(x), argv[2].split(','))
    run_model(net_outputs, argv[1], 'rnn', argv[3] == 'True')
if __name__ == '__main__':
    main()