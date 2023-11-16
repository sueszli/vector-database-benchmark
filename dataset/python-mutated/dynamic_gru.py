import tensorflow as tf
import numpy as np
from sys import argv
from util import run_model

def main():
    if False:
        i = 10
        return i + 15
    tf.set_random_seed(10)
    with tf.Session() as sess:
        rnn_cell = tf.nn.rnn_cell.GRUCell(10)
        initial_state = rnn_cell.zero_state(4, dtype=tf.float32)
        inputs = tf.Variable(tf.random_uniform(shape=(4, 30, 100)), name='input')
        inputs = tf.identity(inputs, 'input_node')
        (outputs, state) = tf.nn.dynamic_rnn(rnn_cell, inputs, initial_state=initial_state, dtype=tf.float32)
        y1 = tf.identity(outputs, 'outputs')
        y2 = tf.identity(state, 'state')
        t1 = tf.ones([4, 30, 10])
        t2 = tf.ones([4, 10])
        loss = tf.reduce_sum((y1 - t1) * (y1 - t1)) + tf.reduce_sum((y2 - t2) * (y2 - t2))
        tf.identity(loss, name='gru_loss')
        net_outputs = map(lambda x: tf.get_default_graph().get_tensor_by_name(x), argv[2].split(','))
        run_model(net_outputs, argv[1], None, argv[3] == 'True')
if __name__ == '__main__':
    main()