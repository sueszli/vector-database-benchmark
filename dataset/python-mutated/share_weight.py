import tensorflow as tf
import numpy as np
from sys import argv
from util import merge_checkpoint

def main():
    if False:
        for i in range(10):
            print('nop')
    '\n    Run this command to generate the pb file\n    1. mkdir model\n    2. python test.py\n    3. wget https://raw.githubusercontent.com/tensorflow/tensorflow/v1.0.0/tensorflow/python/tools/freeze_graph.py\n    4. python freeze_graph.py --input_graph model/share_weight.pbtxt --input_checkpoint model/share_weight.chkp --output_node_names=output --output_graph "share_weight.pb"\n    '
    xs = tf.placeholder(tf.float32, [None, 10])
    W1 = tf.Variable(tf.random_normal([10, 10]))
    b1 = tf.Variable(tf.random_normal([10]))
    Wx_plus_b1 = tf.nn.bias_add(tf.matmul(xs, W1), b1)
    output = tf.nn.tanh(Wx_plus_b1)
    Wx_plus_b2 = tf.nn.bias_add(tf.matmul(output, W1), b1)
    W2 = tf.Variable(tf.random_normal([10, 1]))
    b2 = tf.Variable(tf.random_normal([1]))
    final = tf.nn.bias_add(tf.matmul(Wx_plus_b2, W2), b2, name='output')
    dir = argv[1]
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        checkpointpath = saver.save(sess, dir + '/model.chkp')
        tf.train.write_graph(sess.graph, dir, 'model.pbtxt')
    input_graph = dir + '/model.pbtxt'
    input_checkpoint = dir + '/model.chkp'
    output_node_names = 'output'
    output_graph = dir + '/model.pb'
    merge_checkpoint(input_graph, input_checkpoint, [output_node_names], output_graph)
if __name__ == '__main__':
    main()