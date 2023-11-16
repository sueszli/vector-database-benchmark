import tensorflow as tf
from sys import argv
from util import run_model

def main():
    if False:
        for i in range(10):
            print('nop')
    inputs = tf.Variable(tf.reshape(tf.range(0.0, 4.0), [4, 1]), name='input')
    inputs = tf.identity(inputs, 'input_node')
    W1 = tf.Variable(tf.zeros([1, 10]) + 0.2)
    b1 = tf.Variable(tf.zeros([10]) + 0.1)
    out1 = tf.nn.bias_add(tf.matmul(inputs, W1), b1)
    W2 = tf.Variable(tf.zeros([1, 10]) + 0.2)
    b2 = tf.Variable(tf.zeros([10]) + 0.1)
    out2 = tf.nn.bias_add(tf.matmul(inputs, W2), b2)
    with tf.control_dependencies([inputs]):
        output = tf.add_n([out1, out2])
    named_output = tf.nn.relu(output, name='output')
    net_outputs = map(lambda x: tf.get_default_graph().get_tensor_by_name(x), argv[2].split(','))
    run_model(net_outputs, argv[1], 'control_dep', argv[3] == 'True')
if __name__ == '__main__':
    main()