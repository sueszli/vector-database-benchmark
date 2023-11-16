import tensorflow as tf
from sys import argv
from util import run_model

def main():
    if False:
        return 10
    inputs = tf.Variable(tf.reshape(tf.range(0.0, 16), [1, 4, 4, 1]), name='input')
    inputs = tf.identity(inputs, 'input_node')
    output = tf.layers.batch_normalization(inputs, axis=3, training=True)
    named_output = tf.nn.relu(output, name='output')
    net_outputs = map(lambda x: tf.get_default_graph().get_tensor_by_name(x), argv[2].split(','))
    run_model(net_outputs, argv[1], 'batchNorm', argv[3] == 'True')
if __name__ == '__main__':
    main()