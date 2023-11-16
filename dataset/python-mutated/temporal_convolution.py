import tensorflow as tf
from sys import argv
from util import run_model

def main():
    if False:
        print('Hello World!')
    '\n    You can also run these commands manually to generate the pb file\n    1. git clone https://github.com/tensorflow/models.git\n    2. export PYTHONPATH=Path_to_your_model_folder\n    3. python temporal_convolution.py\n    '
    tf.set_random_seed(1024)
    input_width = 32
    input_channel = 3
    inputs = tf.Variable(tf.random_uniform((1, input_width, input_channel)), name='input')
    inputs = tf.identity(inputs, 'input_node')
    filter_width = 4
    output_channels = 6
    filters = tf.Variable(tf.random_uniform((filter_width, input_channel, output_channels)))
    conv_out = tf.nn.conv1d(inputs, filters, stride=1, padding='VALID')
    bias = tf.Variable(tf.zeros([output_channels]))
    output = tf.nn.tanh(tf.nn.bias_add(conv_out, bias), name='output')
    net_outputs = map(lambda x: tf.get_default_graph().get_tensor_by_name(x), argv[2].split(','))
    run_model(net_outputs, argv[1], backward=argv[3] == 'True')
if __name__ == '__main__':
    main()