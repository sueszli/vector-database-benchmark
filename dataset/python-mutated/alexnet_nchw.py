import tensorflow as tf
from sys import argv
from alexnet_v1 import alexnet_v1
from util import run_model

def main():
    if False:
        i = 10
        return i + 15
    '\n    You can also run these commands manually to generate the pb file\n    1. git clone https://github.com/tensorflow/models.git\n    2. export PYTHONPATH=Path_to_your_model_folder\n    3. python alexnet.py\n    '
    (height, width) = (224, 224)
    batchSize = 1
    if len(argv) == 5:
        batchSize = int(argv[4])
    inputs = tf.Variable(tf.random_uniform((batchSize, 3, height, width)), name='input')
    inputs = tf.identity(inputs, 'input_node')
    (net, end_points) = alexnet_v1(inputs, is_training=False, spatial_squeeze=False)
    print('nodes in the graph')
    for n in end_points:
        print(n + ' => ' + str(end_points[n]))
    net_outputs = map(lambda x: tf.get_default_graph().get_tensor_by_name(x), argv[2].split(','))
    run_model(net_outputs, argv[1], 'alexnet', argv[3] == 'True')
if __name__ == '__main__':
    main()