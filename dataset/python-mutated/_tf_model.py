from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import abc as _abc
import six as _six

@_six.add_metaclass(_abc.ABCMeta)
class TensorFlowModel(object):
    """
    Base Class for neural networks written in tensorflow used to abstract across model
    architectures. It defines the computational graph and initialize a session to run the graph.

    Make placeholders for input and targets
    self.data = tf.placeholder()
    self.target = tf.placeholder()

    Make dictionaries for weights and biases
    self.weights = {
        'conv' : tf.Variable()
        'dense0' : tf.Variable()
    }
    self.biases = {
        'conv' : tf.Variable()
        'dense0' : tf.Variable()
    }

    Make the graph
    conv = tf.nn.conv1d(self.data, self.weights['conv'], ..)
    dense = tf.add(tf.matmul() + self.bias())
    ...

    Make loss_op with the loss and train_op with the optimizer
    loss_op =
    train_op =

    Define Session
    self.sess = tf.Session()
    """

    @_abc.abstractmethod
    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        Train will do a forward and backward pass and update weights\n        This accepts a dictionary that has feature/target as key and\n        the numpy arrays as value corresponding to them respectively.\n        It returns a dictionary of loss and output (probabilities)\n        This matches model backend train\n\n        Argument : A dictionary of input and true labels\n        Returns : A dictionary of expected output (toolkit specific)\n\n        It will train a mini batch by running the optimizer in the session\n        Running the optimizer is thepart that does back propogation\n        self.sess.run([train_op, loss_op, ..], feed_dict= {self.data = ..., self.target= ..})\n        '
        raise NotImplementedError

    def train(self, feed_dict):
        if False:
            for i in range(10):
                print('nop')
        '\n        Predict does only a forward pass and does not update any weights\n        This accepts a dictionary that has feature/target as key and\n        the numpy arrays as value corresponding to them respectively.\n        It also returns a dictionary of loss and output\n        This matches the model backend predict\n\n        Argument : A dictionary of input and true labels\n        Returns : A dictionary of expected output (toolkit specific)\n\n        It will calculate the specified outputs w\n        self.sess.run([loss_op, ..], feed_dict= {self.data = ..., self.target= ..})\n        '
        raise NotImplementedError

    def predict(self, feed_dict):
        if False:
            i = 10
            return i + 15
        '\n        Exports the network weights in CoreML format.\n        Returns : A dictionary of weight names as keys and\n\n        layer_names = tf.trainable_variables()\n        layer_weights = self.sess.run(tvars)\n\n        This will get you the layer names from tensorflow and their corresponding\n        values. They need to be converted to CoreML format and stored back in a\n        dictionary with their names and values of correct shapes.\n        '
        raise NotImplementedError

    def export_weights(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the optimizer to learn at the specified learning rate or using a learning rate scheduler.\n        '
        raise NotImplementedError

    def set_learning_rate(self, learning_rate):
        if False:
            while True:
                i = 10
        raise NotImplementedError