"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np

class Layer(object):

    def __init__(self, inputs, in_size, out_size, activation_function=None):
        if False:
            i = 10
            return i + 15
        self.W = theano.shared(np.random.normal(0, 1, (in_size, out_size)))
        self.b = theano.shared(np.zeros((out_size,)) + 0.1)
        self.Wx_plus_b = T.dot(inputs, self.W) + self.b
        self.activation_function = activation_function
        if activation_function is None:
            self.outputs = self.Wx_plus_b
        else:
            self.outputs = self.activation_function(self.Wx_plus_b)
'\nto define the layer like this:\nl1 = Layer(inputs, 1, 10, T.nnet.relu)\nl2 = Layer(l1.outputs, 10, 1, None)\n'