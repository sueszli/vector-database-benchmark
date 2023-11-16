from __future__ import print_function, division
from builtins import range, input
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_resnet_convblock import ConvLayer, BatchNormLayer

class IdentityBlock:

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def predict(self, X):
        if False:
            return 10
        pass
if __name__ == '__main__':
    identity_block = IdentityBlock()
    X = np.random.random((1, 224, 224, 256))
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        identity_block.set_session(session)
        session.run(init)
        output = identity_block.predict(X)
        print('output.shape:', output.shape)