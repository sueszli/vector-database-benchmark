from __future__ import print_function, division
from builtins import range, input
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_resnet_convblock import ConvLayer, BatchNormLayer

class IdentityBlock:

    def __init__(self, mi, fm_sizes, activation=tf.nn.relu):
        if False:
            print('Hello World!')
        assert len(fm_sizes) == 3
        self.session = None
        self.f = tf.nn.relu
        self.conv1 = ConvLayer(1, mi, fm_sizes[0], 1)
        self.bn1 = BatchNormLayer(fm_sizes[0])
        self.conv2 = ConvLayer(3, fm_sizes[0], fm_sizes[1], 1, 'SAME')
        self.bn2 = BatchNormLayer(fm_sizes[1])
        self.conv3 = ConvLayer(1, fm_sizes[1], fm_sizes[2], 1)
        self.bn3 = BatchNormLayer(fm_sizes[2])
        self.layers = [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3]
        self.input_ = tf.placeholder(tf.float32, shape=(1, 224, 224, mi))
        self.output = self.forward(self.input_)

    def forward(self, X):
        if False:
            for i in range(10):
                print('nop')
        FX = self.conv1.forward(X)
        FX = self.bn1.forward(FX)
        FX = self.f(FX)
        FX = self.conv2.forward(FX)
        FX = self.bn2.forward(FX)
        FX = self.f(FX)
        FX = self.conv3.forward(FX)
        FX = self.bn3.forward(FX)
        return self.f(FX + X)

    def predict(self, X):
        if False:
            i = 10
            return i + 15
        assert self.session is not None
        return self.session.run(self.output, feed_dict={self.input_: X})

    def set_session(self, session):
        if False:
            return 10
        self.session = session
        self.conv1.session = session
        self.bn1.session = session
        self.conv2.session = session
        self.bn2.session = session
        self.conv3.session = session
        self.bn3.session = session

    def copyFromKerasLayers(self, layers):
        if False:
            i = 10
            return i + 15
        assert len(layers) == 10
        self.conv1.copyFromKerasLayers(layers[0])
        self.bn1.copyFromKerasLayers(layers[1])
        self.conv2.copyFromKerasLayers(layers[3])
        self.bn2.copyFromKerasLayers(layers[4])
        self.conv3.copyFromKerasLayers(layers[6])
        self.bn3.copyFromKerasLayers(layers[7])

    def get_params(self):
        if False:
            print('Hello World!')
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params
if __name__ == '__main__':
    identity_block = IdentityBlock(mi=256, fm_sizes=[64, 64, 256])
    X = np.random.random((1, 224, 224, 256))
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        identity_block.set_session(session)
        session.run(init)
        output = identity_block.predict(X)
        print('output.shape:', output.shape)