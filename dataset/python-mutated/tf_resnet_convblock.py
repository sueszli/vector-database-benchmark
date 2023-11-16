from __future__ import print_function, division
from builtins import range, input
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def init_filter(d, mi, mo, stride):
    if False:
        for i in range(10):
            print('nop')
    return (np.random.randn(d, d, mi, mo) * np.sqrt(2.0 / (d * d * mi))).astype(np.float32)

class ConvLayer:

    def __init__(self, d, mi, mo, stride=2, padding='VALID'):
        if False:
            print('Hello World!')
        self.W = tf.Variable(init_filter(d, mi, mo, stride))
        self.b = tf.Variable(np.zeros(mo, dtype=np.float32))
        self.stride = stride
        self.padding = padding

    def forward(self, X):
        if False:
            while True:
                i = 10
        X = tf.nn.conv2d(X, self.W, strides=[1, self.stride, self.stride, 1], padding=self.padding)
        X = X + self.b
        return X

    def copyFromKerasLayers(self, layer):
        if False:
            i = 10
            return i + 15
        (W, b) = layer.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)
        self.session.run((op1, op2))

    def get_params(self):
        if False:
            i = 10
            return i + 15
        return [self.W, self.b]

class BatchNormLayer:

    def __init__(self, D):
        if False:
            return 10
        self.running_mean = tf.Variable(np.zeros(D, dtype=np.float32), trainable=False)
        self.running_var = tf.Variable(np.ones(D, dtype=np.float32), trainable=False)
        self.gamma = tf.Variable(np.ones(D, dtype=np.float32))
        self.beta = tf.Variable(np.zeros(D, dtype=np.float32))

    def forward(self, X):
        if False:
            return 10
        return tf.nn.batch_normalization(X, self.running_mean, self.running_var, self.beta, self.gamma, 0.001)

    def copyFromKerasLayers(self, layer):
        if False:
            while True:
                i = 10
        (gamma, beta, running_mean, running_var) = layer.get_weights()
        op1 = self.running_mean.assign(running_mean)
        op2 = self.running_var.assign(running_var)
        op3 = self.gamma.assign(gamma)
        op4 = self.beta.assign(beta)
        self.session.run((op1, op2, op3, op4))

    def get_params(self):
        if False:
            while True:
                i = 10
        return [self.running_mean, self.running_var, self.gamma, self.beta]

class ConvBlock:

    def __init__(self, mi, fm_sizes, stride=2, activation=tf.nn.relu):
        if False:
            i = 10
            return i + 15
        assert len(fm_sizes) == 3
        self.session = None
        self.f = tf.nn.relu
        self.conv1 = ConvLayer(1, mi, fm_sizes[0], stride)
        self.bn1 = BatchNormLayer(fm_sizes[0])
        self.conv2 = ConvLayer(3, fm_sizes[0], fm_sizes[1], 1, 'SAME')
        self.bn2 = BatchNormLayer(fm_sizes[1])
        self.conv3 = ConvLayer(1, fm_sizes[1], fm_sizes[2], 1)
        self.bn3 = BatchNormLayer(fm_sizes[2])
        self.convs = ConvLayer(1, mi, fm_sizes[2], stride)
        self.bns = BatchNormLayer(fm_sizes[2])
        self.layers = [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.convs, self.bns]
        self.input_ = tf.placeholder(tf.float32, shape=(1, 224, 224, mi))
        self.output = self.forward(self.input_)

    def forward(self, X):
        if False:
            i = 10
            return i + 15
        FX = self.conv1.forward(X)
        FX = self.bn1.forward(FX)
        FX = self.f(FX)
        FX = self.conv2.forward(FX)
        FX = self.bn2.forward(FX)
        FX = self.f(FX)
        FX = self.conv3.forward(FX)
        FX = self.bn3.forward(FX)
        SX = self.convs.forward(X)
        SX = self.bns.forward(SX)
        return self.f(FX + SX)

    def predict(self, X):
        if False:
            while True:
                i = 10
        assert self.session is not None
        return self.session.run(self.output, feed_dict={self.input_: X})

    def set_session(self, session):
        if False:
            i = 10
            return i + 15
        self.session = session
        self.conv1.session = session
        self.bn1.session = session
        self.conv2.session = session
        self.bn2.session = session
        self.conv3.session = session
        self.bn3.session = session
        self.convs.session = session
        self.bns.session = session

    def copyFromKerasLayers(self, layers):
        if False:
            print('Hello World!')
        self.conv1.copyFromKerasLayers(layers[0])
        self.bn1.copyFromKerasLayers(layers[1])
        self.conv2.copyFromKerasLayers(layers[3])
        self.bn2.copyFromKerasLayers(layers[4])
        self.conv3.copyFromKerasLayers(layers[6])
        self.bn3.copyFromKerasLayers(layers[8])
        self.convs.copyFromKerasLayers(layers[7])
        self.bns.copyFromKerasLayers(layers[9])

    def get_params(self):
        if False:
            while True:
                i = 10
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params
if __name__ == '__main__':
    conv_block = ConvBlock(mi=3, fm_sizes=[64, 64, 256], stride=1)
    X = np.random.random((1, 224, 224, 3))
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        conv_block.set_session(session)
        session.run(init)
        output = conv_block.predict(X)
        print('output.shape:', output.shape)