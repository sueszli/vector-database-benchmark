from __future__ import print_function, division
from builtins import range, input
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.layers import Dense
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tf_resnet_convblock import ConvLayer, BatchNormLayer, ConvBlock
from tf_resnet_identity_block import IdentityBlock
from tf_resnet_first_layers import ReLULayer, MaxPoolLayer

class AvgPool:

    def __init__(self, ksize):
        if False:
            return 10
        self.ksize = ksize

    def forward(self, X):
        if False:
            return 10
        return tf.nn.avg_pool(X, ksize=[1, self.ksize, self.ksize, 1], strides=[1, 1, 1, 1], padding='VALID')

    def get_params(self):
        if False:
            while True:
                i = 10
        return []

class Flatten:

    def forward(self, X):
        if False:
            i = 10
            return i + 15
        return tf.contrib.layers.flatten(X)

    def get_params(self):
        if False:
            for i in range(10):
                print('nop')
        return []

def custom_softmax(x):
    if False:
        return 10
    m = tf.reduce_max(x, 1)
    x = x - m
    e = tf.exp(x)
    return e / tf.reduce_sum(e, -1)

class DenseLayer:

    def __init__(self, mi, mo):
        if False:
            for i in range(10):
                print('nop')
        self.W = tf.Variable((np.random.randn(mi, mo) * np.sqrt(2.0 / mi)).astype(np.float32))
        self.b = tf.Variable(np.zeros(mo, dtype=np.float32))

    def forward(self, X):
        if False:
            i = 10
            return i + 15
        return tf.matmul(X, self.W) + self.b

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
            for i in range(10):
                print('nop')
        return [self.W, self.b]

class TFResNet:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.layers = [ConvLayer(d=7, mi=3, mo=64, stride=2, padding='SAME'), BatchNormLayer(64), ReLULayer(), MaxPoolLayer(dim=3), ConvBlock(mi=64, fm_sizes=[64, 64, 256], stride=1), IdentityBlock(mi=256, fm_sizes=[64, 64, 256]), IdentityBlock(mi=256, fm_sizes=[64, 64, 256]), ConvBlock(mi=256, fm_sizes=[128, 128, 512], stride=2), IdentityBlock(mi=512, fm_sizes=[128, 128, 512]), IdentityBlock(mi=512, fm_sizes=[128, 128, 512]), IdentityBlock(mi=512, fm_sizes=[128, 128, 512]), ConvBlock(mi=512, fm_sizes=[256, 256, 1024], stride=2), IdentityBlock(mi=1024, fm_sizes=[256, 256, 1024]), IdentityBlock(mi=1024, fm_sizes=[256, 256, 1024]), IdentityBlock(mi=1024, fm_sizes=[256, 256, 1024]), IdentityBlock(mi=1024, fm_sizes=[256, 256, 1024]), IdentityBlock(mi=1024, fm_sizes=[256, 256, 1024]), ConvBlock(mi=1024, fm_sizes=[512, 512, 2048], stride=2), IdentityBlock(mi=2048, fm_sizes=[512, 512, 2048]), IdentityBlock(mi=2048, fm_sizes=[512, 512, 2048]), AvgPool(ksize=7), Flatten(), DenseLayer(mi=2048, mo=1000)]
        self.input_ = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        self.output = self.forward(self.input_)

    def copyFromKerasLayers(self, layers):
        if False:
            i = 10
            return i + 15
        self.layers[0].copyFromKerasLayers(layers[1])
        self.layers[1].copyFromKerasLayers(layers[2])
        self.layers[4].copyFromKerasLayers(layers[5:17])
        self.layers[5].copyFromKerasLayers(layers[17:27])
        self.layers[6].copyFromKerasLayers(layers[27:37])
        self.layers[7].copyFromKerasLayers(layers[37:49])
        self.layers[8].copyFromKerasLayers(layers[49:59])
        self.layers[9].copyFromKerasLayers(layers[59:69])
        self.layers[10].copyFromKerasLayers(layers[69:79])
        self.layers[11].copyFromKerasLayers(layers[79:91])
        self.layers[12].copyFromKerasLayers(layers[91:101])
        self.layers[13].copyFromKerasLayers(layers[101:111])
        self.layers[14].copyFromKerasLayers(layers[111:121])
        self.layers[15].copyFromKerasLayers(layers[121:131])
        self.layers[16].copyFromKerasLayers(layers[131:141])
        self.layers[17].copyFromKerasLayers(layers[141:153])
        self.layers[18].copyFromKerasLayers(layers[153:163])
        self.layers[19].copyFromKerasLayers(layers[163:173])
        self.layers[22].copyFromKerasLayers(layers[175])

    def forward(self, X):
        if False:
            return 10
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def predict(self, X):
        if False:
            i = 10
            return i + 15
        assert self.session is not None
        return self.session.run(self.output, feed_dict={self.input_: X})

    def set_session(self, session):
        if False:
            for i in range(10):
                print('nop')
        self.session = session
        for layer in self.layers:
            if isinstance(layer, ConvBlock) or isinstance(layer, IdentityBlock):
                layer.set_session(session)
            else:
                layer.session = session

    def get_params(self):
        if False:
            return 10
        params = []
        for layer in self.layers:
            params += layer.get_params()
if __name__ == '__main__':
    resnet_ = ResNet50(weights='imagenet')
    x = resnet_.layers[-2].output
    (W, b) = resnet_.layers[-1].get_weights()
    y = Dense(1000)(x)
    resnet = Model(resnet_.input, y)
    resnet.layers[-1].set_weights([W, b])
    partial_model = Model(inputs=resnet.input, outputs=resnet.layers[175].output)
    print(partial_model.summary())
    my_partial_resnet = TFResNet()
    X = np.random.random((1, 224, 224, 3))
    keras_output = partial_model.predict(X)
    init = tf.variables_initializer(my_partial_resnet.get_params())
    session = keras.backend.get_session()
    my_partial_resnet.set_session(session)
    session.run(init)
    first_output = my_partial_resnet.predict(X)
    print('first_output.shape:', first_output.shape)
    my_partial_resnet.copyFromKerasLayers(partial_model.layers)
    output = my_partial_resnet.predict(X)
    diff = np.abs(output - keras_output).sum()
    if diff < 1e-10:
        print("Everything's great!")
    else:
        print('diff = %s' % diff)