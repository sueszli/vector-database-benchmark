from __future__ import print_function, division
from builtins import range, input
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tf_resnet_convblock import ConvLayer, BatchNormLayer, ConvBlock

class PartialResNet:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def copyFromKerasLayers(self, layers):
        if False:
            print('Hello World!')
        pass

    def predict(self, X):
        if False:
            while True:
                i = 10
        pass

    def set_session(self, session):
        if False:
            i = 10
            return i + 15
        self.session = session

    def get_params(self):
        if False:
            i = 10
            return i + 15
        params = []
if __name__ == '__main__':
    resnet = ResNet50(weights='imagenet')
    partial_model = Model(inputs=resnet.input, outputs=resnet.layers[16].output)
    print(partial_model.summary())
    my_partial_resnet = PartialResNet()
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