"""
Title: Customizing the convolution operation of a Conv2D layer
Author: [lukewood](https://lukewood.xyz)
Date created: 11/03/2021
Last modified: 11/03/2021
Description: This example shows how to implement custom convolution layers using the `Conv.convolution_op()` API.
Accelerator: GPU
"""
'\n## Introduction\n\nYou may sometimes need to implement custom versions of convolution layers like `Conv1D` and `Conv2D`.\nKeras enables you do this without implementing the entire layer from scratch: you can reuse\nmost of the base convolution layer and just customize the convolution op itself via the\n`convolution_op()` method.\n\nThis method was introduced in Keras 2.7. So before using the\n`convolution_op()` API, ensure that you are running Keras version 2.7.0 or greater.\n'
'\n## A Simple `StandardizedConv2D` implementation\n\nThere are two ways to use the `Conv.convolution_op()` API. The first way\nis to override the `convolution_op()` method on a convolution layer subclass.\nUsing this approach, we can quickly implement a\n[StandardizedConv2D](https://arxiv.org/abs/1903.10520) as shown below.\n'
import tensorflow as tf
import keras
from keras import layers
import numpy as np

class StandardizedConv2DWithOverride(layers.Conv2D):

    def convolution_op(self, inputs, kernel):
        if False:
            print('Hello World!')
        (mean, var) = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        return tf.nn.conv2d(inputs, (kernel - mean) / tf.sqrt(var + 1e-10), padding='valid', strides=list(self.strides), name=self.__class__.__name__)
'\nThe other way to use the `Conv.convolution_op()` API is to directly call the\n`convolution_op()` method from the `call()` method of a convolution layer subclass.\nA comparable class implemented using this approach is shown below.\n'

class StandardizedConv2DWithCall(layers.Conv2D):

    def call(self, inputs):
        if False:
            return 10
        (mean, var) = tf.nn.moments(self.kernel, axes=[0, 1, 2], keepdims=True)
        result = self.convolution_op(inputs, (self.kernel - mean) / tf.sqrt(var + 1e-10))
        if self.use_bias:
            result = result + self.bias
        return result
'\n## Example Usage\n\nBoth of these layers work as drop-in replacements for `Conv2D`. The following\ndemonstration performs classification on the MNIST dataset.\n'
num_classes = 10
input_shape = (28, 28, 1)
((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = keras.Sequential([keras.layers.Input(shape=input_shape), StandardizedConv2DWithCall(32, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), StandardizedConv2DWithOverride(64, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), layers.Flatten(), layers.Dropout(0.5), layers.Dense(num_classes, activation='softmax')])
model.summary()
'\n\n'
batch_size = 128
epochs = 5
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_split=0.1)
'\n## Conclusion\n\nThe `Conv.convolution_op()` API provides an easy and readable way to implement custom\nconvolution layers. A `StandardizedConvolution` implementation using the API is quite\nterse, consisting of only four lines of code.\n'