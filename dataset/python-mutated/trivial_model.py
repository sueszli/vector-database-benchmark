"""A trivial model for Keras."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models

def trivial_model(num_classes):
    if False:
        i = 10
        return i + 15
    'Trivial model for ImageNet dataset.'
    input_shape = (224, 224, 3)
    img_input = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda x: backend.reshape(x, [-1, 224 * 224 * 3]), name='reshape')(img_input)
    x = layers.Dense(1, name='fc1')(x)
    x = layers.Dense(num_classes, name='fc1000')(x)
    x = layers.Activation('softmax', dtype='float32')(x)
    return models.Model(img_input, x, name='trivial')