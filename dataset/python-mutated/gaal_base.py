"""Base file for Generative Adversarial Active Learning.
Part of the codes are adapted from
https://github.com/leibinghe/GAAL-based-outlier-detection
"""
from __future__ import division
from __future__ import print_function
import math
from .base_dl import _get_tensorflow_version
if _get_tensorflow_version() <= 200:
    import keras
    from keras.layers import Input, Dense
    from keras.models import Sequential, Model
else:
    import tensorflow.keras as keras
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Sequential, Model

def create_discriminator(latent_size, data_size):
    if False:
        print('Hello World!')
    'Create the discriminator of the GAN for a given latent size.\n\n    Parameters\n    ----------\n    latent_size : int\n        The size of the latent space of the generator.\n\n    data_size : int\n        Size of the input data.\n\n    Returns\n    -------\n    D : Keras model() object\n        Returns a model() object.\n    '
    dis = Sequential()
    dis.add(Dense(int(math.ceil(math.sqrt(data_size))), input_dim=latent_size, activation='relu', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    dis.add(Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    data = Input(shape=(latent_size,))
    fake = dis(data)
    return Model(data, fake)

def create_generator(latent_size):
    if False:
        for i in range(10):
            print('nop')
    'Create the generator of the GAN for a given latent size.\n\n    Parameters\n    ----------\n    latent_size : int\n        The size of the latent space of the generator\n\n    Returns\n    -------\n    D : Keras model() object\n        Returns a model() object.\n    '
    gen = Sequential()
    gen.add(Dense(latent_size, input_dim=latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
    gen.add(Dense(latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
    latent = Input(shape=(latent_size,))
    fake_data = gen(latent)
    return Model(latent, fake_data)