from __future__ import print_function, division
from builtins import range, input
from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
import tensorflow as tf
if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()

def VGG16_AvgPool(shape):
    if False:
        i = 10
        return i + 15
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)
    i = vgg.input
    x = i
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            x = AveragePooling2D()(x)
        else:
            x = layer(x)
    return Model(i, x)

def VGG16_AvgPool_CutOff(shape, num_convs):
    if False:
        while True:
            i = 10
    if num_convs < 1 or num_convs > 13:
        print('num_convs must be in the range [1, 13]')
        return None
    model = VGG16_AvgPool(shape)
    n = 0
    output = None
    for layer in model.layers:
        if layer.__class__ == Conv2D:
            n += 1
        if n >= num_convs:
            output = layer.output
            break
    return Model(model.input, output)

def unpreprocess(img):
    if False:
        i = 10
        return i + 15
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img

def scale_img(x):
    if False:
        return 10
    x = x - x.min()
    x = x / x.max()
    return x
if __name__ == '__main__':
    path = 'content/elephant.jpg'
    img = image.load_img(path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    batch_shape = x.shape
    shape = x.shape[1:]
    content_model = VGG16_AvgPool_CutOff(shape, 11)
    target = K.variable(content_model.predict(x))
    loss = K.mean(K.square(target - content_model.output))
    grads = K.gradients(loss, content_model.input)
    get_loss_and_grads = K.function(inputs=[content_model.input], outputs=[loss] + grads)

    def get_loss_and_grads_wrapper(x_vec):
        if False:
            while True:
                i = 10
        (l, g) = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return (l.astype(np.float64), g.flatten().astype(np.float64))
    from datetime import datetime
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i in range(10):
        (x, l, _) = fmin_l_bfgs_b(func=get_loss_and_grads_wrapper, x0=x, maxfun=20)
        x = np.clip(x, -127, 127)
        print('iter=%s, loss=%s' % (i, l))
        losses.append(l)
    print('duration:', datetime.now() - t0)
    plt.plot(losses)
    plt.show()
    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)
    plt.imshow(scale_img(final_img[0]))
    plt.show()