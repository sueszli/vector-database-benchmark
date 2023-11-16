"""
Title: PixelCNN
Author: [ADMoreau](https://github.com/ADMoreau)
Date created: 2020/05/17
Last modified: 2020/05/23
Description: PixelCNN implemented in Keras.
Accelerator: GPU
"""
'\n## Introduction\n\nPixelCNN is a generative model proposed in 2016 by van den Oord et al.\n(reference: [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328)).\nIt is designed to generate images (or other data types) iteratively\nfrom an input vector where the probability distribution of prior elements dictates the\nprobability distribution of later elements. In the following example, images are generated\nin this fashion, pixel-by-pixel, via a masked convolution kernel that only looks at data\nfrom previously generated pixels (origin at the top left) to generate later pixels.\nDuring inference, the output of the network is used as a probability ditribution\nfrom which new pixel values are sampled to generate a new image\n(here, with MNIST, the pixels values are either black or white).\n'
import numpy as np
import keras
from keras import layers
from keras import ops
from tqdm import tqdm
'\n## Getting the Data\n'
num_classes = 10
input_shape = (28, 28, 1)
n_residual_blocks = 5
((x, _), (y, _)) = keras.datasets.mnist.load_data()
data = np.concatenate((x, y), axis=0)
data = np.where(data < 0.33 * 256, 0, 1)
data = data.astype(np.float32)
'\n## Create two classes for the requisite Layers for the model\n'

class PixelConvLayer(layers.Layer):

    def __init__(self, mask_type, **kwargs):
        if False:
            return 10
        super().__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        if False:
            i = 10
            return i + 15
        self.conv.build(input_shape)
        kernel_shape = ops.shape(self.conv.kernel)
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[:kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, :kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == 'B':
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        if False:
            while True:
                i = 10
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

class ResidualBlock(keras.layers.Layer):

    def __init__(self, filters, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(filters=filters, kernel_size=1, activation='relu')
        self.pixel_conv = PixelConvLayer(mask_type='B', filters=filters // 2, kernel_size=3, activation='relu', padding='same')
        self.conv2 = keras.layers.Conv2D(filters=filters, kernel_size=1, activation='relu')

    def call(self, inputs):
        if False:
            while True:
                i = 10
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])
'\n## Build the model based on the original paper\n'
inputs = keras.Input(shape=input_shape)
x = PixelConvLayer(mask_type='A', filters=128, kernel_size=7, activation='relu', padding='same')(inputs)
for _ in range(n_residual_blocks):
    x = ResidualBlock(filters=128)(x)
for _ in range(2):
    x = PixelConvLayer(mask_type='B', filters=128, kernel_size=1, strides=1, activation='relu', padding='valid')(x)
out = keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, activation='sigmoid', padding='valid')(x)
pixel_cnn = keras.Model(inputs, out)
adam = keras.optimizers.Adam(learning_rate=0.0005)
pixel_cnn.compile(optimizer=adam, loss='binary_crossentropy')
pixel_cnn.summary()
pixel_cnn.fit(x=data, y=data, batch_size=128, epochs=50, validation_split=0.1, verbose=2)
'\n## Demonstration\n\nThe PixelCNN cannot generate the full image at once. Instead, it must generate each pixel in\norder, append the last generated pixel to the current image, and feed the image back into the\nmodel to repeat the process.\n'
from IPython.display import Image, display
batch = 4
pixels = np.zeros(shape=(batch,) + pixel_cnn.input_shape[1:])
(batch, rows, cols, channels) = pixels.shape
for row in tqdm(range(rows)):
    for col in range(cols):
        for channel in range(channels):
            probs = pixel_cnn.predict(pixels)[:, row, col, channel]
            pixels[:, row, col, channel] = ops.ceil(probs - keras.random.uniform(probs.shape))

def deprocess_image(x):
    if False:
        return 10
    x = np.stack((x, x, x), 2)
    x *= 255.0
    x = np.clip(x, 0, 255).astype('uint8')
    return x
for (i, pic) in enumerate(pixels):
    keras.utils.save_img('generated_image_{}.png'.format(i), deprocess_image(np.squeeze(pic, -1)))
display(Image('generated_image_0.png'))
display(Image('generated_image_1.png'))
display(Image('generated_image_2.png'))
display(Image('generated_image_3.png'))