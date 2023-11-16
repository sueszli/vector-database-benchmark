"""
Title: Image Super-Resolution using an Efficient Sub-Pixel CNN
Author: [Xingyu Long](https://github.com/xingyu-long)
Date created: 2020/07/28
Last modified: 2020/08/27
Description: Implementing Super-Resolution using Efficient sub-pixel model on BSDS500.
Accelerator: GPU
Converted to Keras 3 by: [Md Awsfalur Rahman](https://awsaf49.github.io)
"""
'\n## Introduction\n\nESPCN (Efficient Sub-Pixel CNN), proposed by [Shi, 2016](https://arxiv.org/abs/1609.05158)\nis a model that reconstructs a high-resolution version of an image given a low-resolution\nversion.\nIt leverages efficient "sub-pixel convolution" layers, which learns an array of\nimage upscaling filters.\n\nIn this code example, we will implement the model from the paper and train it on a small\ndataset,\n[BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).\n[BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).\n'
'\n## Setup\n'
import keras
from keras import layers
from keras import ops
from keras.utils import load_img
from keras.utils import array_to_img
from keras.utils import img_to_array
from keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
import os
import math
import numpy as np
from IPython.display import display
'\n## Load data: BSDS500 dataset\n\n### Download dataset\n\nWe use the built-in `keras.utils.get_file` utility to retrieve the dataset.\n'
dataset_url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz'
data_dir = keras.utils.get_file(origin=dataset_url, fname='BSR', untar=True)
root_dir = os.path.join(data_dir, 'BSDS500/data')
'\nWe create training and validation datasets via `image_dataset_from_directory`.\n'
crop_size = 300
upscale_factor = 3
input_size = crop_size // upscale_factor
batch_size = 8
train_ds = image_dataset_from_directory(root_dir, batch_size=batch_size, image_size=(crop_size, crop_size), validation_split=0.2, subset='training', seed=1337, label_mode=None)
valid_ds = image_dataset_from_directory(root_dir, batch_size=batch_size, image_size=(crop_size, crop_size), validation_split=0.2, subset='validation', seed=1337, label_mode=None)
'\nWe rescale the images to take values in the range [0, 1].\n'

def scaling(input_image):
    if False:
        return 10
    input_image = input_image / 255.0
    return input_image
train_ds = train_ds.map(scaling)
valid_ds = valid_ds.map(scaling)
"\nLet's visualize a few sample images:\n"
for batch in train_ds.take(1):
    for img in batch:
        display(array_to_img(img))
'\nWe prepare a dataset of test image paths that we will use for\nvisual evaluation at the end of this example.\n'
dataset = os.path.join(root_dir, 'images')
test_path = os.path.join(dataset, 'test')
test_img_paths = sorted([os.path.join(test_path, fname) for fname in os.listdir(test_path) if fname.endswith('.jpg')])
"\n## Crop and resize images\n\nLet's process image data.\nFirst, we convert our images from the RGB color space to the\n[YUV colour space](https://en.wikipedia.org/wiki/YUV).\n\nFor the input data (low-resolution images),\nwe crop the image, retrieve the `y` channel (luninance),\nand resize it with the `area` method (use `BICUBIC` if you use PIL).\nWe only consider the luminance channel\nin the YUV color space because humans are more sensitive to\nluminance change.\n\nFor the target data (high-resolution images), we just crop the image\nand retrieve the `y` channel.\n"

def process_input(input, input_size, upscale_factor):
    if False:
        while True:
            i = 10
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    (y, u, v) = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method='area')

def process_target(input):
    if False:
        while True:
            i = 10
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    (y, u, v) = tf.split(input, 3, axis=last_dimension_axis)
    return y
train_ds = train_ds.map(lambda x: (process_input(x, input_size, upscale_factor), process_target(x)))
train_ds = train_ds.prefetch(buffer_size=32)
valid_ds = valid_ds.map(lambda x: (process_input(x, input_size, upscale_factor), process_target(x)))
valid_ds = valid_ds.prefetch(buffer_size=32)
"\nLet's take a look at the input and target data.\n"
for batch in train_ds.take(1):
    for img in batch[0]:
        display(array_to_img(img))
    for img in batch[1]:
        display(array_to_img(img))
'\n## Build a model\n\nCompared to the paper, we add one more layer and we use the `relu` activation function\ninstead of `tanh`.\nIt achieves better performance even though we train the model for fewer epochs.\n'

class DepthToSpace(layers.Layer):

    def __init__(self, block_size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.block_size = block_size

    def call(self, input):
        if False:
            print('Hello World!')
        (batch, height, width, depth) = ops.shape(input)
        depth = depth // self.block_size ** 2
        x = ops.reshape(input, [batch, height, width, self.block_size, self.block_size, depth])
        x = ops.transpose(x, [0, 1, 3, 2, 4, 5])
        x = ops.reshape(x, [batch, height * self.block_size, width * self.block_size, depth])
        return x

def get_model(upscale_factor=3, channels=1):
    if False:
        print('Hello World!')
    conv_args = {'activation': 'relu', 'kernel_initializer': 'orthogonal', 'padding': 'same'}
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels * upscale_factor ** 2, 3, **conv_args)(x)
    outputs = DepthToSpace(upscale_factor)(x)
    return keras.Model(inputs, outputs)
'\n## Define utility functions\n\nWe need to define several utility functions to monitor our results:\n\n- `plot_results` to plot an save an image.\n- `get_lowres_image` to convert an image to its low-resolution version.\n- `upscale_image` to turn a low-resolution image to\na high-resolution version reconstructed by the model.\nIn this function, we use the `y` channel from the YUV color space\nas input to the model and then combine the output with the\nother channels to obtain an RGB image.\n'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import PIL

def plot_results(img, prefix, title):
    if False:
        i = 10
        return i + 15
    'Plot the result with zoom-in area.'
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    (fig, ax) = plt.subplots()
    im = ax.imshow(img_array[::-1], origin='lower')
    plt.title(title)
    axins = zoomed_inset_axes(ax, 2, loc=2)
    axins.imshow(img_array[::-1], origin='lower')
    (x1, x2, y1, y2) = (200, 300, 100, 200)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    mark_inset(ax, axins, loc1=1, loc2=3, fc='none', ec='blue')
    plt.savefig(str(prefix) + '-' + title + '.png')
    plt.show()

def get_lowres_image(img, upscale_factor):
    if False:
        for i in range(10):
            print('nop')
    'Return low-resolution image to use as model input.'
    return img.resize((img.size[0] // upscale_factor, img.size[1] // upscale_factor), PIL.Image.BICUBIC)

def upscale_image(model, img):
    if False:
        while True:
            i = 10
    'Predict the result based on input image and restore the image as RGB.'
    ycbcr = img.convert('YCbCr')
    (y, cb, cr) = ycbcr.split()
    y = img_to_array(y)
    y = y.astype('float32') / 255.0
    input = np.expand_dims(y, axis=0)
    out = model.predict(input)
    out_img_y = out[0]
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode='L')
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge('YCbCr', (out_img_y, out_img_cb, out_img_cr)).convert('RGB')
    return out_img
'\n## Define callbacks to monitor training\n\nThe `ESPCNCallback` object will compute and display\nthe [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) metric.\nThis is the main metric we use to evaluate super-resolution performance.\n'

class ESPCNCallback(keras.callbacks.Callback):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.test_img = get_lowres_image(load_img(test_img_paths[0]), upscale_factor)

    def on_epoch_begin(self, epoch, logs=None):
        if False:
            return 10
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        if False:
            print('Hello World!')
        print('Mean PSNR for epoch: %.2f' % np.mean(self.psnr))
        if epoch % 20 == 0:
            prediction = upscale_image(self.model, self.test_img)
            plot_results(prediction, 'epoch-' + str(epoch), 'prediction')

    def on_test_batch_end(self, batch, logs=None):
        if False:
            i = 10
            return i + 15
        self.psnr.append(10 * math.log10(1 / logs['loss']))
'\nDefine `ModelCheckpoint` and `EarlyStopping` callbacks.\n'
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
checkpoint_filepath = '/tmp/checkpoint.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='loss', mode='min', save_best_only=True)
model = get_model(upscale_factor=upscale_factor, channels=1)
model.summary()
callbacks = [ESPCNCallback(), early_stopping_callback, model_checkpoint_callback]
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
'\n## Train the model\n'
epochs = 100
model.compile(optimizer=optimizer, loss=loss_fn)
model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=2)
model.load_weights(checkpoint_filepath)
"\n## Run model prediction and plot the results\n\nLet's compute the reconstructed version of a few images and save the results.\n"
total_bicubic_psnr = 0.0
total_test_psnr = 0.0
for (index, test_img_path) in enumerate(test_img_paths[50:60]):
    img = load_img(test_img_path)
    lowres_input = get_lowres_image(img, upscale_factor)
    w = lowres_input.size[0] * upscale_factor
    h = lowres_input.size[1] * upscale_factor
    highres_img = img.resize((w, h))
    prediction = upscale_image(model, lowres_input)
    lowres_img = lowres_input.resize((w, h))
    lowres_img_arr = img_to_array(lowres_img)
    highres_img_arr = img_to_array(highres_img)
    predict_img_arr = img_to_array(prediction)
    bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
    test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)
    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr
    print('PSNR of low resolution image and high resolution image is %.4f' % bicubic_psnr)
    print('PSNR of predict and high resolution is %.4f' % test_psnr)
    plot_results(lowres_img, index, 'lowres')
    plot_results(highres_img, index, 'highres')
    plot_results(prediction, index, 'prediction')
print('Avg. PSNR of lowres images is %.4f' % (total_bicubic_psnr / 10))
print('Avg. PSNR of reconstructions is %.4f' % (total_test_psnr / 10))