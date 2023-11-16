"""
Title: Zero-DCE for low-light image enhancement
Author: [Soumik Rakshit](http://github.com/soumik12345)
Converted to Keras 3 by: [Soumik Rakshit](http://github.com/soumik12345)
Date created: 2021/09/18
Last modified: 2023/07/15
Description: Implementing Zero-Reference Deep Curve Estimation for low-light image enhancement.
Accelerator: GPU
"""
'\n## Introduction\n\n**Zero-Reference Deep Curve Estimation** or **Zero-DCE** formulates low-light image\nenhancement as the task of estimating an image-specific\n[*tonal curve*](https://en.wikipedia.org/wiki/Curve_(tonality)) with a deep neural network.\nIn this example, we train a lightweight deep network, **DCE-Net**, to estimate\npixel-wise and high-order tonal curves for dynamic range adjustment of a given image.\n\nZero-DCE takes a low-light image as input and produces high-order tonal curves as its output.\nThese curves are then used for pixel-wise adjustment on the dynamic range of the input to\nobtain an enhanced image. The curve estimation process is done in such a way that it maintains\nthe range of the enhanced image and preserves the contrast of neighboring pixels. This\ncurve estimation is inspired by curves adjustment used in photo editing software such as\nAdobe Photoshop where users can adjust points throughout an image’s tonal range.\n\nZero-DCE is appealing because of its relaxed assumptions with regard to reference images:\nit does not require any input/output image pairs during training.\nThis is achieved through a set of carefully formulated non-reference loss functions,\nwhich implicitly measure the enhancement quality and guide the training of the network.\n\n### References\n\n- [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement](https://arxiv.org/pdf/2001.06826.pdf)\n- [Curves adjustment in Adobe Photoshop](https://helpx.adobe.com/photoshop/using/curves-adjustment.html)\n'
'\n## Downloading LOLDataset\n\nThe **LoL Dataset** has been created for low-light image enhancement. It provides 485\nimages for training and 15 for testing. Each image pair in the dataset consists of a\nlow-light input image and its corresponding well-exposed reference image.\n'
import os
import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import keras
from keras import layers
import tensorflow as tf
'shell\nwget https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip\nunzip -q lol_dataset.zip && rm lol_dataset.zip\n'
'\n## Creating a TensorFlow Dataset\n\nWe use 300 low-light images from the LoL Dataset training set for training, and we use\nthe remaining 185 low-light images for validation. We resize the images to size `256 x\n256` to be used for both training and validation. Note that in order to train the DCE-Net,\nwe will not require the corresponding enhanced images.\n'
IMAGE_SIZE = 256
BATCH_SIZE = 16
MAX_TRAIN_IMAGES = 400

def load_data(image_path):
    if False:
        return 10
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    image = image / 255.0
    return image

def data_generator(low_light_images):
    if False:
        return 10
    dataset = tf.data.Dataset.from_tensor_slices(low_light_images)
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset
train_low_light_images = sorted(glob('./lol_dataset/our485/low/*'))[:MAX_TRAIN_IMAGES]
val_low_light_images = sorted(glob('./lol_dataset/our485/low/*'))[MAX_TRAIN_IMAGES:]
test_low_light_images = sorted(glob('./lol_dataset/eval15/low/*'))
train_dataset = data_generator(train_low_light_images)
val_dataset = data_generator(val_low_light_images)
print('Train Dataset:', train_dataset)
print('Validation Dataset:', val_dataset)
'\n## The Zero-DCE Framework\n\nThe goal of DCE-Net is to estimate a set of best-fitting light-enhancement curves\n(LE-curves) given an input image. The framework then maps all pixels of the input’s RGB\nchannels by applying the curves iteratively to obtain the final enhanced image.\n\n### Understanding light-enhancement curves\n\nA ligh-enhancement curve is a kind of curve that can map a low-light image\nto its enhanced version automatically,\nwhere the self-adaptive curve parameters are solely dependent on the input image.\nWhen designing such a curve, three objectives should be taken into account:\n\n- Each pixel value of the enhanced image should be in the normalized range `[0,1]`, in order to\navoid information loss induced by overflow truncation.\n- It should be monotonous, to preserve the contrast between neighboring pixels.\n- The shape of this curve should be as simple as possible,\nand the curve should be differentiable to allow backpropagation.\n\nThe light-enhancement curve is separately applied to three RGB channels instead of solely on the\nillumination channel. The three-channel adjustment can better preserve the inherent color and reduce\nthe risk of over-saturation.\n\n![](https://li-chongyi.github.io/Zero-DCE_files/framework.png)\n\n### DCE-Net\n\nThe DCE-Net is a lightweight deep neural network that learns the mapping between an input\nimage and its best-fitting curve parameter maps. The input to the DCE-Net is a low-light\nimage while the outputs are a set of pixel-wise curve parameter maps for corresponding\nhigher-order curves. It is a plain CNN of seven convolutional layers with symmetrical\nconcatenation. Each layer consists of 32 convolutional kernels of size 3×3 and stride 1\nfollowed by the ReLU activation function. The last convolutional layer is followed by the\nTanh activation function, which produces 24 parameter maps for 8 iterations, where each\niteration requires three curve parameter maps for the three channels.\n\n![](https://i.imgur.com/HtIg34W.png)\n'

def build_dce_net():
    if False:
        for i in range(10):
            print('nop')
    input_img = keras.Input(shape=[None, None, 3])
    conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same')(input_img)
    conv2 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same')(conv1)
    conv3 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same')(conv2)
    conv4 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same')(conv3)
    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    conv5 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same')(int_con1)
    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same')(int_con2)
    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    x_r = layers.Conv2D(24, (3, 3), strides=(1, 1), activation='tanh', padding='same')(int_con3)
    return keras.Model(inputs=input_img, outputs=x_r)
'\n## Loss functions\n\nTo enable zero-reference learning in DCE-Net, we use a set of differentiable\nzero-reference losses that allow us to evaluate the quality of enhanced images.\n'
'\n### Color constancy loss\n\nThe *color constancy loss* is used to correct the potential color deviations in the\nenhanced image.\n'

def color_constancy_loss(x):
    if False:
        for i in range(10):
            print('nop')
    mean_rgb = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
    (mr, mg, mb) = (mean_rgb[:, :, :, 0], mean_rgb[:, :, :, 1], mean_rgb[:, :, :, 2])
    d_rg = tf.square(mr - mg)
    d_rb = tf.square(mr - mb)
    d_gb = tf.square(mb - mg)
    return tf.sqrt(tf.square(d_rg) + tf.square(d_rb) + tf.square(d_gb))
'\n### Exposure loss\n\nTo restrain under-/over-exposed regions, we use the *exposure control loss*.\nIt measures the distance between the average intensity value of a local region\nand a preset well-exposedness level (set to `0.6`).\n'

def exposure_loss(x, mean_val=0.6):
    if False:
        for i in range(10):
            print('nop')
    x = tf.reduce_mean(x, axis=3, keepdims=True)
    mean = tf.nn.avg_pool2d(x, ksize=16, strides=16, padding='VALID')
    return tf.reduce_mean(tf.square(mean - mean_val))
'\n### Illumination smoothness loss\n\nTo preserve the monotonicity relations between neighboring pixels, the\n*illumination smoothness loss* is added to each curve parameter map.\n'

def illumination_smoothness_loss(x):
    if False:
        print('Hello World!')
    batch_size = tf.shape(x)[0]
    h_x = tf.shape(x)[1]
    w_x = tf.shape(x)[2]
    count_h = (tf.shape(x)[2] - 1) * tf.shape(x)[3]
    count_w = tf.shape(x)[2] * (tf.shape(x)[3] - 1)
    h_tv = tf.reduce_sum(tf.square(x[:, 1:, :, :] - x[:, :h_x - 1, :, :]))
    w_tv = tf.reduce_sum(tf.square(x[:, :, 1:, :] - x[:, :, :w_x - 1, :]))
    batch_size = tf.cast(batch_size, dtype=tf.float32)
    count_h = tf.cast(count_h, dtype=tf.float32)
    count_w = tf.cast(count_w, dtype=tf.float32)
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
'\n### Spatial consistency loss\n\nThe *spatial consistency loss* encourages spatial coherence of the enhanced image by\npreserving the contrast between neighboring regions across the input image and its enhanced version.\n'

class SpatialConsistencyLoss(keras.losses.Loss):

    def __init__(self, **kwargs):
        if False:
            return 10
        super().__init__(reduction='none')
        self.left_kernel = tf.constant([[[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32)
        self.right_kernel = tf.constant([[[[0, 0, 0]], [[0, 1, -1]], [[0, 0, 0]]]], dtype=tf.float32)
        self.up_kernel = tf.constant([[[[0, -1, 0]], [[0, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32)
        self.down_kernel = tf.constant([[[[0, 0, 0]], [[0, 1, 0]], [[0, -1, 0]]]], dtype=tf.float32)

    def call(self, y_true, y_pred):
        if False:
            i = 10
            return i + 15
        original_mean = tf.reduce_mean(y_true, 3, keepdims=True)
        enhanced_mean = tf.reduce_mean(y_pred, 3, keepdims=True)
        original_pool = tf.nn.avg_pool2d(original_mean, ksize=4, strides=4, padding='VALID')
        enhanced_pool = tf.nn.avg_pool2d(enhanced_mean, ksize=4, strides=4, padding='VALID')
        d_original_left = tf.nn.conv2d(original_pool, self.left_kernel, strides=[1, 1, 1, 1], padding='SAME')
        d_original_right = tf.nn.conv2d(original_pool, self.right_kernel, strides=[1, 1, 1, 1], padding='SAME')
        d_original_up = tf.nn.conv2d(original_pool, self.up_kernel, strides=[1, 1, 1, 1], padding='SAME')
        d_original_down = tf.nn.conv2d(original_pool, self.down_kernel, strides=[1, 1, 1, 1], padding='SAME')
        d_enhanced_left = tf.nn.conv2d(enhanced_pool, self.left_kernel, strides=[1, 1, 1, 1], padding='SAME')
        d_enhanced_right = tf.nn.conv2d(enhanced_pool, self.right_kernel, strides=[1, 1, 1, 1], padding='SAME')
        d_enhanced_up = tf.nn.conv2d(enhanced_pool, self.up_kernel, strides=[1, 1, 1, 1], padding='SAME')
        d_enhanced_down = tf.nn.conv2d(enhanced_pool, self.down_kernel, strides=[1, 1, 1, 1], padding='SAME')
        d_left = tf.square(d_original_left - d_enhanced_left)
        d_right = tf.square(d_original_right - d_enhanced_right)
        d_up = tf.square(d_original_up - d_enhanced_up)
        d_down = tf.square(d_original_down - d_enhanced_down)
        return d_left + d_right + d_up + d_down
'\n### Deep curve estimation model\n\nWe implement the Zero-DCE framework as a Keras subclassed model.\n'

class ZeroDCE(keras.Model):

    def __init__(self, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.dce_model = build_dce_net()

    def compile(self, learning_rate, **kwargs):
        if False:
            while True:
                i = 10
        super().compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.spatial_constancy_loss = SpatialConsistencyLoss(reduction='none')
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.illumination_smoothness_loss_tracker = keras.metrics.Mean(name='illumination_smoothness_loss')
        self.spatial_constancy_loss_tracker = keras.metrics.Mean(name='spatial_constancy_loss')
        self.color_constancy_loss_tracker = keras.metrics.Mean(name='color_constancy_loss')
        self.exposure_loss_tracker = keras.metrics.Mean(name='exposure_loss')

    @property
    def metrics(self):
        if False:
            return 10
        return [self.total_loss_tracker, self.illumination_smoothness_loss_tracker, self.spatial_constancy_loss_tracker, self.color_constancy_loss_tracker, self.exposure_loss_tracker]

    def get_enhanced_image(self, data, output):
        if False:
            print('Hello World!')
        r1 = output[:, :, :, :3]
        r2 = output[:, :, :, 3:6]
        r3 = output[:, :, :, 6:9]
        r4 = output[:, :, :, 9:12]
        r5 = output[:, :, :, 12:15]
        r6 = output[:, :, :, 15:18]
        r7 = output[:, :, :, 18:21]
        r8 = output[:, :, :, 21:24]
        x = data + r1 * (tf.square(data) - data)
        x = x + r2 * (tf.square(x) - x)
        x = x + r3 * (tf.square(x) - x)
        enhanced_image = x + r4 * (tf.square(x) - x)
        x = enhanced_image + r5 * (tf.square(enhanced_image) - enhanced_image)
        x = x + r6 * (tf.square(x) - x)
        x = x + r7 * (tf.square(x) - x)
        enhanced_image = x + r8 * (tf.square(x) - x)
        return enhanced_image

    def call(self, data):
        if False:
            i = 10
            return i + 15
        dce_net_output = self.dce_model(data)
        return self.get_enhanced_image(data, dce_net_output)

    def compute_losses(self, data, output):
        if False:
            return 10
        enhanced_image = self.get_enhanced_image(data, output)
        loss_illumination = 200 * illumination_smoothness_loss(output)
        loss_spatial_constancy = tf.reduce_mean(self.spatial_constancy_loss(enhanced_image, data))
        loss_color_constancy = 5 * tf.reduce_mean(color_constancy_loss(enhanced_image))
        loss_exposure = 10 * tf.reduce_mean(exposure_loss(enhanced_image))
        total_loss = loss_illumination + loss_spatial_constancy + loss_color_constancy + loss_exposure
        return {'total_loss': total_loss, 'illumination_smoothness_loss': loss_illumination, 'spatial_constancy_loss': loss_spatial_constancy, 'color_constancy_loss': loss_color_constancy, 'exposure_loss': loss_exposure}

    def train_step(self, data):
        if False:
            print('Hello World!')
        with tf.GradientTape() as tape:
            output = self.dce_model(data)
            losses = self.compute_losses(data, output)
        gradients = tape.gradient(losses['total_loss'], self.dce_model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.dce_model.trainable_weights))
        self.total_loss_tracker.update_state(losses['total_loss'])
        self.illumination_smoothness_loss_tracker.update_state(losses['illumination_smoothness_loss'])
        self.spatial_constancy_loss_tracker.update_state(losses['spatial_constancy_loss'])
        self.color_constancy_loss_tracker.update_state(losses['color_constancy_loss'])
        self.exposure_loss_tracker.update_state(losses['exposure_loss'])
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        if False:
            return 10
        output = self.dce_model(data)
        losses = self.compute_losses(data, output)
        self.total_loss_tracker.update_state(losses['total_loss'])
        self.illumination_smoothness_loss_tracker.update_state(losses['illumination_smoothness_loss'])
        self.spatial_constancy_loss_tracker.update_state(losses['spatial_constancy_loss'])
        self.color_constancy_loss_tracker.update_state(losses['color_constancy_loss'])
        self.exposure_loss_tracker.update_state(losses['exposure_loss'])
        return {metric.name: metric.result() for metric in self.metrics}

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        if False:
            for i in range(10):
                print('nop')
        'While saving the weights, we simply save the weights of the DCE-Net'
        self.dce_model.save_weights(filepath, overwrite=overwrite, save_format=save_format, options=options)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        if False:
            print('Hello World!')
        'While loading the weights, we simply load the weights of the DCE-Net'
        self.dce_model.load_weights(filepath=filepath, by_name=by_name, skip_mismatch=skip_mismatch, options=options)
'\n## Training\n'
zero_dce_model = ZeroDCE()
zero_dce_model.compile(learning_rate=0.0001)
history = zero_dce_model.fit(train_dataset, validation_data=val_dataset, epochs=100)

def plot_result(item):
    if False:
        for i in range(10):
            print('nop')
    plt.plot(history.history[item], label=item)
    plt.plot(history.history['val_' + item], label='val_' + item)
    plt.xlabel('Epochs')
    plt.ylabel(item)
    plt.title('Train and Validation {} Over Epochs'.format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()
plot_result('total_loss')
plot_result('illumination_smoothness_loss')
plot_result('spatial_constancy_loss')
plot_result('color_constancy_loss')
plot_result('exposure_loss')
'\n## Inference\n'

def plot_results(images, titles, figure_size=(12, 12)):
    if False:
        print('Hello World!')
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis('off')
    plt.show()

def infer(original_image):
    if False:
        for i in range(10):
            print('nop')
    image = keras.utils.img_to_array(original_image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = zero_dce_model(image)
    output_image = tf.cast(output_image[0, :, :, :] * 255, dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    return output_image
'\n### Inference on test images\n\nWe compare the test images from LOLDataset enhanced by MIRNet with images enhanced via\nthe `PIL.ImageOps.autocontrast()` function.\n\nYou can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/low-light-image-enhancement)\nand try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/low-light-image-enhancement).\n'
for val_image_file in test_low_light_images:
    original_image = Image.open(val_image_file)
    enhanced_image = infer(original_image)
    plot_results([original_image, ImageOps.autocontrast(original_image), enhanced_image], ['Original', 'PIL Autocontrast', 'Enhanced'], (20, 12))