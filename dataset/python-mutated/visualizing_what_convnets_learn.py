"""
Title: Visualizing what convnets learn
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/05/29
Last modified: 2020/05/29
Description: Displaying the visual patterns that convnet filters respond to.
Accelerator: GPU
"""
"\n## Introduction\n\nIn this example, we look into what sort of visual patterns image classification models\nlearn. We'll be using the `ResNet50V2` model, trained on the ImageNet dataset.\n\nOur process is simple: we will create input images that maximize the activation of\nspecific filters in a target layer (picked somewhere in the middle of the model: layer\n`conv3_block4_out`). Such images represent a visualization of the\npattern that the filter responds to.\n"
'\n## Setup\n'
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import numpy as np
import tensorflow as tf
img_width = 180
img_height = 180
layer_name = 'conv3_block4_out'
'\n## Build a feature extraction model\n'
model = keras.applications.ResNet50V2(weights='imagenet', include_top=False)
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
'\n## Set up the gradient ascent process\n\nThe "loss" we will maximize is simply the mean of the activation of a specific filter in\nour target layer. To avoid border effects, we exclude border pixels.\n'

def compute_loss(input_image, filter_index):
    if False:
        return 10
    activation = feature_extractor(input_image)
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)
'\nOur gradient ascent function simply computes the gradients of the loss above\nwith regard to the input image, and update the update image so as to move it\ntowards a state that will activate the target filter more strongly.\n'

@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    if False:
        i = 10
        return i + 15
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    grads = tape.gradient(loss, img)
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return (loss, img)
'\n## Set up the end-to-end filter visualization loop\n\nOur process is as follow:\n\n- Start from a random image that is close to "all gray" (i.e. visually netural)\n- Repeatedly apply the gradient ascent step function defined above\n- Convert the resulting input image back to a displayable form, by normalizing it,\ncenter-cropping it, and restricting it to the [0, 255] range.\n'

def initialize_image():
    if False:
        while True:
            i = 10
    img = tf.random.uniform((1, img_width, img_height, 3))
    return (img - 0.5) * 0.25

def visualize_filter(filter_index):
    if False:
        while True:
            i = 10
    iterations = 30
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        (loss, img) = gradient_ascent_step(img, filter_index, learning_rate)
    img = deprocess_image(img[0].numpy())
    return (loss, img)

def deprocess_image(img):
    if False:
        i = 10
        return i + 15
    img -= img.mean()
    img /= img.std() + 1e-05
    img *= 0.15
    img = img[25:-25, 25:-25, :]
    img += 0.5
    img = np.clip(img, 0, 1)
    img *= 255
    img = np.clip(img, 0, 255).astype('uint8')
    return img
"\nLet's try it out with filter 0 in the target layer:\n"
from IPython.display import Image, display
(loss, img) = visualize_filter(0)
keras.utils.save_img('0.png', img)
'\nThis is what an input that maximizes the response of filter 0 in the target layer would\nlook like:\n'
display(Image('0.png'))
"\n## Visualize the first 64 filters in the target layer\n\nNow, let's make a 8x8 grid of the first 64 filters\nin the target layer to get of feel for the range\nof different visual patterns that the model has learned.\n"
all_imgs = []
for filter_index in range(64):
    print('Processing filter %d' % (filter_index,))
    (loss, img) = visualize_filter(filter_index)
    all_imgs.append(img)
margin = 5
n = 8
cropped_width = img_width - 25 * 2
cropped_height = img_height - 25 * 2
width = n * cropped_width + (n - 1) * margin
height = n * cropped_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))
for i in range(n):
    for j in range(n):
        img = all_imgs[i * n + j]
        stitched_filters[(cropped_width + margin) * i:(cropped_width + margin) * i + cropped_width, (cropped_height + margin) * j:(cropped_height + margin) * j + cropped_height, :] = img
keras.utils.save_img('stiched_filters.png', stitched_filters)
from IPython.display import Image, display
display(Image('stiched_filters.png'))
'\nImage classification models see the world by decomposing their inputs over a "vector\nbasis" of texture filters such as these.\n\nSee also\n[this old blog post](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)\nfor analysis and interpretation.\n\nExample available on HuggingFace.\n\n[![Generic badge](https://img.shields.io/badge/🤗%20Spaces-What%20Convnets%20Learn-black.svg)](https://huggingface.co/spaces/keras-io/what-convnets-learn)\n'