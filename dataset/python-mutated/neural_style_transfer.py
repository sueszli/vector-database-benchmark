"""
Title: Neural style transfer
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2016/01/11
Last modified: 2020/05/02
Description: Transferring the style of a reference image to target image using gradient descent.
Accelerator: GPU
"""
'\n## Introduction\n\nStyle transfer consists in generating an image\nwith the same "content" as a base image, but with the\n"style" of a different picture (typically artistic).\nThis is achieved through the optimization of a loss function\nthat has 3 components: "style loss", "content loss",\nand "total variation loss":\n\n- The total variation loss imposes local spatial continuity between\nthe pixels of the combination image, giving it visual coherence.\n- The style loss is where the deep learning keeps in --that one is defined\nusing a deep convolutional neural network. Precisely, it consists in a sum of\nL2 distances between the Gram matrices of the representations of\nthe base image and the style reference image, extracted from\ndifferent layers of a convnet (trained on ImageNet). The general idea\nis to capture color/texture information at different spatial\nscales (fairly large scales --defined by the depth of the layer considered).\n- The content loss is a L2 distance between the features of the base\nimage (extracted from a deep layer) and the features of the combination image,\nkeeping the generated image close enough to the original one.\n\n**Reference:** [A Neural Algorithm of Artistic Style](\n  http://arxiv.org/abs/1508.06576)\n'
'\n## Setup\n'
import numpy as np
import tensorflow as tf
import keras
from keras.applications import vgg19
base_image_path = keras.utils.get_file('paris.jpg', 'https://i.imgur.com/F28w3Ac.jpg')
style_reference_image_path = keras.utils.get_file('starry_night.jpg', 'https://i.imgur.com/9ooB60I.jpg')
result_prefix = 'paris_generated'
total_variation_weight = 1e-06
style_weight = 1e-06
content_weight = 2.5e-08
(width, height) = keras.utils.load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)
"\n## Let's take a look at our base (content) image and our style reference image\n"
from IPython.display import Image, display
display(Image(base_image_path))
display(Image(style_reference_image_path))
'\n## Image preprocessing / deprocessing utilities\n'

def preprocess_image(image_path):
    if False:
        while True:
            i = 10
    img = keras.utils.load_img(image_path, target_size=(img_nrows, img_ncols))
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

def deprocess_image(x):
    if False:
        while True:
            i = 10
    x = x.reshape((img_nrows, img_ncols, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
'\n## Compute the style transfer loss\n\nFirst, we need to define 4 utility functions:\n\n- `gram_matrix` (used to compute the style loss)\n- The `style_loss` function, which keeps the generated image close to the local textures\nof the style reference image\n- The `content_loss` function, which keeps the high-level representation of the\ngenerated image close to that of the base image\n- The `total_variation_loss` function, a regularization loss which keeps the generated\nimage locally-coherent\n'

def gram_matrix(x):
    if False:
        for i in range(10):
            print('nop')
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style, combination):
    if False:
        print('Hello World!')
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * channels ** 2 * size ** 2)

def content_loss(base, combination):
    if False:
        return 10
    return tf.reduce_sum(tf.square(combination - base))

def total_variation_loss(x):
    if False:
        while True:
            i = 10
    a = tf.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = tf.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))
"\nNext, let's create a feature extraction model that retrieves the intermediate activations\nof VGG19 (as a dict, by name).\n"
model = vgg19.VGG19(weights='imagenet', include_top=False)
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)
"\nFinally, here's the code that computes the style transfer loss.\n"
style_layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layer_name = 'block5_conv2'

def compute_loss(combination_image, base_image, style_reference_image):
    if False:
        print('Hello World!')
    input_tensor = tf.concat([base_image, style_reference_image, combination_image], axis=0)
    features = feature_extractor(input_tensor)
    loss = tf.zeros(shape=())
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(base_image_features, combination_features)
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += style_weight / len(style_layer_names) * sl
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss
'\n## Add a tf.function decorator to loss & gradient computation\n\nTo compile it, and thus make it fast.\n'

@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    if False:
        print('Hello World!')
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return (loss, grads)
'\n## The training loop\n\nRepeatedly run vanilla gradient descent steps to minimize the loss, and save the\nresulting image every 100 iterations.\n\nWe decay the learning rate by 0.96 every 100 steps.\n'
optimizer = keras.optimizers.SGD(keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96))
base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_reference_image_path)
combination_image = tf.Variable(preprocess_image(base_image_path))
iterations = 4000
for i in range(1, iterations + 1):
    (loss, grads) = compute_loss_and_grads(combination_image, base_image, style_reference_image)
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 100 == 0:
        print('Iteration %d: loss=%.2f' % (i, loss))
        img = deprocess_image(combination_image.numpy())
        fname = result_prefix + '_at_iteration_%d.png' % i
        keras.utils.save_img(fname, img)
'\nAfter 4000 iterations, you get the following result:\n'
display(Image(result_prefix + '_at_iteration_4000.png'))
'\n**Example available on HuggingFace**\nTrained Model | Demo \n--- | --- \n[![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Neural%20style%20transfer-black.svg)](https://huggingface.co/keras-io/VGG19) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Neural%20style%20transfer-black.svg)](https://huggingface.co/spaces/keras-io/neural-style-transfer)\n'