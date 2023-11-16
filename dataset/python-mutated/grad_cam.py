"""
Title: Grad-CAM class activation visualization
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/26
Last modified: 2021/03/07
Description: How to obtain a class activation heatmap for an image classification model.
Accelerator: NONE
"""
'\nAdapted from Deep Learning with Python (2017).\n## Setup\n'
import numpy as np
import tensorflow as tf
import keras
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
keras.config.disable_traceback_filtering()
'\n## Configurable parameters\n\nYou can change these to another model.\n\nTo get the values for `last_conv_layer_name` use `model.summary()`\nto see the names of all layers in the model.\n\n'
model_builder = keras.applications.xception.Xception
img_size = (299, 299)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions
last_conv_layer_name = 'block14_sepconv2_act'
img_path = keras.utils.get_file(fname='african_elephant.jpg', origin='https://i.imgur.com/Bvro0YD.png')
display(Image(img_path))
'\n## The Grad-CAM algorithm\n'

def get_img_array(img_path, size):
    if False:
        for i in range(10):
            print('nop')
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    if False:
        print('Hello World!')
    grad_model = keras.models.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        (last_conv_layer_output, preds) = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
"\n## Let's test-drive it\n\n"
img_array = preprocess_input(get_img_array(img_path, size=img_size))
model = model_builder(weights='imagenet')
model.layers[-1].activation = None
preds = model.predict(img_array)
print('Predicted:', decode_predictions(preds, top=1)[0])
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
plt.matshow(heatmap)
plt.show()
'\n## Create a superimposed visualization\n\n'

def save_and_display_gradcam(img_path, heatmap, cam_path='cam.jpg', alpha=0.4):
    if False:
        while True:
            i = 10
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    display(Image(cam_path))
save_and_display_gradcam(img_path, heatmap)
"\n## Let's try another image\n\nWe will see how the grad cam explains the model's outputs for a multi-label image. Let's\ntry an image with a cat and a dog together, and see how the grad cam behaves.\n"
img_path = keras.utils.get_file(fname='cat_and_dog.jpg', origin='https://storage.googleapis.com/petbacker/images/blog/2017/dog-and-cat-cover.jpg')
display(Image(img_path))
img_array = preprocess_input(get_img_array(img_path, size=img_size))
preds = model.predict(img_array)
print('Predicted:', decode_predictions(preds, top=2)[0])
'\nWe generate class activation heatmap for "chow," the class index is 260\n\n'
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=260)
save_and_display_gradcam(img_path, heatmap)
'\nWe generate class activation heatmap for "egyptian cat," the class index is 285\n\n'
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=285)
save_and_display_gradcam(img_path, heatmap)