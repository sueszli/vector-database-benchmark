"""Summaries utility file to share between train and eval."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
tfgan = tf.contrib.gan

def add_reconstruction_summaries(images, reconstructions, prebinary, num_imgs_to_visualize=8):
    if False:
        i = 10
        return i + 15
    'Adds image summaries.'
    reshaped_img = stack_images(images, reconstructions, num_imgs_to_visualize)
    tf.summary.image('real_vs_reconstruction', reshaped_img, max_outputs=1)
    if prebinary is not None:
        tf.summary.histogram('prebinary_codes', prebinary)

def stack_images(images, reconstructions, num_imgs_to_visualize=8):
    if False:
        i = 10
        return i + 15
    'Stack and reshape images to see compression effects.'
    to_reshape = tf.unstack(images)[:num_imgs_to_visualize] + tf.unstack(reconstructions)[:num_imgs_to_visualize]
    reshaped_img = tfgan.eval.image_reshaper(to_reshape, num_cols=num_imgs_to_visualize)
    return reshaped_img