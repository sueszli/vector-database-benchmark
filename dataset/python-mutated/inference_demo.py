"""Demo that makes inference requests against a running inference server."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
from absl import flags
import numpy as np
import PIL
import tensorflow as tf
import data_provider
import networks
tfgan = tf.contrib.gan
flags.DEFINE_string('checkpoint_path', '', 'CycleGAN checkpoint path created by train.py. (e.g. "/mylogdir/model.ckpt-18442")')
flags.DEFINE_string('image_set_x_glob', '', 'Optional: Glob path to images of class X to feed through the CycleGAN.')
flags.DEFINE_string('image_set_y_glob', '', 'Optional: Glob path to images of class Y to feed through the CycleGAN.')
flags.DEFINE_string('generated_x_dir', '/tmp/generated_x/', 'If image_set_y_glob is defined, where to output the generated X images.')
flags.DEFINE_string('generated_y_dir', '/tmp/generated_y/', 'If image_set_x_glob is defined, where to output the generated Y images.')
flags.DEFINE_integer('patch_dim', 128, 'The patch size of images that was used in train.py.')
FLAGS = flags.FLAGS

def _make_dir_if_not_exists(dir_path):
    if False:
        print('Hello World!')
    'Make a directory if it does not exist.'
    if not tf.gfile.Exists(dir_path):
        tf.gfile.MakeDirs(dir_path)

def _file_output_path(dir_path, input_file_path):
    if False:
        return 10
    'Create output path for an individual file.'
    return os.path.join(dir_path, os.path.basename(input_file_path))

def make_inference_graph(model_name, patch_dim):
    if False:
        while True:
            i = 10
    "Build the inference graph for either the X2Y or Y2X GAN.\n\n  Args:\n    model_name: The var scope name 'ModelX2Y' or 'ModelY2X'.\n    patch_dim: An integer size of patches to feed to the generator.\n\n  Returns:\n    Tuple of (input_placeholder, generated_tensor).\n  "
    input_hwc_pl = tf.placeholder(tf.float32, [None, None, 3])
    images_x = tf.expand_dims(data_provider.full_image_to_patch(input_hwc_pl, patch_dim), 0)
    with tf.variable_scope(model_name):
        with tf.variable_scope('Generator'):
            generated = networks.generator(images_x)
    return (input_hwc_pl, generated)

def export(sess, input_pl, output_tensor, input_file_pattern, output_dir):
    if False:
        i = 10
        return i + 15
    'Exports inference outputs to an output directory.\n\n  Args:\n    sess: tf.Session with variables already loaded.\n    input_pl: tf.Placeholder for input (HWC format).\n    output_tensor: Tensor for generated outut images.\n    input_file_pattern: Glob file pattern for input images.\n    output_dir: Output directory.\n  '
    if output_dir:
        _make_dir_if_not_exists(output_dir)
    if input_file_pattern:
        for file_path in tf.gfile.Glob(input_file_pattern):
            input_np = np.asarray(PIL.Image.open(file_path))
            output_np = sess.run(output_tensor, feed_dict={input_pl: input_np})
            image_np = data_provider.undo_normalize_image(output_np)
            output_path = _file_output_path(output_dir, file_path)
            PIL.Image.fromarray(image_np).save(output_path)

def _validate_flags():
    if False:
        i = 10
        return i + 15
    flags.register_validator('checkpoint_path', bool, 'Must provide `checkpoint_path`.')
    flags.register_validator('generated_x_dir', lambda x: False if FLAGS.image_set_y_glob and (not x) else True, 'Must provide `generated_x_dir`.')
    flags.register_validator('generated_y_dir', lambda x: False if FLAGS.image_set_x_glob and (not x) else True, 'Must provide `generated_y_dir`.')

def main(_):
    if False:
        while True:
            i = 10
    _validate_flags()
    (images_x_hwc_pl, generated_y) = make_inference_graph('ModelX2Y', FLAGS.patch_dim)
    (images_y_hwc_pl, generated_x) = make_inference_graph('ModelY2X', FLAGS.patch_dim)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.checkpoint_path)
        export(sess, images_x_hwc_pl, generated_y, FLAGS.image_set_x_glob, FLAGS.generated_y_dir)
        export(sess, images_y_hwc_pl, generated_x, FLAGS.image_set_y_glob, FLAGS.generated_x_dir)
if __name__ == '__main__':
    app.run()