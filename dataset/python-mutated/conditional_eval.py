"""Evaluates a conditional TFGAN trained MNIST model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
import tensorflow as tf
import data_provider
import networks
import util
tfgan = tf.contrib.gan
flags.DEFINE_string('checkpoint_dir', '/tmp/mnist/', 'Directory where the model was written to.')
flags.DEFINE_string('eval_dir', '/tmp/mnist/', 'Directory where the results are saved to.')
flags.DEFINE_integer('num_images_per_class', 10, 'Number of images to generate per class.')
flags.DEFINE_integer('noise_dims', 64, 'Dimensions of the generator noise vector')
flags.DEFINE_string('classifier_filename', None, 'Location of the pretrained classifier. If `None`, use default.')
flags.DEFINE_integer('max_number_of_evaluations', None, 'Number of times to run evaluation. If `None`, run forever.')
flags.DEFINE_boolean('write_to_disk', True, 'If `True`, run images to disk.')
FLAGS = flags.FLAGS
NUM_CLASSES = 10

def main(_, run_eval_loop=True):
    if False:
        for i in range(10):
            print('nop')
    with tf.name_scope('inputs'):
        (noise, one_hot_labels) = _get_generator_inputs(FLAGS.num_images_per_class, NUM_CLASSES, FLAGS.noise_dims)
    with tf.variable_scope('Generator'):
        images = networks.conditional_generator((noise, one_hot_labels), is_training=False)
    reshaped_img = tfgan.eval.image_reshaper(images, num_cols=FLAGS.num_images_per_class)
    tf.summary.image('generated_images', reshaped_img, max_outputs=1)
    tf.summary.scalar('MNIST_Classifier_score', util.mnist_score(images, FLAGS.classifier_filename))
    tf.summary.scalar('MNIST_Cross_entropy', util.mnist_cross_entropy(images, one_hot_labels, FLAGS.classifier_filename))
    image_write_ops = None
    if FLAGS.write_to_disk:
        image_write_ops = tf.write_file('%s/%s' % (FLAGS.eval_dir, 'conditional_gan.png'), tf.image.encode_png(data_provider.float_image_to_uint8(reshaped_img[0])))
    if not run_eval_loop:
        return
    tf.contrib.training.evaluate_repeatedly(FLAGS.checkpoint_dir, hooks=[tf.contrib.training.SummaryAtEndHook(FLAGS.eval_dir), tf.contrib.training.StopAfterNEvalsHook(1)], eval_ops=image_write_ops, max_number_of_evaluations=FLAGS.max_number_of_evaluations)

def _get_generator_inputs(num_images_per_class, num_classes, noise_dims):
    if False:
        while True:
            i = 10
    num_images_generated = num_images_per_class * num_classes
    noise = tf.random_normal([num_images_generated, noise_dims])
    labels = [lbl for lbl in range(num_classes) for _ in range(num_images_per_class)]
    one_hot_labels = tf.one_hot(tf.constant(labels), num_classes)
    return (noise, one_hot_labels)
if __name__ == '__main__':
    app.run(main)