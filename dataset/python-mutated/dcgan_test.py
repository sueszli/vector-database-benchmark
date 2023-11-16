"""Tests for dcgan."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import tensorflow as tf
from nets import dcgan

class DCGANTest(tf.test.TestCase):

    def test_generator_run(self):
        if False:
            for i in range(10):
                print('nop')
        tf.set_random_seed(1234)
        noise = tf.random_normal([100, 64])
        (image, _) = dcgan.generator(noise)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            image.eval()

    def test_generator_graph(self):
        if False:
            for i in range(10):
                print('nop')
        tf.set_random_seed(1234)
        for (i, batch_size) in zip(xrange(3, 7), xrange(3, 8)):
            tf.reset_default_graph()
            final_size = 2 ** i
            noise = tf.random_normal([batch_size, 64])
            (image, end_points) = dcgan.generator(noise, depth=32, final_size=final_size)
            self.assertAllEqual([batch_size, final_size, final_size, 3], image.shape.as_list())
            expected_names = ['deconv%i' % j for j in xrange(1, i)] + ['logits']
            self.assertSetEqual(set(expected_names), set(end_points.keys()))
            for j in range(1, i):
                layer = end_points['deconv%i' % j]
                self.assertEqual(32 * 2 ** (i - j - 1), layer.get_shape().as_list()[-1])

    def test_generator_invalid_input(self):
        if False:
            print('Hello World!')
        wrong_dim_input = tf.zeros([5, 32, 32])
        with self.assertRaises(ValueError):
            dcgan.generator(wrong_dim_input)
        correct_input = tf.zeros([3, 2])
        with self.assertRaisesRegexp(ValueError, 'must be a power of 2'):
            dcgan.generator(correct_input, final_size=30)
        with self.assertRaisesRegexp(ValueError, 'must be greater than 8'):
            dcgan.generator(correct_input, final_size=4)

    def test_discriminator_run(self):
        if False:
            for i in range(10):
                print('nop')
        image = tf.random_uniform([5, 32, 32, 3], -1, 1)
        (output, _) = dcgan.discriminator(image)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output.eval()

    def test_discriminator_graph(self):
        if False:
            print('Hello World!')
        for (i, batch_size) in zip(xrange(1, 6), xrange(3, 8)):
            tf.reset_default_graph()
            img_w = 2 ** i
            image = tf.random_uniform([batch_size, img_w, img_w, 3], -1, 1)
            (output, end_points) = dcgan.discriminator(image, depth=32)
            self.assertAllEqual([batch_size, 1], output.get_shape().as_list())
            expected_names = ['conv%i' % j for j in xrange(1, i + 1)] + ['logits']
            self.assertSetEqual(set(expected_names), set(end_points.keys()))
            for j in range(1, i + 1):
                layer = end_points['conv%i' % j]
                self.assertEqual(32 * 2 ** (j - 1), layer.get_shape().as_list()[-1])

    def test_discriminator_invalid_input(self):
        if False:
            return 10
        wrong_dim_img = tf.zeros([5, 32, 32])
        with self.assertRaises(ValueError):
            dcgan.discriminator(wrong_dim_img)
        spatially_undefined_shape = tf.placeholder(tf.float32, [5, 32, None, 3])
        with self.assertRaises(ValueError):
            dcgan.discriminator(spatially_undefined_shape)
        not_square = tf.zeros([5, 32, 16, 3])
        with self.assertRaisesRegexp(ValueError, 'not have equal width and height'):
            dcgan.discriminator(not_square)
        not_power_2 = tf.zeros([5, 30, 30, 3])
        with self.assertRaisesRegexp(ValueError, 'not a power of 2'):
            dcgan.discriminator(not_power_2)
if __name__ == '__main__':
    tf.test.main()