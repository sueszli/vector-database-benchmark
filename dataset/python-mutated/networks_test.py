"""Tests for gan.image_compression.networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import tensorflow as tf
import networks

class NetworksTest(tf.test.TestCase):

    def test_last_conv_layer(self):
        if False:
            print('Hello World!')
        x = tf.constant(1.0)
        y = tf.constant(0.0)
        end_points = {'silly': y, 'conv2': y, 'conv4': x, 'logits': y, 'conv-1': y}
        self.assertEqual(x, networks._last_conv_layer(end_points))

    def test_generator_run(self):
        if False:
            return 10
        img_batch = tf.zeros([3, 16, 16, 3])
        model_output = networks.compression_model(img_batch)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(model_output)

    def test_generator_graph(self):
        if False:
            print('Hello World!')
        for (i, batch_size) in zip(xrange(3, 7), xrange(3, 11, 2)):
            tf.reset_default_graph()
            patch_size = 2 ** i
            bits = 2 ** i
            img = tf.ones([batch_size, patch_size, patch_size, 3])
            (uncompressed, binary_codes, prebinary) = networks.compression_model(img, bits)
            self.assertAllEqual([batch_size, patch_size, patch_size, 3], uncompressed.shape.as_list())
            self.assertEqual([batch_size, bits], binary_codes.shape.as_list())
            self.assertEqual([batch_size, bits], prebinary.shape.as_list())

    def test_generator_invalid_input(self):
        if False:
            for i in range(10):
                print('nop')
        wrong_dim_input = tf.zeros([5, 32, 32])
        with self.assertRaisesRegexp(ValueError, 'Shape .* must have rank 4'):
            networks.compression_model(wrong_dim_input)
        not_fully_defined = tf.placeholder(tf.float32, [3, None, 32, 3])
        with self.assertRaisesRegexp(ValueError, 'Shape .* is not fully defined'):
            networks.compression_model(not_fully_defined)

    def test_discriminator_run(self):
        if False:
            i = 10
            return i + 15
        img_batch = tf.zeros([3, 70, 70, 3])
        disc_output = networks.discriminator(img_batch)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(disc_output)

    def test_discriminator_graph(self):
        if False:
            for i in range(10):
                print('nop')
        for (batch_size, patch_size) in zip([3, 6], [70, 128]):
            tf.reset_default_graph()
            img = tf.ones([batch_size, patch_size, patch_size, 3])
            disc_output = networks.discriminator(img)
            self.assertEqual(2, disc_output.shape.ndims)
            self.assertEqual(batch_size, disc_output.shape[0])

    def test_discriminator_invalid_input(self):
        if False:
            print('Hello World!')
        wrong_dim_input = tf.zeros([5, 32, 32])
        with self.assertRaisesRegexp(ValueError, 'Shape must be rank 4'):
            networks.discriminator(wrong_dim_input)
        not_fully_defined = tf.placeholder(tf.float32, [3, None, 32, 3])
        with self.assertRaisesRegexp(ValueError, 'Shape .* is not fully defined'):
            networks.compression_model(not_fully_defined)
if __name__ == '__main__':
    tf.test.main()