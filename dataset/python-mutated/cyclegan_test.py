"""Tests for tensorflow.contrib.slim.nets.cyclegan."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from nets import cyclegan

class CycleganTest(tf.test.TestCase):

    def test_generator_inference(self):
        if False:
            i = 10
            return i + 15
        'Check one inference step.'
        img_batch = tf.zeros([2, 32, 32, 3])
        (model_output, _) = cyclegan.cyclegan_generator_resnet(img_batch)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(model_output)

    def _test_generator_graph_helper(self, shape):
        if False:
            for i in range(10):
                print('nop')
        'Check that generator can take small and non-square inputs.'
        (output_imgs, _) = cyclegan.cyclegan_generator_resnet(tf.ones(shape))
        self.assertAllEqual(shape, output_imgs.shape.as_list())

    def test_generator_graph_small(self):
        if False:
            return 10
        self._test_generator_graph_helper([4, 32, 32, 3])

    def test_generator_graph_medium(self):
        if False:
            print('Hello World!')
        self._test_generator_graph_helper([3, 128, 128, 3])

    def test_generator_graph_nonsquare(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_generator_graph_helper([2, 80, 400, 3])

    def test_generator_unknown_batch_dim(self):
        if False:
            while True:
                i = 10
        'Check that generator can take unknown batch dimension inputs.'
        img = tf.placeholder(tf.float32, shape=[None, 32, None, 3])
        (output_imgs, _) = cyclegan.cyclegan_generator_resnet(img)
        self.assertAllEqual([None, 32, None, 3], output_imgs.shape.as_list())

    def _input_and_output_same_shape_helper(self, kernel_size):
        if False:
            for i in range(10):
                print('nop')
        img_batch = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        (output_img_batch, _) = cyclegan.cyclegan_generator_resnet(img_batch, kernel_size=kernel_size)
        self.assertAllEqual(img_batch.shape.as_list(), output_img_batch.shape.as_list())

    def input_and_output_same_shape_kernel3(self):
        if False:
            return 10
        self._input_and_output_same_shape_helper(3)

    def input_and_output_same_shape_kernel4(self):
        if False:
            print('Hello World!')
        self._input_and_output_same_shape_helper(4)

    def input_and_output_same_shape_kernel5(self):
        if False:
            print('Hello World!')
        self._input_and_output_same_shape_helper(5)

    def input_and_output_same_shape_kernel6(self):
        if False:
            return 10
        self._input_and_output_same_shape_helper(6)

    def _error_if_height_not_multiple_of_four_helper(self, height):
        if False:
            i = 10
            return i + 15
        self.assertRaisesRegexp(ValueError, 'The input height must be a multiple of 4.', cyclegan.cyclegan_generator_resnet, tf.placeholder(tf.float32, shape=[None, height, 32, 3]))

    def test_error_if_height_not_multiple_of_four_height29(self):
        if False:
            for i in range(10):
                print('nop')
        self._error_if_height_not_multiple_of_four_helper(29)

    def test_error_if_height_not_multiple_of_four_height30(self):
        if False:
            while True:
                i = 10
        self._error_if_height_not_multiple_of_four_helper(30)

    def test_error_if_height_not_multiple_of_four_height31(self):
        if False:
            while True:
                i = 10
        self._error_if_height_not_multiple_of_four_helper(31)

    def _error_if_width_not_multiple_of_four_helper(self, width):
        if False:
            i = 10
            return i + 15
        self.assertRaisesRegexp(ValueError, 'The input width must be a multiple of 4.', cyclegan.cyclegan_generator_resnet, tf.placeholder(tf.float32, shape=[None, 32, width, 3]))

    def test_error_if_width_not_multiple_of_four_width29(self):
        if False:
            return 10
        self._error_if_width_not_multiple_of_four_helper(29)

    def test_error_if_width_not_multiple_of_four_width30(self):
        if False:
            return 10
        self._error_if_width_not_multiple_of_four_helper(30)

    def test_error_if_width_not_multiple_of_four_width31(self):
        if False:
            print('Hello World!')
        self._error_if_width_not_multiple_of_four_helper(31)
if __name__ == '__main__':
    tf.test.main()