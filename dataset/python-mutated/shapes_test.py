"""Tests for shapes."""
import numpy as np
import tensorflow as tf
import shapes

def _rand(*size):
    if False:
        return 10
    return np.random.uniform(size=size).astype('f')

class ShapesTest(tf.test.TestCase):
    """Tests just the shapes from a call to transposing_reshape."""

    def __init__(self, other):
        if False:
            while True:
                i = 10
        super(ShapesTest, self).__init__(other)
        self.batch_size = 4
        self.im_height = 24
        self.im_width = 36
        self.depth = 20

    def testReshapeTile(self):
        if False:
            print('Hello World!')
        'Tests that a tiled input can be reshaped to the batch dimension.'
        fake = tf.placeholder(tf.float32, shape=(None, None, None, self.depth), name='inputs')
        real = _rand(self.batch_size, self.im_height, self.im_width, self.depth)
        with self.test_session() as sess:
            outputs = shapes.transposing_reshape(fake, src_dim=2, part_a=3, part_b=-1, dest_dim_a=0, dest_dim_b=2)
            res_image = sess.run([outputs], feed_dict={fake: real})
            self.assertEqual(tuple(res_image[0].shape), (self.batch_size * 3, self.im_height, self.im_width / 3, self.depth))

    def testReshapeDepth(self):
        if False:
            i = 10
            return i + 15
        'Tests that depth can be reshaped to the x dimension.'
        fake = tf.placeholder(tf.float32, shape=(None, None, None, self.depth), name='inputs')
        real = _rand(self.batch_size, self.im_height, self.im_width, self.depth)
        with self.test_session() as sess:
            outputs = shapes.transposing_reshape(fake, src_dim=3, part_a=4, part_b=-1, dest_dim_a=2, dest_dim_b=3)
            res_image = sess.run([outputs], feed_dict={fake: real})
            self.assertEqual(tuple(res_image[0].shape), (self.batch_size, self.im_height, self.im_width * 4, self.depth / 4))

class DataTest(tf.test.TestCase):
    """Tests that the data is moved correctly in a call to transposing_reshape.

  """

    def testTransposingReshape_2_2_3_2_1(self):
        if False:
            i = 10
            return i + 15
        'Case: dest_a == src, dest_b < src: Split with Least sig part going left.\n    '
        with self.test_session() as sess:
            fake = tf.placeholder(tf.float32, shape=(None, None, None, 2), name='inputs')
            outputs = shapes.transposing_reshape(fake, src_dim=2, part_a=2, part_b=3, dest_dim_a=2, dest_dim_b=1)
            real = np.arange(120).reshape((5, 2, 6, 2))
            np_array = sess.run([outputs], feed_dict={fake: real})[0]
            self.assertEqual(tuple(np_array.shape), (5, 6, 2, 2))
            self.assertAllEqual(np_array[0, :, :, :], [[[0, 1], [6, 7]], [[12, 13], [18, 19]], [[2, 3], [8, 9]], [[14, 15], [20, 21]], [[4, 5], [10, 11]], [[16, 17], [22, 23]]])

    def testTransposingReshape_2_2_3_2_3(self):
        if False:
            i = 10
            return i + 15
        'Case: dest_a == src, dest_b > src: Split with Least sig part going right.\n    '
        with self.test_session() as sess:
            fake = tf.placeholder(tf.float32, shape=(None, None, None, 2), name='inputs')
            outputs = shapes.transposing_reshape(fake, src_dim=2, part_a=2, part_b=3, dest_dim_a=2, dest_dim_b=3)
            real = np.arange(120).reshape((5, 2, 6, 2))
            np_array = sess.run([outputs], feed_dict={fake: real})[0]
            self.assertEqual(tuple(np_array.shape), (5, 2, 2, 6))
            self.assertAllEqual(np_array[0, :, :, :], [[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], [[12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]]])

    def testTransposingReshape_2_2_3_2_2(self):
        if False:
            return 10
        'Case: dest_a == src, dest_b == src. Transpose within dimension 2.\n    '
        with self.test_session() as sess:
            fake = tf.placeholder(tf.float32, shape=(None, None, None, 2), name='inputs')
            outputs = shapes.transposing_reshape(fake, src_dim=2, part_a=2, part_b=3, dest_dim_a=2, dest_dim_b=2)
            real = np.arange(120).reshape((5, 2, 6, 2))
            np_array = sess.run([outputs], feed_dict={fake: real})[0]
            self.assertEqual(tuple(np_array.shape), (5, 2, 6, 2))
            self.assertAllEqual(np_array[0, :, :, :], [[[0, 1], [6, 7], [2, 3], [8, 9], [4, 5], [10, 11]], [[12, 13], [18, 19], [14, 15], [20, 21], [16, 17], [22, 23]]])

    def testTransposingReshape_2_2_3_1_2(self):
        if False:
            print('Hello World!')
        'Case: dest_a < src, dest_b == src. Split with Most sig part going left.\n    '
        with self.test_session() as sess:
            fake = tf.placeholder(tf.float32, shape=(None, None, None, 2), name='inputs')
            outputs = shapes.transposing_reshape(fake, src_dim=2, part_a=2, part_b=3, dest_dim_a=1, dest_dim_b=2)
            real = np.arange(120).reshape((5, 2, 6, 2))
            np_array = sess.run([outputs], feed_dict={fake: real})[0]
            self.assertEqual(tuple(np_array.shape), (5, 4, 3, 2))
            self.assertAllEqual(np_array[0, :, :, :], [[[0, 1], [2, 3], [4, 5]], [[12, 13], [14, 15], [16, 17]], [[6, 7], [8, 9], [10, 11]], [[18, 19], [20, 21], [22, 23]]])

    def testTransposingReshape_2_2_3_3_2(self):
        if False:
            i = 10
            return i + 15
        'Case: dest_a < src, dest_b == src. Split with Most sig part going right.\n    '
        with self.test_session() as sess:
            fake = tf.placeholder(tf.float32, shape=(None, None, None, 2), name='inputs')
            outputs = shapes.transposing_reshape(fake, src_dim=2, part_a=2, part_b=3, dest_dim_a=3, dest_dim_b=2)
            real = np.arange(120).reshape((5, 2, 6, 2))
            np_array = sess.run([outputs], feed_dict={fake: real})[0]
            self.assertEqual(tuple(np_array.shape), (5, 2, 3, 4))
            self.assertAllEqual(np_array[0, :, :, :], [[[0, 1, 6, 7], [2, 3, 8, 9], [4, 5, 10, 11]], [[12, 13, 18, 19], [14, 15, 20, 21], [16, 17, 22, 23]]])
if __name__ == '__main__':
    tf.test.main()