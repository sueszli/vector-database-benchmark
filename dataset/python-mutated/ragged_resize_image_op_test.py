"""Tests for RaggedTensor dispatch of tf.images.resize."""
from absl.testing import parameterized
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest

@test_util.run_all_in_graph_and_eager_modes
class RaggedResizeImageOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    def make_image_batch(self, sizes, channels):
        if False:
            return 10
        if not sizes:
            return ragged_tensor.RaggedTensor.from_tensor(array_ops.zeros([0, 5, 5, channels]), ragged_rank=2)
        images = [array_ops.reshape(math_ops.range(w * h * channels * 1.0), [w, h, channels]) for (w, h) in sizes]
        return ragged_concat_ops.stack(images)

    @parameterized.parameters([dict(src_sizes=[], dst_size=(4, 4), v1=True), dict(src_sizes=[], dst_size=(4, 4), v1=False), dict(src_sizes=[(2, 2)], dst_size=(4, 4), v1=True), dict(src_sizes=[(2, 2)], dst_size=(4, 4), v1=False), dict(src_sizes=[(2, 8), (3, 5), (10, 10)], dst_size=(5, 5), v1=True), dict(src_sizes=[(2, 8), (3, 5), (10, 10)], dst_size=(5, 5), v1=False)])
    def testResize(self, src_sizes, dst_size, v1=False):
        if False:
            i = 10
            return i + 15
        resize = image_ops.resize_images if v1 else image_ops.resize_images_v2
        channels = 3
        images = self.make_image_batch(src_sizes, channels)
        expected_shape = [len(src_sizes)] + list(dst_size) + [channels]
        resized_images = resize(images, dst_size)
        self.assertIsInstance(resized_images, tensor.Tensor)
        self.assertEqual(resized_images.shape.as_list(), expected_shape)
        for i in range(len(src_sizes)):
            actual = resized_images[i]
            expected = resize(images[i].to_tensor(), dst_size)
            self.assertAllClose(actual, expected)

    @parameterized.parameters([dict(src_shape=[None, None, None, None], src_sizes=[], dst_size=(4, 4)), dict(src_shape=[None, None, None, 3], src_sizes=[], dst_size=(4, 4)), dict(src_shape=[0, None, None, None], src_sizes=[], dst_size=(4, 4)), dict(src_shape=[0, None, None, 3], src_sizes=[], dst_size=(4, 4)), dict(src_shape=[None, None, None, None], src_sizes=[(2, 2)], dst_size=(4, 4)), dict(src_shape=[None, None, None, None], src_sizes=[(2, 8), (3, 5), (10, 10)], dst_size=(5, 5)), dict(src_shape=[None, None, None, 1], src_sizes=[(2, 8), (3, 5), (10, 10)], dst_size=(5, 5)), dict(src_shape=[3, None, None, 1], src_sizes=[(2, 8), (3, 5), (10, 10)], dst_size=(5, 5))])
    def testResizeWithPartialStaticShape(self, src_shape, src_sizes, dst_size):
        if False:
            i = 10
            return i + 15
        channels = src_shape[-1] or 3
        images = self.make_image_batch(src_sizes, channels)
        rt_spec = ragged_tensor.RaggedTensorSpec(src_shape, ragged_rank=images.ragged_rank)
        expected_shape = [len(src_sizes)] + list(dst_size) + [channels]

        @def_function.function(input_signature=[rt_spec])
        def do_resize(images):
            if False:
                print('Hello World!')
            return image_ops.resize_images_v2(images, dst_size)
        resized_images = do_resize(images)
        self.assertIsInstance(resized_images, tensor.Tensor)
        self.assertTrue(resized_images.shape.is_compatible_with(expected_shape))
        for i in range(len(src_sizes)):
            actual = resized_images[i]
            expected = image_ops.resize_images_v2(images[i].to_tensor(), dst_size)
            self.assertAllClose(actual, expected)

    def testSizeIsTensor(self):
        if False:
            print('Hello World!')

        @def_function.function
        def do_resize(images, new_size):
            if False:
                while True:
                    i = 10
            return image_ops.resize_images_v2(images, new_size)
        src_images = self.make_image_batch([[5, 8], [3, 2], [10, 4]], 3)
        resized_images = do_resize(src_images, constant_op.constant([2, 2]))
        self.assertIsInstance(resized_images, tensor.Tensor)
        self.assertTrue(resized_images.shape.is_compatible_with([3, 2, 2, 3]))

    def testBadRank(self):
        if False:
            print('Hello World!')
        rt = ragged_tensor.RaggedTensor.from_tensor(array_ops.zeros([5, 5, 3]))
        with self.assertRaisesRegex(ValueError, 'rank must be 4'):
            image_ops.resize_images_v2(rt, [10, 10])
if __name__ == '__main__':
    googletest.main()