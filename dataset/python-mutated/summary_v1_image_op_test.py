"""Tests for summary V1 image op."""
import numpy as np
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import image_ops
import tensorflow.python.ops.nn_grad
from tensorflow.python.platform import test
from tensorflow.python.summary import summary

class SummaryV1ImageOpTest(test.TestCase):

    def _AsSummary(self, s):
        if False:
            while True:
                i = 10
        summ = summary_pb2.Summary()
        summ.ParseFromString(s)
        return summ

    def _CheckProto(self, image_summ, shape):
        if False:
            i = 10
            return i + 15
        'Verify that the non-image parts of the image_summ proto match shape.'
        for v in image_summ.value:
            v.image.ClearField('encoded_image_string')
        expected = '\n'.join(('\n        value {\n          tag: "img/image/%d"\n          image { height: %d width: %d colorspace: %d }\n        }' % ((i,) + shape[1:]) for i in range(3)))
        self.assertProtoEquals(expected, image_summ)

    @test_util.run_deprecated_v1
    def testImageSummary(self):
        if False:
            return 10
        for depth in (1, 3, 4):
            for positive in (False, True):
                with self.session(graph=ops.Graph()) as sess:
                    shape = (4, 5, 7) + (depth,)
                    bad_color = [255, 0, 0, 255][:depth]
                    const = np.random.randn(*shape).astype(np.float32)
                    const[0, 1, 2] = 0
                    if positive:
                        const = 1 + np.maximum(const, 0)
                        scale = 255 / const.reshape(4, -1).max(axis=1)
                        offset = 0
                    else:
                        scale = 127 / np.abs(const.reshape(4, -1)).max(axis=1)
                        offset = 128
                    adjusted = np.floor(scale[:, None, None, None] * const + offset)
                    const[0, 1, 2, depth // 2] = np.nan
                    summ = summary.image('img', const)
                    value = self.evaluate(summ)
                    self.assertEqual([], summ.get_shape())
                    image_summ = self._AsSummary(value)
                    image = image_ops.decode_png(image_summ.value[0].image.encoded_image_string).eval()
                    self.assertAllEqual(image[1, 2], bad_color)
                    image[1, 2] = adjusted[0, 1, 2]
                    self.assertAllClose(image, adjusted[0], rtol=2e-05, atol=2e-05)
                    self._CheckProto(image_summ, shape)

    @test_util.run_deprecated_v1
    def testImageSummaryUint8(self):
        if False:
            print('Hello World!')
        np.random.seed(7)
        for depth in (1, 3, 4):
            with self.session(graph=ops.Graph()) as sess:
                shape = (4, 5, 7) + (depth,)
                images = np.random.randint(256, size=shape).astype(np.uint8)
                tf_images = ops.convert_to_tensor(images)
                self.assertEqual(tf_images.dtype, dtypes.uint8)
                summ = summary.image('img', tf_images)
                value = self.evaluate(summ)
                self.assertEqual([], summ.get_shape())
                image_summ = self._AsSummary(value)
                image = image_ops.decode_png(image_summ.value[0].image.encoded_image_string).eval()
                self.assertAllEqual(image, images[0])
                self._CheckProto(image_summ, shape)
if __name__ == '__main__':
    test.main()