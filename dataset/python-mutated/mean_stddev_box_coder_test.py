"""Tests for object_detection.box_coder.mean_stddev_boxcoder."""
import tensorflow as tf
from object_detection.box_coders import mean_stddev_box_coder
from object_detection.core import box_list

class MeanStddevBoxCoderTest(tf.test.TestCase):

    def testGetCorrectRelativeCodesAfterEncoding(self):
        if False:
            return 10
        box_corners = [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]]
        boxes = box_list.BoxList(tf.constant(box_corners))
        expected_rel_codes = [[0.0, 0.0, 0.0, 0.0], [-5.0, -5.0, -5.0, -3.0]]
        prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8]])
        priors = box_list.BoxList(prior_means)
        coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)
        rel_codes = coder.encode(boxes, priors)
        with self.test_session() as sess:
            rel_codes_out = sess.run(rel_codes)
            self.assertAllClose(rel_codes_out, expected_rel_codes)

    def testGetCorrectBoxesAfterDecoding(self):
        if False:
            while True:
                i = 10
        rel_codes = tf.constant([[0.0, 0.0, 0.0, 0.0], [-5.0, -5.0, -5.0, -3.0]])
        expected_box_corners = [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]]
        prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8]])
        priors = box_list.BoxList(prior_means)
        coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)
        decoded_boxes = coder.decode(rel_codes, priors)
        decoded_box_corners = decoded_boxes.get()
        with self.test_session() as sess:
            decoded_out = sess.run(decoded_box_corners)
            self.assertAllClose(decoded_out, expected_box_corners)
if __name__ == '__main__':
    tf.test.main()