"""Tests for object_detection.core.box_coder."""
import tensorflow as tf
from object_detection.core import box_coder
from object_detection.core import box_list

class MockBoxCoder(box_coder.BoxCoder):
    """Test BoxCoder that encodes/decodes using the multiply-by-two function."""

    def code_size(self):
        if False:
            return 10
        return 4

    def _encode(self, boxes, anchors):
        if False:
            for i in range(10):
                print('nop')
        return 2.0 * boxes.get()

    def _decode(self, rel_codes, anchors):
        if False:
            print('Hello World!')
        return box_list.BoxList(rel_codes / 2.0)

class BoxCoderTest(tf.test.TestCase):

    def test_batch_decode(self):
        if False:
            for i in range(10):
                print('nop')
        mock_anchor_corners = tf.constant([[0, 0.1, 0.2, 0.3], [0.2, 0.4, 0.4, 0.6]], tf.float32)
        mock_anchors = box_list.BoxList(mock_anchor_corners)
        mock_box_coder = MockBoxCoder()
        expected_boxes = [[[0.0, 0.1, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8]], [[0.1, 0.2, 0.3, 0.4], [0.7, 0.8, 0.9, 1.0]]]
        encoded_boxes_list = [mock_box_coder.encode(box_list.BoxList(tf.constant(boxes)), mock_anchors) for boxes in expected_boxes]
        encoded_boxes = tf.stack(encoded_boxes_list)
        decoded_boxes = box_coder.batch_decode(encoded_boxes, mock_box_coder, mock_anchors)
        with self.test_session() as sess:
            decoded_boxes_result = sess.run(decoded_boxes)
            self.assertAllClose(expected_boxes, decoded_boxes_result)
if __name__ == '__main__':
    tf.test.main()