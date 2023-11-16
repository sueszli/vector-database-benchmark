"""Tests for lstm_object_detection.tf_sequence_example_decoder."""
import numpy as np
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import parsing_ops
from lstm_object_detection.inputs import tf_sequence_example_decoder
from object_detection.core import standard_fields as fields

class TFSequenceExampleDecoderTest(tf.test.TestCase):
    """Tests for sequence example decoder."""

    def _EncodeImage(self, image_tensor, encoding_type='jpeg'):
        if False:
            for i in range(10):
                print('nop')
        with self.test_session():
            if encoding_type == 'jpeg':
                image_encoded = tf.image.encode_jpeg(tf.constant(image_tensor)).eval()
            else:
                raise ValueError('Invalid encoding type.')
        return image_encoded

    def _DecodeImage(self, image_encoded, encoding_type='jpeg'):
        if False:
            print('Hello World!')
        with self.test_session():
            if encoding_type == 'jpeg':
                image_decoded = tf.image.decode_jpeg(tf.constant(image_encoded)).eval()
            else:
                raise ValueError('Invalid encoding type.')
        return image_decoded

    def testDecodeJpegImageAndBoundingBox(self):
        if False:
            for i in range(10):
                print('nop')
        'Test if the decoder can correctly decode the image and bounding box.\n\n    A set of random images (represented as an image tensor) is first decoded as\n    the groundtrue image. Meanwhile, the image tensor will be encoded and pass\n    through the sequence example, and then decoded as images. The groundtruth\n    image and the decoded image are expected to be equal. Similar tests are\n    also applied to labels such as bounding box.\n    '
        image_tensor = np.random.randint(256, size=(256, 256, 3)).astype(np.uint8)
        encoded_jpeg = self._EncodeImage(image_tensor)
        decoded_jpeg = self._DecodeImage(encoded_jpeg)
        sequence_example = example_pb2.SequenceExample(feature_lists=feature_pb2.FeatureLists(feature_list={'image/encoded': feature_pb2.FeatureList(feature=[feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[encoded_jpeg]))]), 'bbox/xmin': feature_pb2.FeatureList(feature=[feature_pb2.Feature(float_list=feature_pb2.FloatList(value=[0.0]))]), 'bbox/xmax': feature_pb2.FeatureList(feature=[feature_pb2.Feature(float_list=feature_pb2.FloatList(value=[1.0]))]), 'bbox/ymin': feature_pb2.FeatureList(feature=[feature_pb2.Feature(float_list=feature_pb2.FloatList(value=[0.0]))]), 'bbox/ymax': feature_pb2.FeatureList(feature=[feature_pb2.Feature(float_list=feature_pb2.FloatList(value=[1.0]))])})).SerializeToString()
        example_decoder = tf_sequence_example_decoder.TFSequenceExampleDecoder()
        tensor_dict = example_decoder.decode(tf.convert_to_tensor(sequence_example))
        self.assertAllEqual(tensor_dict[fields.InputDataFields.image].get_shape().as_list(), [None, None, None, 3])
        with self.test_session() as sess:
            tensor_dict[fields.InputDataFields.image] = tf.squeeze(tensor_dict[fields.InputDataFields.image])
            tensor_dict[fields.InputDataFields.groundtruth_boxes] = tf.squeeze(tensor_dict[fields.InputDataFields.groundtruth_boxes])
            tensor_dict = sess.run(tensor_dict)
        self.assertAllEqual(decoded_jpeg, tensor_dict[fields.InputDataFields.image])
        self.assertAllEqual([0.0, 0.0, 1.0, 1.0], tensor_dict[fields.InputDataFields.groundtruth_boxes])
if __name__ == '__main__':
    tf.test.main()