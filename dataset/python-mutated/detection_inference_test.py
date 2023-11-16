"""Tests for detection_inference.py."""
import os
import StringIO
import numpy as np
from PIL import Image
import tensorflow as tf
from object_detection.core import standard_fields
from object_detection.inference import detection_inference
from object_detection.utils import dataset_util

def get_mock_tfrecord_path():
    if False:
        i = 10
        return i + 15
    return os.path.join(tf.test.get_temp_dir(), 'mock.tfrec')

def create_mock_tfrecord():
    if False:
        while True:
            i = 10
    pil_image = Image.fromarray(np.array([[[123, 0, 0]]], dtype=np.uint8), 'RGB')
    image_output_stream = StringIO.StringIO()
    pil_image.save(image_output_stream, format='png')
    encoded_image = image_output_stream.getvalue()
    feature_map = {'test_field': dataset_util.float_list_feature([1, 2, 3, 4]), standard_fields.TfExampleFields.image_encoded: dataset_util.bytes_feature(encoded_image)}
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_map))
    with tf.python_io.TFRecordWriter(get_mock_tfrecord_path()) as writer:
        writer.write(tf_example.SerializeToString())

def get_mock_graph_path():
    if False:
        i = 10
        return i + 15
    return os.path.join(tf.test.get_temp_dir(), 'mock_graph.pb')

def create_mock_graph():
    if False:
        print('Hello World!')
    g = tf.Graph()
    with g.as_default():
        in_image_tensor = tf.placeholder(tf.uint8, shape=[1, None, None, 3], name='image_tensor')
        tf.constant([2.0], name='num_detections')
        tf.constant([[[0, 0.8, 0.7, 1], [0.1, 0.2, 0.8, 0.9], [0.2, 0.3, 0.4, 0.5]]], name='detection_boxes')
        tf.constant([[0.1, 0.2, 0.3]], name='detection_scores')
        tf.identity(tf.constant([[1.0, 2.0, 3.0]]) * tf.reduce_sum(tf.cast(in_image_tensor, dtype=tf.float32)), name='detection_classes')
        graph_def = g.as_graph_def()
    with tf.gfile.Open(get_mock_graph_path(), 'w') as fl:
        fl.write(graph_def.SerializeToString())

class InferDetectionsTests(tf.test.TestCase):

    def test_simple(self):
        if False:
            return 10
        create_mock_graph()
        create_mock_tfrecord()
        (serialized_example_tensor, image_tensor) = detection_inference.build_input([get_mock_tfrecord_path()])
        self.assertAllEqual(image_tensor.get_shape().as_list(), [1, None, None, 3])
        (detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor) = detection_inference.build_inference_graph(image_tensor, get_mock_graph_path())
        with self.test_session(use_gpu=False) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            tf.train.start_queue_runners()
            tf_example = detection_inference.infer_detections_and_add_to_example(serialized_example_tensor, detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor, False)
        self.assertProtoEquals('\n        features {\n          feature {\n            key: "image/detection/bbox/ymin"\n            value { float_list { value: [0.0, 0.1] } } }\n          feature {\n            key: "image/detection/bbox/xmin"\n            value { float_list { value: [0.8, 0.2] } } }\n          feature {\n            key: "image/detection/bbox/ymax"\n            value { float_list { value: [0.7, 0.8] } } }\n          feature {\n            key: "image/detection/bbox/xmax"\n            value { float_list { value: [1.0, 0.9] } } }\n          feature {\n            key: "image/detection/label"\n            value { int64_list { value: [123, 246] } } }\n          feature {\n            key: "image/detection/score"\n            value { float_list { value: [0.1, 0.2] } } }\n          feature {\n            key: "image/encoded"\n            value { bytes_list { value:\n              "\\211PNG\\r\\n\\032\\n\\000\\000\\000\\rIHDR\\000\\000\\000\\001\\000\\000"\n              "\\000\\001\\010\\002\\000\\000\\000\\220wS\\336\\000\\000\\000\\022IDATx"\n              "\\234b\\250f`\\000\\000\\000\\000\\377\\377\\003\\000\\001u\\000|gO\\242"\n              "\\213\\000\\000\\000\\000IEND\\256B`\\202" } } }\n          feature {\n            key: "test_field"\n            value { float_list { value: [1.0, 2.0, 3.0, 4.0] } } } }\n    ', tf_example)

    def test_discard_image(self):
        if False:
            for i in range(10):
                print('nop')
        create_mock_graph()
        create_mock_tfrecord()
        (serialized_example_tensor, image_tensor) = detection_inference.build_input([get_mock_tfrecord_path()])
        (detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor) = detection_inference.build_inference_graph(image_tensor, get_mock_graph_path())
        with self.test_session(use_gpu=False) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            tf.train.start_queue_runners()
            tf_example = detection_inference.infer_detections_and_add_to_example(serialized_example_tensor, detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor, True)
        self.assertProtoEquals('\n        features {\n          feature {\n            key: "image/detection/bbox/ymin"\n            value { float_list { value: [0.0, 0.1] } } }\n          feature {\n            key: "image/detection/bbox/xmin"\n            value { float_list { value: [0.8, 0.2] } } }\n          feature {\n            key: "image/detection/bbox/ymax"\n            value { float_list { value: [0.7, 0.8] } } }\n          feature {\n            key: "image/detection/bbox/xmax"\n            value { float_list { value: [1.0, 0.9] } } }\n          feature {\n            key: "image/detection/label"\n            value { int64_list { value: [123, 246] } } }\n          feature {\n            key: "image/detection/score"\n            value { float_list { value: [0.1, 0.2] } } }\n          feature {\n            key: "test_field"\n            value { float_list { value: [1.0, 2.0, 3.0, 4.0] } } } }\n    ', tf_example)
if __name__ == '__main__':
    tf.test.main()