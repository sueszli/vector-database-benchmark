"""Tests for oid_tfrecord_creation.py."""
import pandas as pd
import tensorflow as tf
from object_detection.dataset_tools import oid_tfrecord_creation

def create_test_data():
    if False:
        print('Hello World!')
    data = {'ImageID': ['i1', 'i1', 'i1', 'i1', 'i1', 'i2', 'i2'], 'LabelName': ['a', 'a', 'b', 'b', 'c', 'b', 'c'], 'YMin': [0.3, 0.6, 0.8, 0.1, None, 0.0, 0.0], 'XMin': [0.1, 0.3, 0.7, 0.0, None, 0.1, 0.1], 'XMax': [0.2, 0.3, 0.8, 0.5, None, 0.9, 0.9], 'YMax': [0.3, 0.6, 1, 0.8, None, 0.8, 0.8], 'IsOccluded': [0, 1, 1, 0, None, 0, 0], 'IsTruncated': [0, 0, 0, 1, None, 0, 0], 'IsGroupOf': [0, 0, 0, 0, None, 0, 1], 'IsDepiction': [1, 0, 0, 0, None, 0, 0], 'ConfidenceImageLabel': [None, None, None, None, 0, None, None]}
    df = pd.DataFrame(data=data)
    label_map = {'a': 0, 'b': 1, 'c': 2}
    return (label_map, df)

class TfExampleFromAnnotationsDataFrameTests(tf.test.TestCase):

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        (label_map, df) = create_test_data()
        tf_example = oid_tfrecord_creation.tf_example_from_annotations_data_frame(df[df.ImageID == 'i1'], label_map, 'encoded_image_test')
        self.assertProtoEquals('\n        features {\n          feature {\n            key: "image/encoded"\n            value { bytes_list { value: "encoded_image_test" } } }\n          feature {\n            key: "image/filename"\n            value { bytes_list { value: "i1.jpg" } } }\n          feature {\n            key: "image/object/bbox/ymin"\n            value { float_list { value: [0.3, 0.6, 0.8, 0.1] } } }\n          feature {\n            key: "image/object/bbox/xmin"\n            value { float_list { value: [0.1, 0.3, 0.7, 0.0] } } }\n          feature {\n            key: "image/object/bbox/ymax"\n            value { float_list { value: [0.3, 0.6, 1.0, 0.8] } } }\n          feature {\n            key: "image/object/bbox/xmax"\n            value { float_list { value: [0.2, 0.3, 0.8, 0.5] } } }\n          feature {\n            key: "image/object/class/label"\n            value { int64_list { value: [0, 0, 1, 1] } } }\n          feature {\n            key: "image/object/class/text"\n            value { bytes_list { value: ["a", "a", "b", "b"] } } }\n          feature {\n            key: "image/source_id"\n            value { bytes_list { value: "i1" } } }\n          feature {\n            key: "image/object/depiction"\n            value { int64_list { value: [1, 0, 0, 0] } } }\n          feature {\n            key: "image/object/group_of"\n            value { int64_list { value: [0, 0, 0, 0] } } }\n          feature {\n            key: "image/object/occluded"\n            value { int64_list { value: [0, 1, 1, 0] } } }\n          feature {\n            key: "image/object/truncated"\n            value { int64_list { value: [0, 0, 0, 1] } } }\n          feature {\n            key: "image/class/label"\n            value { int64_list { value: [2] } } }\n          feature {\n            key: "image/class/text"\n            value { bytes_list { value: ["c"] } } } }\n    ', tf_example)

    def test_no_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        (label_map, df) = create_test_data()
        del df['IsDepiction']
        del df['IsGroupOf']
        del df['IsOccluded']
        del df['IsTruncated']
        del df['ConfidenceImageLabel']
        tf_example = oid_tfrecord_creation.tf_example_from_annotations_data_frame(df[df.ImageID == 'i2'], label_map, 'encoded_image_test')
        self.assertProtoEquals('\n        features {\n          feature {\n            key: "image/encoded"\n            value { bytes_list { value: "encoded_image_test" } } }\n          feature {\n            key: "image/filename"\n            value { bytes_list { value: "i2.jpg" } } }\n          feature {\n            key: "image/object/bbox/ymin"\n            value { float_list { value: [0.0, 0.0] } } }\n          feature {\n            key: "image/object/bbox/xmin"\n            value { float_list { value: [0.1, 0.1] } } }\n          feature {\n            key: "image/object/bbox/ymax"\n            value { float_list { value: [0.8, 0.8] } } }\n          feature {\n            key: "image/object/bbox/xmax"\n            value { float_list { value: [0.9, 0.9] } } }\n          feature {\n            key: "image/object/class/label"\n            value { int64_list { value: [1, 2] } } }\n          feature {\n            key: "image/object/class/text"\n            value { bytes_list { value: ["b", "c"] } } }\n          feature {\n            key: "image/source_id"\n           value { bytes_list { value: "i2" } } } }\n    ', tf_example)

    def test_label_filtering(self):
        if False:
            for i in range(10):
                print('nop')
        (label_map, df) = create_test_data()
        label_map = {'a': 0}
        tf_example = oid_tfrecord_creation.tf_example_from_annotations_data_frame(df[df.ImageID == 'i1'], label_map, 'encoded_image_test')
        self.assertProtoEquals('\n        features {\n          feature {\n            key: "image/encoded"\n            value { bytes_list { value: "encoded_image_test" } } }\n          feature {\n            key: "image/filename"\n            value { bytes_list { value: "i1.jpg" } } }\n          feature {\n            key: "image/object/bbox/ymin"\n            value { float_list { value: [0.3, 0.6] } } }\n          feature {\n            key: "image/object/bbox/xmin"\n            value { float_list { value: [0.1, 0.3] } } }\n          feature {\n            key: "image/object/bbox/ymax"\n            value { float_list { value: [0.3, 0.6] } } }\n          feature {\n            key: "image/object/bbox/xmax"\n            value { float_list { value: [0.2, 0.3] } } }\n          feature {\n            key: "image/object/class/label"\n            value { int64_list { value: [0, 0] } } }\n          feature {\n            key: "image/object/class/text"\n            value { bytes_list { value: ["a", "a"] } } }\n          feature {\n            key: "image/source_id"\n            value { bytes_list { value: "i1" } } }\n          feature {\n            key: "image/object/depiction"\n            value { int64_list { value: [1, 0] } } }\n          feature {\n            key: "image/object/group_of"\n            value { int64_list { value: [0, 0] } } }\n          feature {\n            key: "image/object/occluded"\n            value { int64_list { value: [0, 1] } } }\n          feature {\n            key: "image/object/truncated"\n            value { int64_list { value: [0, 0] } } }\n          feature {\n            key: "image/class/label"\n            value { int64_list { } } }\n          feature {\n            key: "image/class/text"\n            value { bytes_list { } } } }\n    ', tf_example)
if __name__ == '__main__':
    tf.test.main()