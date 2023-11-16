"""Tests for oid_vrd_challenge_evaluation_utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
from object_detection.core import standard_fields
from object_detection.metrics import oid_vrd_challenge_evaluation_utils as utils
from object_detection.utils import vrd_evaluation

class OidVrdChallengeEvaluationUtilsTest(tf.test.TestCase):

    def testBuildGroundtruthDictionary(self):
        if False:
            print('Hello World!')
        np_data = pd.DataFrame([['fe58ec1b06db2bb7', '/m/04bcr3', '/m/083vt', 0.0, 0.3, 0.5, 0.6, 0.0, 0.3, 0.5, 0.6, 'is', None, None], ['fe58ec1b06db2bb7', '/m/04bcr3', '/m/02gy9n', 0.0, 0.3, 0.5, 0.6, 0.1, 0.2, 0.3, 0.4, 'under', None, None], ['fe58ec1b06db2bb7', '/m/04bcr3', '/m/083vt', 0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3, 'is', None, None], ['fe58ec1b06db2bb7', '/m/083vt', '/m/04bcr3', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 'at', None, None], ['fe58ec1b06db2bb7', None, None, None, None, None, None, None, None, None, None, None, '/m/04bcr3', 1.0], ['fe58ec1b06db2bb7', None, None, None, None, None, None, None, None, None, None, None, '/m/083vt', 0.0], ['fe58ec1b06db2bb7', None, None, None, None, None, None, None, None, None, None, None, '/m/02gy9n', 0.0]], columns=['ImageID', 'LabelName1', 'LabelName2', 'XMin1', 'XMax1', 'YMin1', 'YMax1', 'XMin2', 'XMax2', 'YMin2', 'YMax2', 'RelationshipLabel', 'LabelName', 'Confidence'])
        class_label_map = {'/m/04bcr3': 1, '/m/083vt': 2, '/m/02gy9n': 3}
        relationship_label_map = {'is': 1, 'under': 2, 'at': 3}
        groundtruth_dictionary = utils.build_groundtruth_vrd_dictionary(np_data, class_label_map, relationship_label_map)
        self.assertTrue(standard_fields.InputDataFields.groundtruth_boxes in groundtruth_dictionary)
        self.assertTrue(standard_fields.InputDataFields.groundtruth_classes in groundtruth_dictionary)
        self.assertTrue(standard_fields.InputDataFields.groundtruth_image_classes in groundtruth_dictionary)
        self.assertAllEqual(np.array([(1, 2, 1), (1, 3, 2), (1, 2, 1), (2, 1, 3)], dtype=vrd_evaluation.label_data_type), groundtruth_dictionary[standard_fields.InputDataFields.groundtruth_classes])
        expected_vrd_data = np.array([([0.5, 0.0, 0.6, 0.3], [0.5, 0.0, 0.6, 0.3]), ([0.5, 0.0, 0.6, 0.3], [0.3, 0.1, 0.4, 0.2]), ([0.2, 0.0, 0.3, 0.1], [0.2, 0.0, 0.3, 0.1]), ([0.3, 0.1, 0.4, 0.2], [0.7, 0.5, 0.8, 0.6])], dtype=vrd_evaluation.vrd_box_data_type)
        for field in expected_vrd_data.dtype.fields:
            self.assertNDArrayNear(expected_vrd_data[field], groundtruth_dictionary[standard_fields.InputDataFields.groundtruth_boxes][field], 1e-05)
        self.assertAllEqual(np.array([1, 2, 3]), groundtruth_dictionary[standard_fields.InputDataFields.groundtruth_image_classes])

    def testBuildPredictionDictionary(self):
        if False:
            for i in range(10):
                print('nop')
        np_data = pd.DataFrame([['fe58ec1b06db2bb7', '/m/04bcr3', '/m/083vt', 0.0, 0.3, 0.5, 0.6, 0.0, 0.3, 0.5, 0.6, 'is', 0.1], ['fe58ec1b06db2bb7', '/m/04bcr3', '/m/02gy9n', 0.0, 0.3, 0.5, 0.6, 0.1, 0.2, 0.3, 0.4, 'under', 0.2], ['fe58ec1b06db2bb7', '/m/04bcr3', '/m/083vt', 0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3, 'is', 0.3], ['fe58ec1b06db2bb7', '/m/083vt', '/m/04bcr3', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 'at', 0.4]], columns=['ImageID', 'LabelName1', 'LabelName2', 'XMin1', 'XMax1', 'YMin1', 'YMax1', 'XMin2', 'XMax2', 'YMin2', 'YMax2', 'RelationshipLabel', 'Score'])
        class_label_map = {'/m/04bcr3': 1, '/m/083vt': 2, '/m/02gy9n': 3}
        relationship_label_map = {'is': 1, 'under': 2, 'at': 3}
        prediction_dictionary = utils.build_predictions_vrd_dictionary(np_data, class_label_map, relationship_label_map)
        self.assertTrue(standard_fields.DetectionResultFields.detection_boxes in prediction_dictionary)
        self.assertTrue(standard_fields.DetectionResultFields.detection_classes in prediction_dictionary)
        self.assertTrue(standard_fields.DetectionResultFields.detection_scores in prediction_dictionary)
        self.assertAllEqual(np.array([(1, 2, 1), (1, 3, 2), (1, 2, 1), (2, 1, 3)], dtype=vrd_evaluation.label_data_type), prediction_dictionary[standard_fields.DetectionResultFields.detection_classes])
        expected_vrd_data = np.array([([0.5, 0.0, 0.6, 0.3], [0.5, 0.0, 0.6, 0.3]), ([0.5, 0.0, 0.6, 0.3], [0.3, 0.1, 0.4, 0.2]), ([0.2, 0.0, 0.3, 0.1], [0.2, 0.0, 0.3, 0.1]), ([0.3, 0.1, 0.4, 0.2], [0.7, 0.5, 0.8, 0.6])], dtype=vrd_evaluation.vrd_box_data_type)
        for field in expected_vrd_data.dtype.fields:
            self.assertNDArrayNear(expected_vrd_data[field], prediction_dictionary[standard_fields.DetectionResultFields.detection_boxes][field], 1e-05)
        self.assertNDArrayNear(np.array([0.1, 0.2, 0.3, 0.4]), prediction_dictionary[standard_fields.DetectionResultFields.detection_scores], 1e-05)
if __name__ == '__main__':
    tf.test.main()