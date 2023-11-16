"""Converts data from CSV format to the VRDDetectionEvaluator format."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from object_detection.core import standard_fields
from object_detection.utils import vrd_evaluation

def build_groundtruth_vrd_dictionary(data, class_label_map, relationship_label_map):
    if False:
        i = 10
        return i + 15
    'Builds a groundtruth dictionary from groundtruth data in CSV file.\n\n  Args:\n    data: Pandas DataFrame with the groundtruth data for a single image.\n    class_label_map: Class labelmap from string label name to an integer.\n    relationship_label_map: Relationship type labelmap from string name to an\n      integer.\n\n  Returns:\n    A dictionary with keys suitable for passing to\n    VRDDetectionEvaluator.add_single_ground_truth_image_info:\n        standard_fields.InputDataFields.groundtruth_boxes: A numpy array\n          of structures with the shape [M, 1], representing M tuples, each tuple\n          containing the same number of named bounding boxes.\n          Each box is of the format [y_min, x_min, y_max, x_max] (see\n          datatype vrd_box_data_type, single_box_data_type above).\n        standard_fields.InputDataFields.groundtruth_classes: A numpy array of\n          structures shape [M, 1], representing  the class labels of the\n          corresponding bounding boxes and possibly additional classes (see\n          datatype label_data_type above).\n        standard_fields.InputDataFields.verified_labels: numpy array\n          of shape [K] containing verified labels.\n  '
    data_boxes = data[data.LabelName.isnull()]
    data_labels = data[data.LabelName1.isnull()]
    boxes = np.zeros(data_boxes.shape[0], dtype=vrd_evaluation.vrd_box_data_type)
    boxes['subject'] = data_boxes[['YMin1', 'XMin1', 'YMax1', 'XMax1']].as_matrix()
    boxes['object'] = data_boxes[['YMin2', 'XMin2', 'YMax2', 'XMax2']].as_matrix()
    labels = np.zeros(data_boxes.shape[0], dtype=vrd_evaluation.label_data_type)
    labels['subject'] = data_boxes['LabelName1'].map(lambda x: class_label_map[x]).as_matrix()
    labels['object'] = data_boxes['LabelName2'].map(lambda x: class_label_map[x]).as_matrix()
    labels['relation'] = data_boxes['RelationshipLabel'].map(lambda x: relationship_label_map[x]).as_matrix()
    return {standard_fields.InputDataFields.groundtruth_boxes: boxes, standard_fields.InputDataFields.groundtruth_classes: labels, standard_fields.InputDataFields.groundtruth_image_classes: data_labels['LabelName'].map(lambda x: class_label_map[x]).as_matrix()}

def build_predictions_vrd_dictionary(data, class_label_map, relationship_label_map):
    if False:
        i = 10
        return i + 15
    'Builds a predictions dictionary from predictions data in CSV file.\n\n  Args:\n    data: Pandas DataFrame with the predictions data for a single image.\n    class_label_map: Class labelmap from string label name to an integer.\n    relationship_label_map: Relationship type labelmap from string name to an\n      integer.\n\n  Returns:\n    Dictionary with keys suitable for passing to\n    VRDDetectionEvaluator.add_single_detected_image_info:\n        standard_fields.DetectionResultFields.detection_boxes: A numpy array of\n          structures with shape [N, 1], representing N tuples, each tuple\n          containing the same number of named bounding boxes.\n          Each box is of the format [y_min, x_min, y_max, x_max] (as an example\n          see datatype vrd_box_data_type, single_box_data_type above).\n        standard_fields.DetectionResultFields.detection_scores: float32 numpy\n          array of shape [N] containing detection scores for the boxes.\n        standard_fields.DetectionResultFields.detection_classes: A numpy array\n          of structures shape [N, 1], representing the class labels of the\n          corresponding bounding boxes and possibly additional classes (see\n          datatype label_data_type above).\n  '
    data_boxes = data
    boxes = np.zeros(data_boxes.shape[0], dtype=vrd_evaluation.vrd_box_data_type)
    boxes['subject'] = data_boxes[['YMin1', 'XMin1', 'YMax1', 'XMax1']].as_matrix()
    boxes['object'] = data_boxes[['YMin2', 'XMin2', 'YMax2', 'XMax2']].as_matrix()
    labels = np.zeros(data_boxes.shape[0], dtype=vrd_evaluation.label_data_type)
    labels['subject'] = data_boxes['LabelName1'].map(lambda x: class_label_map[x]).as_matrix()
    labels['object'] = data_boxes['LabelName2'].map(lambda x: class_label_map[x]).as_matrix()
    labels['relation'] = data_boxes['RelationshipLabel'].map(lambda x: relationship_label_map[x]).as_matrix()
    return {standard_fields.DetectionResultFields.detection_boxes: boxes, standard_fields.DetectionResultFields.detection_classes: labels, standard_fields.DetectionResultFields.detection_scores: data_boxes['Score'].as_matrix()}