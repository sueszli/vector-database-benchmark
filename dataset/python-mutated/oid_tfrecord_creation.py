"""Utilities for creating TFRecords of TF examples for the Open Images dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from object_detection.core import standard_fields
from object_detection.utils import dataset_util

def tf_example_from_annotations_data_frame(annotations_data_frame, label_map, encoded_image):
    if False:
        print('Hello World!')
    'Populates a TF Example message with image annotations from a data frame.\n\n  Args:\n    annotations_data_frame: Data frame containing the annotations for a single\n      image.\n    label_map: String to integer label map.\n    encoded_image: The encoded image string\n\n  Returns:\n    The populated TF Example, if the label of at least one object is present in\n    label_map. Otherwise, returns None.\n  '
    filtered_data_frame = annotations_data_frame[annotations_data_frame.LabelName.isin(label_map)]
    filtered_data_frame_boxes = filtered_data_frame[~filtered_data_frame.YMin.isnull()]
    filtered_data_frame_labels = filtered_data_frame[filtered_data_frame.YMin.isnull()]
    image_id = annotations_data_frame.ImageID.iloc[0]
    feature_map = {standard_fields.TfExampleFields.object_bbox_ymin: dataset_util.float_list_feature(filtered_data_frame_boxes.YMin.as_matrix()), standard_fields.TfExampleFields.object_bbox_xmin: dataset_util.float_list_feature(filtered_data_frame_boxes.XMin.as_matrix()), standard_fields.TfExampleFields.object_bbox_ymax: dataset_util.float_list_feature(filtered_data_frame_boxes.YMax.as_matrix()), standard_fields.TfExampleFields.object_bbox_xmax: dataset_util.float_list_feature(filtered_data_frame_boxes.XMax.as_matrix()), standard_fields.TfExampleFields.object_class_text: dataset_util.bytes_list_feature(filtered_data_frame_boxes.LabelName.as_matrix()), standard_fields.TfExampleFields.object_class_label: dataset_util.int64_list_feature(filtered_data_frame_boxes.LabelName.map(lambda x: label_map[x]).as_matrix()), standard_fields.TfExampleFields.filename: dataset_util.bytes_feature('{}.jpg'.format(image_id)), standard_fields.TfExampleFields.source_id: dataset_util.bytes_feature(image_id), standard_fields.TfExampleFields.image_encoded: dataset_util.bytes_feature(encoded_image)}
    if 'IsGroupOf' in filtered_data_frame.columns:
        feature_map[standard_fields.TfExampleFields.object_group_of] = dataset_util.int64_list_feature(filtered_data_frame_boxes.IsGroupOf.as_matrix().astype(int))
    if 'IsOccluded' in filtered_data_frame.columns:
        feature_map[standard_fields.TfExampleFields.object_occluded] = dataset_util.int64_list_feature(filtered_data_frame_boxes.IsOccluded.as_matrix().astype(int))
    if 'IsTruncated' in filtered_data_frame.columns:
        feature_map[standard_fields.TfExampleFields.object_truncated] = dataset_util.int64_list_feature(filtered_data_frame_boxes.IsTruncated.as_matrix().astype(int))
    if 'IsDepiction' in filtered_data_frame.columns:
        feature_map[standard_fields.TfExampleFields.object_depiction] = dataset_util.int64_list_feature(filtered_data_frame_boxes.IsDepiction.as_matrix().astype(int))
    if 'ConfidenceImageLabel' in filtered_data_frame_labels.columns:
        feature_map[standard_fields.TfExampleFields.image_class_label] = dataset_util.int64_list_feature(filtered_data_frame_labels.LabelName.map(lambda x: label_map[x]).as_matrix())
        feature_map[standard_fields.TfExampleFields.image_class_text] = (dataset_util.bytes_list_feature(filtered_data_frame_labels.LabelName.as_matrix()),)
    return tf.train.Example(features=tf.train.Features(feature=feature_map))