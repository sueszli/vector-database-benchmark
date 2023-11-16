"""Converts data from CSV to the OpenImagesDetectionChallengeEvaluator format."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import base64
import zlib
import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask
from object_detection.core import standard_fields

def _to_normalized_box(mask_np):
    if False:
        while True:
            i = 10
    'Decodes binary segmentation masks into np.arrays and boxes.\n\n  Args:\n    mask_np: np.ndarray of size NxWxH.\n\n  Returns:\n    a np.ndarray of the size Nx4, each row containing normalized coordinates\n    [YMin, XMin, YMax, XMax] of a box computed of axis parallel enclosing box of\n    a mask.\n  '
    (coord1, coord2) = np.nonzero(mask_np)
    if coord1.size > 0:
        ymin = float(min(coord1)) / mask_np.shape[0]
        ymax = float(max(coord1) + 1) / mask_np.shape[0]
        xmin = float(min(coord2)) / mask_np.shape[1]
        xmax = float(max(coord2) + 1) / mask_np.shape[1]
        return np.array([ymin, xmin, ymax, xmax])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])

def _decode_raw_data_into_masks_and_boxes(segments, image_widths, image_heights):
    if False:
        i = 10
        return i + 15
    'Decods binary segmentation masks into np.arrays and boxes.\n\n  Args:\n    segments: pandas Series object containing either\n      None entries, or strings with\n      base64, zlib compressed, COCO RLE-encoded binary masks.\n      All masks are expected to be the same size.\n    image_widths: pandas Series of mask widths.\n    image_heights: pandas Series of mask heights.\n\n  Returns:\n    a np.ndarray of the size NxWxH, where W and H is determined from the encoded\n    masks; for the None values, zero arrays of size WxH are created. If input\n    contains only None values, W=1, H=1.\n  '
    segment_masks = []
    segment_boxes = []
    ind = segments.first_valid_index()
    if ind is not None:
        size = [int(image_heights[ind]), int(image_widths[ind])]
    else:
        return (np.zeros((segments.shape[0], 1, 1), dtype=np.uint8), np.zeros((segments.shape[0], 4), dtype=np.float32))
    for (segment, im_width, im_height) in zip(segments, image_widths, image_heights):
        if pd.isnull(segment):
            segment_masks.append(np.zeros([1, size[0], size[1]], dtype=np.uint8))
            segment_boxes.append(np.expand_dims(np.array([0.0, 0.0, 0.0, 0.0]), 0))
        else:
            compressed_mask = base64.b64decode(segment)
            rle_encoded_mask = zlib.decompress(compressed_mask)
            decoding_dict = {'size': [im_height, im_width], 'counts': rle_encoded_mask}
            mask_tensor = coco_mask.decode(decoding_dict)
            segment_masks.append(np.expand_dims(mask_tensor, 0))
            segment_boxes.append(np.expand_dims(_to_normalized_box(mask_tensor), 0))
    return (np.concatenate(segment_masks, axis=0), np.concatenate(segment_boxes, axis=0))

def merge_boxes_and_masks(box_data, mask_data):
    if False:
        while True:
            i = 10
    return pd.merge(box_data, mask_data, how='outer', on=['LabelName', 'ImageID', 'XMin', 'XMax', 'YMin', 'YMax', 'IsGroupOf'])

def build_groundtruth_dictionary(data, class_label_map):
    if False:
        return 10
    'Builds a groundtruth dictionary from groundtruth data in CSV file.\n\n  Args:\n    data: Pandas DataFrame with the groundtruth data for a single image.\n    class_label_map: Class labelmap from string label name to an integer.\n\n  Returns:\n    A dictionary with keys suitable for passing to\n    OpenImagesDetectionChallengeEvaluator.add_single_ground_truth_image_info:\n        standard_fields.InputDataFields.groundtruth_boxes: float32 numpy array\n          of shape [num_boxes, 4] containing `num_boxes` groundtruth boxes of\n          the format [ymin, xmin, ymax, xmax] in absolute image coordinates.\n        standard_fields.InputDataFields.groundtruth_classes: integer numpy array\n          of shape [num_boxes] containing 1-indexed groundtruth classes for the\n          boxes.\n        standard_fields.InputDataFields.verified_labels: integer 1D numpy array\n          containing all classes for which labels are verified.\n        standard_fields.InputDataFields.groundtruth_group_of: Optional length\n          M numpy boolean array denoting whether a groundtruth box contains a\n          group of instances.\n  '
    data_location = data[data.XMin.notnull()]
    data_labels = data[data.ConfidenceImageLabel.notnull()]
    dictionary = {standard_fields.InputDataFields.groundtruth_boxes: data_location[['YMin', 'XMin', 'YMax', 'XMax']].as_matrix(), standard_fields.InputDataFields.groundtruth_classes: data_location['LabelName'].map(lambda x: class_label_map[x]).as_matrix(), standard_fields.InputDataFields.groundtruth_group_of: data_location['IsGroupOf'].as_matrix().astype(int), standard_fields.InputDataFields.groundtruth_image_classes: data_labels['LabelName'].map(lambda x: class_label_map[x]).as_matrix()}
    if 'Mask' in data_location:
        (segments, _) = _decode_raw_data_into_masks_and_boxes(data_location['Mask'], data_location['ImageWidth'], data_location['ImageHeight'])
        dictionary[standard_fields.InputDataFields.groundtruth_instance_masks] = segments
    return dictionary

def build_predictions_dictionary(data, class_label_map):
    if False:
        for i in range(10):
            print('nop')
    'Builds a predictions dictionary from predictions data in CSV file.\n\n  Args:\n    data: Pandas DataFrame with the predictions data for a single image.\n    class_label_map: Class labelmap from string label name to an integer.\n\n  Returns:\n    Dictionary with keys suitable for passing to\n    OpenImagesDetectionChallengeEvaluator.add_single_detected_image_info:\n        standard_fields.DetectionResultFields.detection_boxes: float32 numpy\n          array of shape [num_boxes, 4] containing `num_boxes` detection boxes\n          of the format [ymin, xmin, ymax, xmax] in absolute image coordinates.\n        standard_fields.DetectionResultFields.detection_scores: float32 numpy\n          array of shape [num_boxes] containing detection scores for the boxes.\n        standard_fields.DetectionResultFields.detection_classes: integer numpy\n          array of shape [num_boxes] containing 1-indexed detection classes for\n          the boxes.\n\n  '
    dictionary = {standard_fields.DetectionResultFields.detection_classes: data['LabelName'].map(lambda x: class_label_map[x]).as_matrix(), standard_fields.DetectionResultFields.detection_scores: data['Score'].as_matrix()}
    if 'Mask' in data:
        (segments, boxes) = _decode_raw_data_into_masks_and_boxes(data['Mask'], data['ImageWidth'], data['ImageHeight'])
        dictionary[standard_fields.DetectionResultFields.detection_masks] = segments
        dictionary[standard_fields.DetectionResultFields.detection_boxes] = boxes
    else:
        dictionary[standard_fields.DetectionResultFields.detection_boxes] = data[['YMin', 'XMin', 'YMax', 'XMax']].as_matrix()
    return dictionary