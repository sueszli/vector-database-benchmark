"""
This module implements the ObjectSeeker certifiably robust defense.

| Paper link: https://arxiv.org/abs/2202.01811
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import logging
from typing import Dict, List, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm
from art.utils import intersection_over_area, non_maximum_suppression
logger = logging.getLogger(__name__)

class ObjectSeekerMixin(abc.ABC):
    """
    Implementation of the ObjectSeeker certifiable robust defense applied to object detection models.
    The original implementation is https://github.com/inspire-group/ObjectSeeker

    | Paper link: https://arxiv.org/abs/2202.01811
    """

    def __init__(self, *args, num_lines: int=3, confidence_threshold: float=0.3, iou_threshold: float=0.5, prune_threshold: float=0.5, epsilon: float=0.1, verbose: bool=False, **kwargs) -> None:
        if False:
            return 10
        '\n        Create an ObjectSeeker wrapper.\n\n        :param num_lines: The number of divisions both vertically and horizontally to make masked predictions.\n        :param confidence_threshold: The confidence threshold to discard bounding boxes.\n        :param iou_threshold: The IoU threshold to discard overlapping bounding boxes.\n        :param prune_threshold: The IoA threshold for pruning and duplicated bounding boxes.\n        :param epsilon: The maximum distance between bounding boxes when merging using DBSCAN.\n        :param verbose: Show progress bars.\n        '
        super().__init__(*args, **kwargs)
        self.num_lines = num_lines
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.prune_threshold = prune_threshold
        self.epsilon = epsilon
        self.verbose = verbose

    @property
    @abc.abstractmethod
    def channels_first(self) -> bool:
        if False:
            return 10
        '\n        :return: Boolean to indicate index of the color channels in the sample `x`.\n        '
        pass

    @property
    @abc.abstractmethod
    def input_shape(self) -> Tuple[int, ...]:
        if False:
            while True:
                i = 10
        '\n        :return: Shape of one input sample.\n        '
        pass

    @abc.abstractmethod
    def _predict_classifier(self, x: np.ndarray, batch_size: int=128, **kwargs) -> List[Dict[str, np.ndarray]]:
        if False:
            return 10
        '\n        Perform prediction for a batch of inputs.\n\n        :param x: Samples of shape NCHW or NHWC.\n        :param batch_size: Batch size.\n        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict\n                 are as follows:\n\n                 - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                 - labels [N]: the labels for each image\n                 - scores [N]: the scores or each prediction.\n        '
        raise NotImplementedError

    def predict(self, x: np.ndarray, batch_size: int=128, **kwargs) -> List[Dict[str, np.ndarray]]:
        if False:
            print('Hello World!')
        '\n        Perform prediction for a batch of inputs.\n\n        :param x: Samples of shape NCHW or NHWC.\n        :param batch_size: Batch size.\n        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict\n                 are as follows:\n\n                 - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                 - labels [N]: the labels for each image\n                 - scores [N]: the scores or each prediction.\n        '
        predictions = []
        for x_i in tqdm(x, desc='ObjectSeeker', disable=not self.verbose):
            (base_preds, masked_preds) = self._masked_predictions(x_i, batch_size=batch_size, **kwargs)
            pruned_preds = self._prune_boxes(masked_preds, base_preds)
            unionized_preds = self._unionize_clusters(pruned_preds)
            preds = {'boxes': np.concatenate([base_preds['boxes'], unionized_preds['boxes']]), 'labels': np.concatenate([base_preds['labels'], unionized_preds['labels']]), 'scores': np.concatenate([base_preds['scores'], unionized_preds['scores']])}
            predictions.append(preds)
        return predictions

    def _masked_predictions(self, x_i: np.ndarray, batch_size: int=128, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        if False:
            while True:
                i = 10
        '\n        Create masked copies of the image for each of lines following the ObjectSeeker algorithm. Then creates\n        predictions on the base unmasked image and each of the masked image.\n\n        :param x_i: A single image of shape CHW or HWC.\n        :batch_size: Batch size.\n        :return: Predictions for the base unmasked image and merged predictions for the masked image.\n        '
        x_mask = np.repeat(x_i[np.newaxis], self.num_lines * 4 + 1, axis=0)
        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]
            x_mask = np.transpose(x_mask, (0, 3, 1, 2))
        idx = 1
        for k in range(1, self.num_lines + 1):
            boundary = int(width / (self.num_lines + 1) * k)
            x_mask[idx, :, :, :boundary] = 0
            idx += 1
        for k in range(1, self.num_lines + 1):
            boundary = width - int(width / (self.num_lines + 1) * k)
            x_mask[idx, :, :, boundary:] = 0
            idx += 1
        for k in range(1, self.num_lines + 1):
            boundary = int(height / (self.num_lines + 1) * k)
            x_mask[idx, :, :boundary, :] = 0
            idx += 1
        for k in range(1, self.num_lines + 1):
            boundary = height - int(height / (self.num_lines + 1) * k)
            x_mask[idx, :, boundary:, :] = 0
            idx += 1
        if not self.channels_first:
            x_mask = np.transpose(x_mask, (0, 2, 3, 1))
        predictions = self._predict_classifier(x=x_mask, batch_size=batch_size, **kwargs)
        filtered_predictions = [non_maximum_suppression(pred, iou_threshold=self.iou_threshold, confidence_threshold=self.confidence_threshold) for pred in predictions]
        base_predictions = filtered_predictions[0]
        boxes = np.concatenate([pred['boxes'] for pred in filtered_predictions[1:]])
        labels = np.concatenate([pred['labels'] for pred in filtered_predictions[1:]])
        scores = np.concatenate([pred['scores'] for pred in filtered_predictions[1:]])
        merged_predictions = {'boxes': boxes, 'labels': labels, 'scores': scores}
        masked_predictions = non_maximum_suppression(merged_predictions, iou_threshold=self.iou_threshold, confidence_threshold=self.confidence_threshold)
        return (base_predictions, masked_predictions)

    def _prune_boxes(self, masked_preds: Dict[str, np.ndarray], base_preds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if False:
            while True:
                i = 10
        '\n        Remove bounding boxes from the masked predictions of a single image based on the IoA score with the boxes\n        on the base unmasked predictions.\n\n        :param masked_preds: The merged masked predictions of a single image.\n        :param base_preds: The base unmasked predictions of a single image.\n        :return: The filtered masked predictions with extraneous boxes removed.\n        '
        masked_boxes = masked_preds['boxes']
        masked_labels = masked_preds['labels']
        masked_scores = masked_preds['scores']
        base_boxes = base_preds['boxes']
        base_labels = base_preds['labels']
        keep_indices = []
        for (idx, (masked_box, masked_label)) in enumerate(zip(masked_boxes, masked_labels)):
            keep = True
            for (base_box, base_label) in zip(base_boxes, base_labels):
                if masked_label == base_label:
                    ioa = intersection_over_area(masked_box, base_box)
                    if ioa >= self.prune_threshold:
                        keep = False
                        break
            if keep:
                keep_indices.append(idx)
        pruned_preds = {'boxes': masked_boxes[keep_indices], 'labels': masked_labels[keep_indices], 'scores': masked_scores[keep_indices]}
        return pruned_preds

    def _unionize_clusters(self, masked_preds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if False:
            while True:
                i = 10
        '\n        Cluster the bounding boxes for the pruned masked predictions.\n\n        :param masked_preds: The merged masked predictions of a single image already pruned.\n        :return: The clustered masked predictions with overlapping boxes merged.\n        '
        boxes = masked_preds['boxes']
        labels = masked_preds['labels']
        scores = masked_preds['scores']
        if len(boxes) <= 1:
            return masked_preds
        unionized_boxes = []
        unionized_labels = []
        unionized_scores = []
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            selected_boxes = boxes[mask]
            selected_scores = scores[mask]
            areas = (selected_boxes[:, 2] - selected_boxes[:, 0]) * (selected_boxes[:, 3] - selected_boxes[:, 1])
            top_left = np.maximum(selected_boxes[:, None, :2], selected_boxes[:, :2])
            bottom_right = np.minimum(selected_boxes[:, None, 2:], selected_boxes[:, 2:])
            pairwise_intersection = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
            pairwise_ioa = pairwise_intersection / areas[:, None]
            distances = 1 - np.maximum(pairwise_ioa, pairwise_ioa.T)
            dbscan = DBSCAN(eps=self.epsilon, min_samples=1, metric='precomputed')
            clusters = dbscan.fit_predict(distances)
            num_clusters = np.max(clusters) + 1
            for cluster in range(num_clusters):
                clustered_boxes = selected_boxes[clusters == cluster]
                clustered_box = [np.min(clustered_boxes[:, 0]), np.min(clustered_boxes[:, 1]), np.max(clustered_boxes[:, 2]), np.max(clustered_boxes[:, 3])]
                clustered_score = np.max(selected_scores)
                unionized_boxes.append(clustered_box)
                unionized_labels.append(label)
                unionized_scores.append(clustered_score)
        unionized_predictions = {'boxes': np.asarray(unionized_boxes), 'labels': np.asarray(unionized_labels), 'scores': np.asarray(unionized_scores)}
        return unionized_predictions

    def certify(self, x: np.ndarray, patch_size: float=0.01, offset: float=0.1, batch_size: int=128) -> List[np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks if there is certifiable IoA robustness for each predicted bounding box.\n\n        :param x: Sample input with shape as expected by the model.\n        :param patch_size: The size of the patch to check against.\n        :param offset: The offset to distinguish between the far and near patches.\n        :return: A list containing an array of bools for each bounding box per image indicating if the bounding\n                 box is certified against the given patch.\n        '
        if self.channels_first:
            (_, height, width) = self.input_shape
        else:
            (height, width, _) = self.input_shape
        patch_size = np.sqrt(height * width * patch_size)
        height_offset = offset * height
        width_offset = offset * width
        predictions = self.predict(x, batch_size=batch_size)
        certifications: List[np.ndarray] = []
        for pred in tqdm(predictions, desc='ObjectSeeker', disable=not self.verbose):
            boxes = pred['boxes']
            far_patch_map = np.ones((len(boxes), height, width), dtype=bool)
            for (i, box) in enumerate(boxes):
                x_1 = int(max(0, box[1] - patch_size - height_offset))
                x_2 = int(min(box[3] + height_offset + 1, height))
                y_1 = int(max(0, box[0] - patch_size - width_offset))
                y_2 = int(min(box[2] + width_offset + 1, width))
                far_patch_map[i, x_1:x_2, y_1:y_2] = False
            far_vulnerable = np.any(far_patch_map, axis=(-2, -1))
            close_patch_map = np.ones((len(boxes), height, width), dtype=bool)
            for (i, box) in enumerate(boxes):
                x_1 = int(max(0, box[1] - patch_size))
                x_2 = int(min(box[3] + 1, height))
                y_1 = int(max(0, box[0] - patch_size))
                y_2 = int(min(box[2] + 1, width))
                close_patch_map[i, x_1:x_2, y_1:y_2] = False
            close_vulnerable = np.any(close_patch_map, axis=(-2, -1))
            close_patch_map = np.ones((len(boxes), height, width), dtype=bool)
            for (i, box) in enumerate(boxes):
                x_1 = int(max(0, box[1] - patch_size))
                x_2 = int(min(box[3] + 1, height))
                y_1 = int(max(0, box[0] - patch_size))
                y_2 = int(min(box[2] + 1, width))
                close_patch_map[i, x_1:x_2, y_1:y_2] = True
            over_vulnerable = np.any(close_patch_map, axis=(-2, -1))
            cert = np.logical_and.reduce((far_vulnerable, close_vulnerable, over_vulnerable))
            certifications.append(cert)
        return certifications