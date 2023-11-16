"""Module for Metric Mixin."""
import typing as t
from abc import abstractmethod
import numpy as np
from deepchecks.vision.metrics_utils.iou_utils import compute_pairwise_ious, group_class_detection_label, jaccard_iou

class MetricMixin:
    """Metric util function mixin."""

    @abstractmethod
    def get_confidences(self, detections) -> t.List[float]:
        if False:
            i = 10
            return i + 15
        'Get detections object of single image and should return confidence for each detection.'
        pass

    @abstractmethod
    def calc_pairwise_ious(self, detections, labels) -> t.Dict[int, np.ndarray]:
        if False:
            print('Hello World!')
        'Get a single result from group_class_detection_label and return a matrix of IoUs.'
        pass

    @abstractmethod
    def group_class_detection_label(self, detections, labels) -> t.Dict[t.Any, t.Dict[str, list]]:
        if False:
            return 10
        "Group detection and labels in dict of format {class_id: {'detected' [...], 'ground_truth': [...]}}."
        pass

    @abstractmethod
    def get_detection_areas(self, detections) -> t.List[int]:
        if False:
            while True:
                i = 10
        'Get detection object of single image and should return area for each detection.'
        pass

    @abstractmethod
    def get_labels_areas(self, labels) -> t.List[int]:
        if False:
            i = 10
            return i + 15
        'Get labels object of single image and should return area for each label.'
        pass

class ObjectDetectionMetricMixin(MetricMixin):
    """Metric util function mixin for object detection."""

    def get_labels_areas(self, labels) -> t.List[int]:
        if False:
            print('Hello World!')
        'Get labels object of single image and should return area for each label.'
        return [bbox[3] * bbox[4] for bbox in labels]

    def group_class_detection_label(self, detections, labels) -> t.Dict[t.Any, t.Dict[str, list]]:
        if False:
            print('Hello World!')
        "Group detection and labels in dict of format {class_id: {'detected' [...], 'ground_truth': [...] }}."
        return group_class_detection_label(detections, labels)

    def get_confidences(self, detections) -> t.List[float]:
        if False:
            while True:
                i = 10
        'Get detections object of single image and should return confidence for each detection.'
        return [d[4] for d in detections]

    def calc_pairwise_ious(self, detections, labels) -> np.ndarray:
        if False:
            print('Hello World!')
        'Get a single result from group_class_detection_label and return a matrix of IoUs.'
        return compute_pairwise_ious(detections, labels, jaccard_iou)

    def get_detection_areas(self, detections) -> t.List[int]:
        if False:
            return 10
        'Get detection object of single image and should return area for each detection.'
        return [d[2] * d[3] for d in detections]