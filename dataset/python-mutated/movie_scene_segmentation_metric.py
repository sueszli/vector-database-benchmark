from typing import Dict
import numpy as np
from modelscope.metainfo import Metrics
from modelscope.utils.registry import default_group
from modelscope.utils.tensor_utils import torch_nested_detach, torch_nested_numpify
from .base import Metric
from .builder import METRICS, MetricKeys

@METRICS.register_module(group_key=default_group, module_name=Metrics.movie_scene_segmentation_metric)
class MovieSceneSegmentationMetric(Metric):
    """The metric computation class for movie scene segmentation classes.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.preds = []
        self.labels = []
        self.eps = 1e-05

    def add(self, outputs: Dict, inputs: Dict):
        if False:
            return 10
        preds = outputs['pred']
        labels = inputs['label']
        self.preds.extend(preds)
        self.labels.extend(labels)

    def evaluate(self):
        if False:
            print('Hello World!')
        gts = np.array(torch_nested_numpify(torch_nested_detach(self.labels)))
        prob = np.array(torch_nested_numpify(torch_nested_detach(self.preds)))
        gt_one = gts == 1
        gt_zero = gts == 0
        pred_one = prob == 1
        pred_zero = prob == 0
        tp = (gt_one * pred_one).sum()
        fp = (gt_zero * pred_one).sum()
        fn = (gt_one * pred_zero).sum()
        precision = 100.0 * tp / (tp + fp + self.eps)
        recall = 100.0 * tp / (tp + fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall)
        return {MetricKeys.F1: f1, MetricKeys.RECALL: recall, MetricKeys.PRECISION: precision}

    def merge(self, other: 'MovieSceneSegmentationMetric'):
        if False:
            while True:
                i = 10
        self.preds.extend(other.preds)
        self.labels.extend(other.labels)

    def __getstate__(self):
        if False:
            while True:
                i = 10
        return (self.preds, self.labels)

    def __setstate__(self, state):
        if False:
            return 10
        self.__init__()
        (self.preds, self.labels) = state