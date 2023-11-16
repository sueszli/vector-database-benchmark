from typing import Dict
from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.fbeta_measure import FBetaMeasure

@Metric.register('f1')
class F1Measure(FBetaMeasure):
    """
    Computes Precision, Recall and F1 with respect to a given `positive_label`.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """

    def __init__(self, positive_label: int) -> None:
        if False:
            return 10
        super().__init__(beta=1, labels=[positive_label])
        self._positive_label = positive_label

    def get_metric(self, reset: bool=False) -> Dict[str, float]:
        if False:
            i = 10
            return i + 15
        '\n        # Returns\n\n        precision : `float`\n        recall : `float`\n        f1-measure : `float`\n        '
        metric = super().get_metric(reset=reset)
        precision = metric['precision'][0]
        recall = metric['recall'][0]
        f1 = metric['fscore'][0]
        return {'precision': precision, 'recall': recall, 'f1': f1}

    @property
    def _true_positives(self):
        if False:
            print('Hello World!')
        if self._true_positive_sum is None:
            return 0.0
        else:
            return self._true_positive_sum[self._positive_label]

    @property
    def _true_negatives(self):
        if False:
            while True:
                i = 10
        if self._true_negative_sum is None:
            return 0.0
        else:
            return self._true_negative_sum[self._positive_label]

    @property
    def _false_positives(self):
        if False:
            return 10
        if self._pred_sum is None:
            return 0.0
        else:
            return self._pred_sum[self._positive_label] - self._true_positives

    @property
    def _false_negatives(self):
        if False:
            i = 10
            return i + 15
        if self._true_sum is None:
            return 0.0
        else:
            return self._true_sum[self._positive_label] - self._true_positives