from typing import Dict, Optional
import torch
from allennlp.training.metrics.metric import Metric
from allennlp.nn.util import dist_reduce_sum

@Metric.register('mean_absolute_error')
class MeanAbsoluteError(Metric):
    """
    This `Metric` calculates the mean absolute error (MAE) between two tensors.
    """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self._absolute_error = 0.0
        self._total_count = 0.0

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        # Parameters\n\n        predictions : `torch.Tensor`, required.\n            A tensor of predictions of shape (batch_size, ...).\n        gold_labels : `torch.Tensor`, required.\n            A tensor of the same shape as `predictions`.\n        mask : `torch.BoolTensor`, optional (default = `None`).\n            A tensor of the same shape as `predictions`.\n        '
        (predictions, gold_labels, mask) = self.detach_tensors(predictions, gold_labels, mask)
        absolute_errors = torch.abs(predictions - gold_labels)
        if mask is not None:
            absolute_errors *= mask
            _total_count = torch.sum(mask)
        else:
            _total_count = gold_labels.numel()
        _absolute_error = torch.sum(absolute_errors)
        self._absolute_error += float(dist_reduce_sum(_absolute_error))
        self._total_count += int(dist_reduce_sum(_total_count))

    def get_metric(self, reset: bool=False) -> Dict[str, float]:
        if False:
            i = 10
            return i + 15
        '\n        # Returns\n\n        The accumulated mean absolute error.\n        '
        mean_absolute_error = self._absolute_error / self._total_count
        if reset:
            self.reset()
        return {'mae': mean_absolute_error}

    def reset(self) -> None:
        if False:
            i = 10
            return i + 15
        self._absolute_error = 0.0
        self._total_count = 0.0