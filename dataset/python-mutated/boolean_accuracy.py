from typing import Optional
import torch
from allennlp.nn.util import dist_reduce_sum
from allennlp.training.metrics.metric import Metric

@Metric.register('boolean_accuracy')
class BooleanAccuracy(Metric):
    """
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.
    That is, if your prediction has shape (batch_size, dim_1, ..., dim_n), this metric considers that
    as a set of `batch_size` predictions and checks that each is *entirely* correct across the remaining dims.
    This means the denominator in the accuracy computation is `batch_size`, with the caveat that predictions
    that are totally masked are ignored (in which case the denominator is the number of predictions that have
    at least one unmasked element).

    This is similar to [`CategoricalAccuracy`](./categorical_accuracy.md), if you've already done a `.max()`
    on your predictions.  If you have categorical output, though, you should typically just use
    `CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    `CategoricalAccuracy`, which assumes a final dimension of size `num_classes`.
    """

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._correct_count = 0.0
        self._total_count = 0.0

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor]=None):
        if False:
            print('Hello World!')
        '\n        # Parameters\n\n        predictions : `torch.Tensor`, required.\n            A tensor of predictions of shape (batch_size, ...).\n        gold_labels : `torch.Tensor`, required.\n            A tensor of the same shape as `predictions`.\n        mask : `torch.BoolTensor`, optional (default = `None`).\n            A tensor of the same shape as `predictions`.\n        '
        (predictions, gold_labels, mask) = self.detach_tensors(predictions, gold_labels, mask)
        if gold_labels.size() != predictions.size():
            raise ValueError(f'gold_labels must have shape == predictions.size() but found tensor of shape: {gold_labels.size()}')
        if mask is not None and mask.size() != predictions.size():
            raise ValueError(f'mask must have shape == predictions.size() but found tensor of shape: {mask.size()}')
        batch_size = predictions.size(0)
        if mask is not None:
            predictions = predictions * mask
            gold_labels = gold_labels * mask
            keep = mask.view(batch_size, -1).max(dim=1)[0]
        else:
            keep = torch.ones(batch_size, device=predictions.device).bool()
        predictions = predictions.view(batch_size, -1)
        gold_labels = gold_labels.view(batch_size, -1)
        correct = predictions.eq(gold_labels).prod(dim=1).float()
        _correct_count = (correct * keep).sum()
        _total_count = keep.sum()
        self._correct_count += dist_reduce_sum(_correct_count).item()
        self._total_count += dist_reduce_sum(_total_count).item()

    def get_metric(self, reset: bool=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        # Returns\n\n        The accumulated accuracy.\n        '
        if self._total_count > 0:
            accuracy = float(self._correct_count) / float(self._total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self._correct_count = 0.0
        self._total_count = 0.0