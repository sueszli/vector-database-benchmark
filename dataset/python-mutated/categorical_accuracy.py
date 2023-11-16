from typing import Optional
import torch
from allennlp.nn.util import dist_reduce_sum
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

@Metric.register('categorical_accuracy')
class CategoricalAccuracy(Metric):
    """
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    Tie break enables equal distribution of scores among the
    classes with same maximum predicted scores.
    """
    supports_distributed = True

    def __init__(self, top_k: int=1, tie_break: bool=False) -> None:
        if False:
            return 10
        if top_k > 1 and tie_break:
            raise ConfigurationError('Tie break in Categorical Accuracy can be done only for maximum (top_k = 1)')
        if top_k <= 0:
            raise ConfigurationError('top_k passed to Categorical Accuracy must be > 0')
        self._top_k = top_k
        self._tie_break = tie_break
        self.correct_count = 0.0
        self.total_count = 0.0

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor]=None):
        if False:
            while True:
                i = 10
        '\n        # Parameters\n\n        predictions : `torch.Tensor`, required.\n            A tensor of predictions of shape (batch_size, ..., num_classes).\n        gold_labels : `torch.Tensor`, required.\n            A tensor of integer class label of shape (batch_size, ...). It must be the same\n            shape as the `predictions` tensor without the `num_classes` dimension.\n        mask : `torch.BoolTensor`, optional (default = `None`).\n            A masking tensor the same size as `gold_labels`.\n        '
        (predictions, gold_labels, mask) = self.detach_tensors(predictions, gold_labels, mask)
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError('gold_labels must have dimension == predictions.size() - 1 but found tensor of shape: {}'.format(predictions.size()))
        if (gold_labels >= num_classes).any():
            raise ConfigurationError('A gold label passed to Categorical Accuracy contains an id >= {}, the number of classes.'.format(num_classes))
        predictions = predictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()
        if not self._tie_break:
            if self._top_k == 1:
                top_k = predictions.max(-1)[1].unsqueeze(-1)
            else:
                (_, sorted_indices) = predictions.sort(dim=-1, descending=True)
                top_k = sorted_indices[..., :min(self._top_k, predictions.shape[-1])]
            correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
        else:
            max_predictions = predictions.max(-1)[0]
            max_predictions_mask = predictions.eq(max_predictions.unsqueeze(-1))
            correct = max_predictions_mask[torch.arange(gold_labels.numel(), device=gold_labels.device).long(), gold_labels].float()
            tie_counts = max_predictions_mask.sum(-1)
            correct /= tie_counts.float()
            correct.unsqueeze_(-1)
        if mask is not None:
            correct *= mask.view(-1, 1)
            _total_count = mask.sum()
        else:
            _total_count = torch.tensor(gold_labels.numel())
        _correct_count = correct.sum()
        self.correct_count += dist_reduce_sum(_correct_count).item()
        self.total_count += dist_reduce_sum(_total_count).item()

    def get_metric(self, reset: bool=False) -> float:
        if False:
            while True:
                i = 10
        '\n        # Returns\n\n        The accumulated accuracy.\n        '
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.correct_count = 0.0
        self.total_count = 0.0