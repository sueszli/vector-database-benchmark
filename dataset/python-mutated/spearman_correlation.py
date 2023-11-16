from typing import Optional
import torch
import torch.distributed as dist
import scipy.stats as stats
from allennlp.common.util import is_distributed
from allennlp.training.metrics.metric import Metric

@Metric.register('spearman_correlation')
class SpearmanCorrelation(Metric):
    """
    This `Metric` calculates the sample Spearman correlation coefficient (r)
    between two tensors. Each element in the two tensors is assumed to be
    a different observation of the variable (i.e., the input tensors are
    implicitly flattened into vectors and the correlation is calculated
    between the vectors).

    <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>
    """

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.total_predictions = torch.zeros(0)
        self.total_gold_labels = torch.zeros(0)

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        # Parameters\n\n        predictions : `torch.Tensor`, required.\n            A tensor of predictions of shape (batch_size, ...).\n        gold_labels : `torch.Tensor`, required.\n            A tensor of the same shape as `predictions`.\n        mask : `torch.BoolTensor`, optional (default = `None`).\n            A tensor of the same shape as `predictions`.\n        '
        (predictions, gold_labels, mask) = self.detach_tensors(predictions, gold_labels, mask)
        predictions = predictions.reshape(-1)
        gold_labels = gold_labels.reshape(-1)
        self.total_predictions = self.total_predictions.to(predictions.device)
        self.total_gold_labels = self.total_gold_labels.to(gold_labels.device)
        if mask is not None:
            mask = mask.reshape(-1)
            self.total_predictions = torch.cat((self.total_predictions, predictions * mask), 0)
            self.total_gold_labels = torch.cat((self.total_gold_labels, gold_labels * mask), 0)
        else:
            self.total_predictions = torch.cat((self.total_predictions, predictions), 0)
            self.total_gold_labels = torch.cat((self.total_gold_labels, gold_labels), 0)
        if is_distributed():
            world_size = dist.get_world_size()
            device = gold_labels.device
            _all_batch_lengths = [torch.tensor(0) for i in range(world_size)]
            dist.all_gather(_all_batch_lengths, torch.tensor(self.total_predictions.shape[0], device=device))
            _all_batch_lengths = [batch_length.item() for batch_length in _all_batch_lengths]
            if len(set(_all_batch_lengths)) > 1:
                raise RuntimeError('Distributed aggregation for SpearmanCorrelation is currently not supported for batches of unequal length.')
            _total_predictions = [torch.zeros(self.total_predictions.shape, device=device) for i in range(world_size)]
            _total_gold_labels = [torch.zeros(self.total_gold_labels.shape, device=device) for i in range(world_size)]
            dist.all_gather(_total_predictions, self.total_predictions)
            dist.all_gather(_total_gold_labels, self.total_gold_labels)
            self.total_predictions = torch.cat(_total_predictions, dim=0)
            self.total_gold_labels = torch.cat(_total_gold_labels, dim=0)

    def get_metric(self, reset: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        # Returns\n\n        The accumulated sample Spearman correlation.\n        '
        spearman_correlation = stats.spearmanr(self.total_predictions.cpu().numpy(), self.total_gold_labels.cpu().numpy())
        if reset:
            self.reset()
        return spearman_correlation[0]

    def reset(self):
        if False:
            print('Hello World!')
        self.total_predictions = torch.zeros(0)
        self.total_gold_labels = torch.zeros(0)