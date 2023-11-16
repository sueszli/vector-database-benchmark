from typing import Optional, List
import torch
from allennlp.nn.util import dist_reduce_sum
from allennlp.training.metrics.metric import Metric

@Metric.register('attachment_scores')
class AttachmentScores(Metric):
    """
    Computes labeled and unlabeled attachment scores for a
    dependency parse, as well as sentence level exact match
    for both labeled and unlabeled trees. Note that the input
    to this metric is the sampled predictions, not the distribution
    itself.

    # Parameters

    ignore_classes : `List[int]`, optional (default = `None`)
        A list of label ids to ignore when computing metrics.
    """

    def __init__(self, ignore_classes: List[int]=None) -> None:
        if False:
            while True:
                i = 10
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0
        self._ignore_classes: List[int] = ignore_classes or []

    def __call__(self, predicted_indices: torch.Tensor, predicted_labels: torch.Tensor, gold_indices: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor]=None):
        if False:
            return 10
        '\n        # Parameters\n\n        predicted_indices : `torch.Tensor`, required.\n            A tensor of head index predictions of shape (batch_size, timesteps).\n        predicted_labels : `torch.Tensor`, required.\n            A tensor of arc label predictions of shape (batch_size, timesteps).\n        gold_indices : `torch.Tensor`, required.\n            A tensor of the same shape as `predicted_indices`.\n        gold_labels : `torch.Tensor`, required.\n            A tensor of the same shape as `predicted_labels`.\n        mask : `torch.BoolTensor`, optional (default = `None`).\n            A tensor of the same shape as `predicted_indices`.\n        '
        detached = self.detach_tensors(predicted_indices, predicted_labels, gold_indices, gold_labels, mask)
        (predicted_indices, predicted_labels, gold_indices, gold_labels, mask) = detached
        if mask is None:
            mask = torch.ones_like(predicted_indices).bool()
        predicted_indices = predicted_indices.long()
        predicted_labels = predicted_labels.long()
        gold_indices = gold_indices.long()
        gold_labels = gold_labels.long()
        for label in self._ignore_classes:
            label_mask = gold_labels.eq(label)
            mask = mask & ~label_mask
        correct_indices = predicted_indices.eq(gold_indices).long() * mask
        unlabeled_exact_match = (correct_indices + ~mask).prod(dim=-1)
        correct_labels = predicted_labels.eq(gold_labels).long() * mask
        correct_labels_and_indices = correct_indices * correct_labels
        labeled_exact_match = (correct_labels_and_indices + ~mask).prod(dim=-1)
        total_sentences = correct_indices.size(0)
        total_words = correct_indices.numel() - (~mask).sum()
        self._unlabeled_correct += dist_reduce_sum(correct_indices).sum()
        self._exact_unlabeled_correct += dist_reduce_sum(unlabeled_exact_match).sum()
        self._labeled_correct += dist_reduce_sum(correct_labels_and_indices).sum()
        self._exact_labeled_correct += dist_reduce_sum(labeled_exact_match).sum()
        self._total_sentences += dist_reduce_sum(total_sentences)
        self._total_words += dist_reduce_sum(total_words)

    def get_metric(self, reset: bool=False):
        if False:
            while True:
                i = 10
        '\n        # Returns\n\n        The accumulated metrics as a dictionary.\n        '
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        unlabeled_exact_match = 0.0
        labeled_exact_match = 0.0
        if self._total_words > 0.0:
            unlabeled_attachment_score = float(self._unlabeled_correct) / float(self._total_words)
            labeled_attachment_score = float(self._labeled_correct) / float(self._total_words)
        if self._total_sentences > 0:
            unlabeled_exact_match = float(self._exact_unlabeled_correct) / float(self._total_sentences)
            labeled_exact_match = float(self._exact_labeled_correct) / float(self._total_sentences)
        if reset:
            self.reset()
        metrics = {'UAS': unlabeled_attachment_score, 'LAS': labeled_attachment_score, 'UEM': unlabeled_exact_match, 'LEM': labeled_exact_match}
        return metrics

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0