from typing import Any, Dict, List, Tuple, Union
import torch
from allennlp.common.testing import AllenNlpTestCase, multi_device, run_distributed_test, global_distributed_metric
from allennlp.training.metrics import AttachmentScores

class AttachmentScoresTest(AllenNlpTestCase):

    def setup_method(self):
        if False:
            print('Hello World!')
        super().setup_method()
        self.scorer = AttachmentScores()
        self.predictions = torch.Tensor([[0, 1, 3, 5, 2, 4], [0, 3, 2, 1, 0, 0]])
        self.gold_indices = torch.Tensor([[0, 1, 3, 5, 2, 4], [0, 3, 2, 1, 0, 0]])
        self.label_predictions = torch.Tensor([[0, 5, 2, 1, 4, 2], [0, 4, 8, 2, 0, 0]])
        self.gold_labels = torch.Tensor([[0, 5, 2, 1, 4, 2], [0, 4, 8, 2, 0, 0]])
        self.mask = torch.tensor([[True, True, True, True, True, True], [True, True, True, True, False, False]])

    def _send_tensors_to_device(self, device: str):
        if False:
            return 10
        self.predictions = self.predictions.to(device)
        self.gold_indices = self.gold_indices.to(device)
        self.label_predictions = self.label_predictions.to(device)
        self.gold_labels = self.gold_labels.to(device)
        self.mask = self.mask.to(device)

    @multi_device
    def test_perfect_scores(self, device: str):
        if False:
            return 10
        self._send_tensors_to_device(device)
        self.scorer(self.predictions, self.label_predictions, self.gold_indices, self.gold_labels, self.mask)
        for value in self.scorer.get_metric().values():
            assert value == 1.0

    @multi_device
    def test_unlabeled_accuracy_ignores_incorrect_labels(self, device: str):
        if False:
            for i in range(10):
                print('nop')
        self._send_tensors_to_device(device)
        label_predictions = self.label_predictions
        label_predictions[0, 3:] = 3
        label_predictions[1, 0] = 7
        self.scorer(self.predictions, label_predictions, self.gold_indices, self.gold_labels, self.mask)
        metrics = self.scorer.get_metric()
        assert metrics['UAS'] == 1.0
        assert metrics['UEM'] == 1.0
        assert metrics['LAS'] == 0.6
        assert metrics['LEM'] == 0.0

    @multi_device
    def test_labeled_accuracy_is_affected_by_incorrect_heads(self, device: str):
        if False:
            return 10
        self._send_tensors_to_device(device)
        predictions = self.predictions
        predictions[0, 3:] = 3
        predictions[1, 0] = 7
        predictions[1, 5] = 7
        self.scorer(predictions, self.label_predictions, self.gold_indices, self.gold_labels, self.mask)
        metrics = self.scorer.get_metric()
        assert metrics['UAS'] == 0.6
        assert metrics['LAS'] == 0.6
        assert metrics['LEM'] == 0.0
        assert metrics['UEM'] == 0.0

    @multi_device
    def test_attachment_scores_can_ignore_labels(self, device: str):
        if False:
            print('Hello World!')
        self._send_tensors_to_device(device)
        scorer = AttachmentScores(ignore_classes=[1])
        label_predictions = self.label_predictions
        label_predictions[0, 3] = 2
        scorer(self.predictions, label_predictions, self.gold_indices, self.gold_labels, self.mask)
        for value in scorer.get_metric().values():
            assert value == 1.0

    def test_distributed_attachment_scores(self):
        if False:
            i = 10
            return i + 15
        predictions = [torch.Tensor([[0, 1, 3, 5, 2, 4]]), torch.Tensor([[0, 3, 2, 1, 0, 0]])]
        gold_indices = [torch.Tensor([[0, 1, 3, 5, 2, 4]]), torch.Tensor([[0, 3, 2, 1, 0, 0]])]
        label_predictions = [torch.Tensor([[0, 5, 2, 3, 3, 3]]), torch.Tensor([[7, 4, 8, 2, 0, 0]])]
        gold_labels = [torch.Tensor([[0, 5, 2, 1, 4, 2]]), torch.Tensor([[0, 4, 8, 2, 0, 0]])]
        mask = [torch.tensor([[True, True, True, True, True, True]]), torch.tensor([[True, True, True, True, False, False]])]
        metric_kwargs = {'predicted_indices': predictions, 'gold_indices': gold_indices, 'predicted_labels': label_predictions, 'gold_labels': gold_labels, 'mask': mask}
        desired_metrics = {'UAS': 1.0, 'LAS': 0.6, 'UEM': 1.0, 'LEM': 0.0}
        run_distributed_test([-1, -1], global_distributed_metric, AttachmentScores(), metric_kwargs, desired_metrics, exact=True)

    def test_multiple_distributed_runs(self):
        if False:
            return 10
        predictions = [torch.Tensor([[0, 1, 3, 5, 2, 4]]), torch.Tensor([[0, 3, 2, 1, 0, 0]])]
        gold_indices = [torch.Tensor([[0, 1, 3, 5, 2, 4]]), torch.Tensor([[0, 3, 2, 1, 0, 0]])]
        label_predictions = [torch.Tensor([[0, 5, 2, 3, 3, 3]]), torch.Tensor([[7, 4, 8, 2, 0, 0]])]
        gold_labels = [torch.Tensor([[0, 5, 2, 1, 4, 2]]), torch.Tensor([[0, 4, 8, 2, 0, 0]])]
        mask = [torch.tensor([[True, True, True, True, True, True]]), torch.tensor([[True, True, True, True, False, False]])]
        metric_kwargs = {'predicted_indices': predictions, 'gold_indices': gold_indices, 'predicted_labels': label_predictions, 'gold_labels': gold_labels, 'mask': mask}
        desired_metrics = {'UAS': 1.0, 'LAS': 0.6, 'UEM': 1.0, 'LEM': 0.0}
        run_distributed_test([-1, -1], multiple_runs, AttachmentScores(), metric_kwargs, desired_metrics, exact=True)

def multiple_runs(global_rank: int, world_size: int, gpu_id: Union[int, torch.device], metric: AttachmentScores, metric_kwargs: Dict[str, List[Any]], desired_values: Dict[str, Any], exact: Union[bool, Tuple[float, float]]=True):
    if False:
        for i in range(10):
            print('nop')
    kwargs = {}
    for argname in metric_kwargs:
        kwargs[argname] = metric_kwargs[argname][global_rank]
    for i in range(200):
        metric(**kwargs)
    metrics = metric.get_metric()
    for key in metrics:
        assert desired_values[key] == metrics[key]