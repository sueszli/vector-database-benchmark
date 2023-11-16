import pytest
import torch
from sklearn import metrics
from torch.testing import assert_allclose
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, multi_device, global_distributed_metric, run_distributed_test
from allennlp.training.metrics import Auc

class AucTest(AllenNlpTestCase):

    @multi_device
    def test_auc_computation(self, device: str):
        if False:
            i = 10
            return i + 15
        auc = Auc()
        all_predictions = []
        all_labels = []
        for _ in range(5):
            predictions = torch.randn(8, device=device)
            labels = torch.randint(0, 2, (8,), dtype=torch.long, device=device)
            auc(predictions, labels)
            all_predictions.append(predictions)
            all_labels.append(labels)
        computed_auc_value = auc.get_metric(reset=True)
        (false_positive_rates, true_positive_rates, _) = metrics.roc_curve(torch.cat(all_labels, dim=0).cpu().numpy(), torch.cat(all_predictions, dim=0).cpu().numpy())
        real_auc_value = metrics.auc(false_positive_rates, true_positive_rates)
        assert_allclose(real_auc_value, computed_auc_value)
        predictions = torch.randn(8, device=device)
        labels = torch.randint(0, 2, (8,), dtype=torch.long, device=device)
        auc(predictions, labels)
        computed_auc_value = auc.get_metric(reset=True)
        (false_positive_rates, true_positive_rates, _) = metrics.roc_curve(labels.cpu().numpy(), predictions.cpu().numpy())
        real_auc_value = metrics.auc(false_positive_rates, true_positive_rates)
        assert_allclose(real_auc_value, computed_auc_value)

    @multi_device
    def test_auc_gold_labels_behaviour(self, device: str):
        if False:
            i = 10
            return i + 15
        auc = Auc(positive_label=4)
        predictions = torch.randn(8, device=device)
        labels = torch.randint(3, 5, (8,), dtype=torch.long, device=device)
        labels[0] = 4
        auc(predictions, labels)
        computed_auc_value = auc.get_metric(reset=True)
        (false_positive_rates, true_positive_rates, _) = metrics.roc_curve(labels.cpu().numpy(), predictions.cpu().numpy(), pos_label=4)
        real_auc_value = metrics.auc(false_positive_rates, true_positive_rates)
        assert_allclose(real_auc_value, computed_auc_value)
        with pytest.raises(ConfigurationError) as _:
            labels = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10], device=device)
            auc(predictions, labels)

    @multi_device
    def test_auc_with_mask(self, device: str):
        if False:
            return 10
        auc = Auc()
        predictions = torch.randn(8, device=device)
        labels = torch.randint(0, 2, (8,), dtype=torch.long, device=device)
        mask = torch.tensor([True, True, True, True, False, False, False, False], device=device)
        auc(predictions, labels, mask)
        computed_auc_value = auc.get_metric(reset=True)
        (false_positive_rates, true_positive_rates, _) = metrics.roc_curve(labels[:4].cpu().numpy(), predictions[:4].cpu().numpy())
        real_auc_value = metrics.auc(false_positive_rates, true_positive_rates)
        assert_allclose(real_auc_value, computed_auc_value)

    @multi_device
    def test_auc_works_without_calling_metric_at_all(self, device: str):
        if False:
            while True:
                i = 10
        auc = Auc()
        auc.get_metric()

    def test_distributed_auc(self):
        if False:
            i = 10
            return i + 15
        predictions = torch.randn(8)
        labels = torch.randint(3, 5, (8,), dtype=torch.long)
        labels[0] = 4
        labels[4] = 4
        (false_positive_rates, true_positive_rates, _) = metrics.roc_curve(labels.cpu().numpy(), predictions.cpu().numpy(), pos_label=4)
        predictions = [predictions[:4], predictions[4:]]
        labels = [labels[:4], labels[4:]]
        metric_kwargs = {'predictions': predictions, 'gold_labels': labels}
        desired_auc = metrics.auc(false_positive_rates, true_positive_rates)
        run_distributed_test([-1, -1], global_distributed_metric, Auc(positive_label=4), metric_kwargs, desired_auc, exact=False)

    def test_distributed_auc_unequal_batches(self):
        if False:
            for i in range(10):
                print('nop')
        predictions = torch.randn(8)
        labels = torch.randint(3, 5, (8,), dtype=torch.long)
        labels[0] = 4
        labels[4] = 4
        (false_positive_rates, true_positive_rates, _) = metrics.roc_curve(labels.cpu().numpy(), predictions.cpu().numpy(), pos_label=4)
        predictions = [predictions[:2], predictions[2:]]
        labels = [labels[:2], labels[2:]]
        metric_kwargs = {'predictions': predictions, 'gold_labels': labels}
        desired_auc = metrics.auc(false_positive_rates, true_positive_rates)
        with pytest.raises(Exception) as _:
            run_distributed_test([-1, -1], global_distributed_metric, Auc(positive_label=4), metric_kwargs, desired_auc, exact=False)