import numpy as np
import torch
from torch.testing import assert_allclose
from allennlp.common.testing import AllenNlpTestCase, multi_device
from allennlp.training.metrics import Covariance

class CovarianceTest(AllenNlpTestCase):

    @multi_device
    def test_covariance_unmasked_computation(self, device: str):
        if False:
            for i in range(10):
                print('nop')
        covariance = Covariance()
        batch_size = 100
        num_labels = 10
        predictions = torch.randn(batch_size, num_labels, device=device)
        labels = 0.5 * predictions + torch.randn(batch_size, num_labels, device=device)
        stride = 10
        for i in range(batch_size // stride):
            timestep_predictions = predictions[stride * i:stride * (i + 1), :]
            timestep_labels = labels[stride * i:stride * (i + 1), :]
            expected_covariance = np.cov(predictions[:stride * (i + 1), :].view(-1).cpu().numpy(), labels[:stride * (i + 1), :].view(-1).cpu().numpy())[0, 1]
            covariance(timestep_predictions, timestep_labels)
            assert_allclose(expected_covariance, covariance.get_metric())
        covariance.reset()
        covariance(predictions, labels)
        assert_allclose(np.cov(predictions.view(-1).cpu().numpy(), labels.view(-1).cpu().numpy())[0, 1], covariance.get_metric())

    @multi_device
    def test_covariance_masked_computation(self, device: str):
        if False:
            for i in range(10):
                print('nop')
        covariance = Covariance()
        batch_size = 100
        num_labels = 10
        predictions = torch.randn(batch_size, num_labels, device=device)
        labels = 0.5 * predictions + torch.randn(batch_size, num_labels, device=device)
        mask = torch.randint(0, 2, size=(batch_size, num_labels), device=device).bool()
        stride = 10
        for i in range(batch_size // stride):
            timestep_predictions = predictions[stride * i:stride * (i + 1), :]
            timestep_labels = labels[stride * i:stride * (i + 1), :]
            timestep_mask = mask[stride * i:stride * (i + 1), :]
            expected_covariance = np.cov(predictions[:stride * (i + 1), :].view(-1).cpu().numpy(), labels[:stride * (i + 1), :].view(-1).cpu().numpy(), fweights=mask[:stride * (i + 1), :].view(-1).cpu().numpy())[0, 1]
            covariance(timestep_predictions, timestep_labels, timestep_mask)
            assert_allclose(expected_covariance, covariance.get_metric())
        covariance.reset()
        covariance(predictions, labels, mask)
        assert_allclose(np.cov(predictions.view(-1).cpu().numpy(), labels.view(-1).cpu().numpy(), fweights=mask.view(-1).cpu().numpy())[0, 1], covariance.get_metric())