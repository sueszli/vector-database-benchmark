import pytest
import torch
import math
import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, multi_device, global_distributed_metric, run_distributed_test
from allennlp.fairness.fairness_metrics import Independence, Separation, Sufficiency

class IndependenceTest(AllenNlpTestCase):

    def test_invalid_dimensions(self):
        if False:
            i = 10
            return i + 15
        independence = Independence(2, 2)
        C = torch.eye(3).long()
        A = torch.eye(4).long()
        with pytest.raises(ConfigurationError):
            independence(C, A)

    def test_invalid_num_classes(self):
        if False:
            while True:
                i = 10
        independence = Independence(1, 1)
        C = torch.eye(3).long()
        A = torch.eye(3).long()
        with pytest.raises(ConfigurationError):
            independence(C, A)

    @multi_device
    def test_independence_unmasked_computation(self, device: str):
        if False:
            print('Hello World!')
        independence = Independence(4, 2)
        A = torch.eye(3, device=device).long()
        C = 2 * A
        expected_kl_divs = {0: 0.4055, 1: 1.0986}
        independence(C, A)
        test_kl_divs = {k: v.item() for (k, v) in independence.get_metric().items()}
        assert expected_kl_divs == pytest.approx(test_kl_divs, abs=0.001)
        independence(C, A)
        test_kl_divs = {k: v.item() for (k, v) in independence.get_metric(reset=True).items()}
        assert expected_kl_divs == pytest.approx(test_kl_divs, abs=0.001)
        test_kl_divs = {k: v.item() if not math.isnan(v.item()) else np.nan for (k, v) in independence.get_metric().items()}
        assert test_kl_divs == {0: np.nan, 1: np.nan}

    def test_independence_with_wasserstein_distance(self):
        if False:
            i = 10
            return i + 15
        independence = Independence(4, 2, 'wasserstein')
        A = torch.eye(3).long()
        C = 2 * A
        expected_distances = {0: 0.6667, 1: 1.3333}
        independence(C, A)
        test_distances = {k: v.item() for (k, v) in independence.get_metric(reset=True).items()}
        assert expected_distances == pytest.approx(test_distances, abs=0.001)

    def test_distributed_independence_masked_computation(self):
        if False:
            for i in range(10):
                print('nop')
        A = torch.eye(3).long()
        C = 2 * A
        mask = torch.ones_like(C).bool()
        expected_kl_divs = {0: 0.4055, 1: 1.0986}
        metric_kwargs = {'predicted_labels': C, 'protected_variable_labels': A, 'mask': mask}
        run_distributed_test([-1, -1], global_distributed_metric, Independence(4, 2), metric_kwargs, expected_kl_divs, exact=False)

class SeparationTest(AllenNlpTestCase):

    def test_invalid_dimensions(self):
        if False:
            for i in range(10):
                print('nop')
        separation = Separation(2, 2)
        C = torch.eye(3).long()
        Y = torch.eye(4).long()
        A = torch.eye(3).long()
        with pytest.raises(ConfigurationError):
            separation(C, Y, A)

    def test_invalid_num_classes(self):
        if False:
            return 10
        separation = Separation(2, 2)
        C = 2 * torch.eye(3).long()
        Y = torch.eye(3).long()
        A = torch.eye(3).long()
        with pytest.raises(ConfigurationError):
            separation(C, Y, A)

    @multi_device
    def test_separation_unmasked_computation(self, device: str):
        if False:
            print('Hello World!')
        separation = Separation(2, 2)
        C = torch.eye(3, device=device).long()
        Y = C
        A = C
        expected_kl_divs = {0: {0: 0.0, 1: np.nan}, 1: {0: np.nan, 1: 0.0}}
        separation(C, Y, A)
        test_kl_divs = {k1: {k2: v2.item() if not math.isnan(v2.item()) else np.nan for (k2, v2) in v1.items()} for (k1, v1) in separation.get_metric().items()}
        assert expected_kl_divs == test_kl_divs
        separation(C, Y, A)
        test_kl_divs = {k1: {k2: v2.item() if not math.isnan(v2.item()) else np.nan for (k2, v2) in v1.items()} for (k1, v1) in separation.get_metric(reset=True).items()}
        assert expected_kl_divs == test_kl_divs
        test_kl_divs = {k1: {k2: v2.item() if not math.isnan(v2.item()) else np.nan for (k2, v2) in v1.items()} for (k1, v1) in separation.get_metric().items()}
        assert test_kl_divs == {0: {0: np.nan, 1: np.nan}, 1: {0: np.nan, 1: np.nan}}

    def test_distributed_separation_masked_computation(self):
        if False:
            for i in range(10):
                print('nop')
        C = torch.eye(3).long()
        Y = C
        A = C
        mask = torch.ones_like(C).bool()
        expected_kl_divs = {0: {0: 0.0, 1: np.nan}, 1: {0: np.nan, 1: 0.0}}
        metric_kwargs = {'predicted_labels': C, 'gold_labels': Y, 'protected_variable_labels': A, 'mask': mask}
        run_distributed_test([-1, -1], global_distributed_metric, Separation(2, 2), metric_kwargs, expected_kl_divs, exact=True)

class SufficiencyTest(AllenNlpTestCase):

    def test_invalid_dimensions(self):
        if False:
            while True:
                i = 10
        sufficiency = Sufficiency(2, 2)
        C = torch.eye(3).long()
        Y = torch.eye(4).long()
        A = torch.eye(3).long()
        with pytest.raises(ConfigurationError):
            sufficiency(C, Y, A)

    def test_invalid_num_classes(self):
        if False:
            while True:
                i = 10
        sufficiency = Sufficiency(2, 2)
        C = 2 * torch.eye(3).long()
        Y = torch.eye(3).long()
        A = torch.eye(3).long()
        with pytest.raises(ConfigurationError):
            sufficiency(C, Y, A)

    @multi_device
    def test_sufficiency_unmasked_computation(self, device: str):
        if False:
            return 10
        sufficiency = Sufficiency(2, 2)
        C = torch.zeros(3, 3, device=device).long()
        Y = torch.eye(3, device=device).long()
        A = Y
        expected_kl_divs = {0: {0: 0.4055, 1: 1.0986}, 1: {0: np.nan, 1: np.nan}}
        sufficiency(C, Y, A)
        test_kl_divs = {k1: {k2: v2.item() if not math.isnan(v2.item()) else np.nan for (k2, v2) in v1.items()} for (k1, v1) in sufficiency.get_metric().items()}
        assert len(expected_kl_divs) == len(test_kl_divs)
        assert expected_kl_divs[0] == pytest.approx(test_kl_divs[0], abs=0.001)
        assert expected_kl_divs[1] == test_kl_divs[1]
        sufficiency(C, Y, A)
        test_kl_divs = {k1: {k2: v2.item() if not math.isnan(v2.item()) else np.nan for (k2, v2) in v1.items()} for (k1, v1) in sufficiency.get_metric(reset=True).items()}
        assert len(expected_kl_divs) == len(test_kl_divs)
        assert expected_kl_divs[0] == pytest.approx(test_kl_divs[0], abs=0.001)
        assert expected_kl_divs[1] == test_kl_divs[1]
        test_kl_divs = {k1: {k2: v2.item() if not math.isnan(v2.item()) else np.nan for (k2, v2) in v1.items()} for (k1, v1) in sufficiency.get_metric().items()}
        assert len(expected_kl_divs) == len(test_kl_divs)
        assert test_kl_divs == {0: {0: np.nan, 1: np.nan}, 1: {0: np.nan, 1: np.nan}}

    def test_distributed_sufficiency_masked_computation(self):
        if False:
            print('Hello World!')
        C = torch.zeros(3, 3).long()
        Y = torch.eye(3).long()
        A = Y
        mask = torch.ones_like(C).bool()
        expected_kl_divs = {0: {0: 0.4055, 1: 1.0986}, 1: {0: np.nan, 1: np.nan}}
        metric_kwargs = {'predicted_labels': C, 'gold_labels': Y, 'protected_variable_labels': A, 'mask': mask}
        run_distributed_test([-1, -1], global_distributed_metric, Sufficiency(2, 2), metric_kwargs, expected_kl_divs, exact=False)