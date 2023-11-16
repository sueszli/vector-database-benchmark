from typing import Tuple
import torch
import numpy as np
from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.sampled_softmax_loss import _choice, SampledSoftmaxLoss
from allennlp.modules import SoftmaxLoss

class TestSampledSoftmaxLoss(AllenNlpTestCase):

    def test_choice(self):
        if False:
            i = 10
            return i + 15
        (sample, num_tries) = _choice(num_words=1000, num_samples=50)
        assert len(set(sample)) == 50
        assert all((0 <= x < 1000 for x in sample))
        assert num_tries >= 50

    def test_sampled_softmax_can_run(self):
        if False:
            i = 10
            return i + 15
        softmax = SampledSoftmaxLoss(num_words=1000, embedding_dim=12, num_samples=50)
        embedding = torch.rand(100, 12)
        targets = torch.randint(0, 1000, (100,)).long()
        _ = softmax(embedding, targets)

    def test_sampled_equals_unsampled_during_eval(self):
        if False:
            return 10
        sampled_softmax = SampledSoftmaxLoss(num_words=10000, embedding_dim=12, num_samples=40)
        unsampled_softmax = SoftmaxLoss(num_words=10000, embedding_dim=12)
        sampled_softmax.eval()
        unsampled_softmax.eval()
        sampled_softmax.softmax_w.data = unsampled_softmax.softmax_w.t()
        sampled_softmax.softmax_b.data = unsampled_softmax.softmax_b
        embedding = torch.rand(100, 12)
        targets = torch.randint(0, 1000, (100,)).long()
        full_loss = unsampled_softmax(embedding, targets).item()
        sampled_loss = sampled_softmax(embedding, targets).item()
        np.testing.assert_almost_equal(sampled_loss, full_loss)

    def test_sampled_softmax_has_greater_loss_in_train_mode(self):
        if False:
            while True:
                i = 10
        sampled_softmax = SampledSoftmaxLoss(num_words=10000, embedding_dim=12, num_samples=10)
        embedding = torch.rand(100, 12)
        targets = torch.randint(0, 1000, (100,)).long()
        sampled_softmax.train()
        train_loss = sampled_softmax(embedding, targets).item()
        sampled_softmax.eval()
        eval_loss = sampled_softmax(embedding, targets).item()
        assert eval_loss > train_loss

    def test_sampled_equals_unsampled_when_biased_against_non_sampled_positions(self):
        if False:
            print('Hello World!')
        sampled_softmax = SampledSoftmaxLoss(num_words=10000, embedding_dim=12, num_samples=10)
        unsampled_softmax = SoftmaxLoss(num_words=10000, embedding_dim=12)
        FAKE_SAMPLES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 9999]

        def fake_choice(num_words: int, num_samples: int) -> Tuple[np.ndarray, int]:
            if False:
                return 10
            assert (num_words, num_samples) == (10000, 10)
            return (np.array(FAKE_SAMPLES), 12)
        sampled_softmax.choice_func = fake_choice
        for i in range(10000):
            if i not in FAKE_SAMPLES:
                unsampled_softmax.softmax_b.data[i] = -10000
        sampled_softmax.softmax_w.data = unsampled_softmax.softmax_w.t()
        sampled_softmax.softmax_b.data = unsampled_softmax.softmax_b
        sampled_softmax.train()
        unsampled_softmax.train()
        embedding = torch.rand(100, 12)
        targets = torch.randint(0, 1000, (100,)).long()
        full_loss = unsampled_softmax(embedding, targets).item()
        sampled_loss = sampled_softmax(embedding, targets).item()
        pct_error = (sampled_loss - full_loss) / full_loss
        assert abs(pct_error) < 0.0003