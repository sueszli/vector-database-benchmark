"""Tests for open_spiel.python.pytorch.losses.rl_losses."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
from open_spiel.python.pytorch.losses import rl_losses
SEED = 24984617

class RLLossesTest(parameterized.TestCase, absltest.TestCase):

    @parameterized.named_parameters(('no_entropy_cost', 0.0), ('with_entropy_cost', 1.0))
    def test_batch_qpg_loss_with_entropy_cost(self, entropy_cost):
        if False:
            return 10
        batch_qpg_loss = rl_losses.BatchQPGLoss(entropy_cost=entropy_cost)
        q_values = torch.FloatTensor([[0.0, -1.0, 1.0], [1.0, -1.0, 0]])
        policy_logits = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 4.0]])
        total_loss = batch_qpg_loss.loss(policy_logits, q_values)
        expected_policy_entropy = (1.0986 + 0.3665) / 2
        expected_policy_loss = (0.0 + 0.0) / 2
        expected_total_loss = expected_policy_loss + entropy_cost * expected_policy_entropy
        np.testing.assert_allclose(total_loss, expected_total_loss, atol=0.0001)

    @parameterized.named_parameters(('no_entropy_cost', 0.0), ('with_entropy_cost', 1.0))
    def test_batch_rm_loss_with_entropy_cost(self, entropy_cost):
        if False:
            for i in range(10):
                print('nop')
        batch_rpg_loss = rl_losses.BatchRMLoss(entropy_cost=entropy_cost)
        q_values = torch.FloatTensor([[0.0, -1.0, 1.0], [1.0, -1.0, 0]])
        policy_logits = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 4.0]])
        total_loss = batch_rpg_loss.loss(policy_logits, q_values)
        expected_policy_entropy = (1.0986 + 0.3665) / 2
        expected_policy_loss = -(0.3333 + 0.0452) / 2
        expected_total_loss = expected_policy_loss + entropy_cost * expected_policy_entropy
        np.testing.assert_allclose(total_loss, expected_total_loss, atol=0.0001)

    @parameterized.named_parameters(('no_entropy_cost', 0.0), ('with_entropy_cost', 1.0))
    def test_batch_rpg_loss_with_entropy_cost(self, entropy_cost):
        if False:
            for i in range(10):
                print('nop')
        batch_rpg_loss = rl_losses.BatchRPGLoss(entropy_cost=entropy_cost)
        q_values = torch.FloatTensor([[0.0, -1.0, 1.0], [1.0, -1.0, 0]])
        policy_logits = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 4.0]])
        total_loss = batch_rpg_loss.loss(policy_logits, q_values)
        expected_policy_entropy = (1.0986 + 0.3665) / 2
        expected_policy_loss = (1.0 + 1.0) / 2
        expected_total_loss = expected_policy_loss + entropy_cost * expected_policy_entropy
        np.testing.assert_allclose(total_loss, expected_total_loss, atol=0.0001)

    @parameterized.named_parameters(('no_entropy_cost', 0.0), ('with_entropy_cost', 1.0))
    def test_batch_a2c_loss_with_entropy_cost(self, entropy_cost):
        if False:
            print('Hello World!')
        batch_a2c_loss = rl_losses.BatchA2CLoss(entropy_cost=entropy_cost)
        policy_logits = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 4.0]])
        baseline = torch.FloatTensor([1.0 / 3, 0.5])
        actions = torch.LongTensor([1, 2])
        returns = torch.FloatTensor([0.0, 1.0])
        total_loss = batch_a2c_loss.loss(policy_logits, baseline, actions, returns)
        expected_policy_entropy = (1.0986 + 0.3665) / 2
        expected_policy_loss = (-0.3662 + 0.04746) / 2
        expected_total_loss = expected_policy_loss + entropy_cost * expected_policy_entropy
        np.testing.assert_allclose(total_loss, expected_total_loss, atol=0.0001)
if __name__ == '__main__':
    torch.manual_seed(SEED)
    absltest.main()