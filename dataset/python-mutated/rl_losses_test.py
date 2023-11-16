"""Tests for open_spiel.python.algorithms.losses.rl_losses."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf
from open_spiel.python.algorithms.losses import rl_losses
tf.disable_v2_behavior()

class RLLossesTest(parameterized.TestCase, tf.test.TestCase):

    @parameterized.named_parameters(('no_entropy_cost', 0.0), ('with_entropy_cost', 1.0))
    def test_batch_qpg_loss_with_entropy_cost(self, entropy_cost):
        if False:
            return 10
        batch_qpg_loss = rl_losses.BatchQPGLoss(entropy_cost=entropy_cost)
        q_values = tf.constant([[0.0, -1.0, 1.0], [1.0, -1.0, 0]], dtype=tf.float32)
        policy_logits = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 4.0]], dtype=tf.float32)
        total_loss = batch_qpg_loss.loss(policy_logits, q_values)
        expected_policy_entropy_loss = -1 * (1.0986 + 0.3665) / 2
        expected_policy_loss = (0.0 + 0.0) / 2
        expected_total_loss = expected_policy_loss + entropy_cost * expected_policy_entropy_loss
        with self.session() as sess:
            np.testing.assert_allclose(sess.run(total_loss), expected_total_loss, atol=0.0001)

    @parameterized.named_parameters(('no_entropy_cost', 0.0), ('with_entropy_cost', 1.0))
    def test_batch_rm_loss_with_entropy_cost(self, entropy_cost):
        if False:
            i = 10
            return i + 15
        batch_rpg_loss = rl_losses.BatchRMLoss(entropy_cost=entropy_cost)
        q_values = tf.constant([[0.0, -1.0, 1.0], [1.0, -1.0, 0]], dtype=tf.float32)
        policy_logits = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 4.0]], dtype=tf.float32)
        total_loss = batch_rpg_loss.loss(policy_logits, q_values)
        expected_policy_entropy_loss = -(1.0986 + 0.3665) / 2
        expected_policy_loss = -(0.3333 + 0.0452) / 2
        expected_total_loss = expected_policy_loss + entropy_cost * expected_policy_entropy_loss
        with self.session() as sess:
            np.testing.assert_allclose(sess.run(total_loss), expected_total_loss, atol=0.001)

    @parameterized.named_parameters(('no_entropy_cost', 0.0), ('with_entropy_cost', 1.0))
    def test_batch_rpg_loss_with_entropy_cost(self, entropy_cost):
        if False:
            i = 10
            return i + 15
        batch_rpg_loss = rl_losses.BatchRPGLoss(entropy_cost=entropy_cost)
        q_values = tf.constant([[0.0, -1.0, 1.0], [1.0, -1.0, 0]], dtype=tf.float32)
        policy_logits = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 4.0]], dtype=tf.float32)
        total_loss = batch_rpg_loss.loss(policy_logits, q_values)
        expected_policy_entropy_loss = -1 * (1.0986 + 0.3665) / 2
        expected_policy_loss = (1.0 + 1.0) / 2
        expected_total_loss = expected_policy_loss + entropy_cost * expected_policy_entropy_loss
        with self.session() as sess:
            np.testing.assert_allclose(sess.run(total_loss), expected_total_loss, atol=0.0001)

    @parameterized.named_parameters(('no_entropy_cost', 0.0), ('with_entropy_cost', 1.0))
    def test_batch_a2c_loss_with_entropy_cost(self, entropy_cost):
        if False:
            while True:
                i = 10
        batch_a2c_loss = rl_losses.BatchA2CLoss(entropy_cost=entropy_cost)
        policy_logits = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 4.0]], dtype=tf.float32)
        baseline = tf.constant([1.0 / 3, 0.5], dtype=tf.float32)
        actions = tf.constant([1, 2], dtype=tf.int32)
        returns = tf.constant([0.0, 1.0], dtype=tf.float32)
        total_loss = batch_a2c_loss.loss(policy_logits, baseline, actions, returns)
        expected_policy_entropy_loss = -1 * (1.0986 + 0.3665) / 2
        expected_policy_loss = (-0.3662 + 0.04746) / 2
        expected_total_loss = expected_policy_loss + entropy_cost * expected_policy_entropy_loss
        with self.session() as sess:
            np.testing.assert_allclose(sess.run(total_loss), expected_total_loss, atol=0.0001)
if __name__ == '__main__':
    tf.test.main()