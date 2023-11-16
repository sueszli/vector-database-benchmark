"""Objectives for full-episode.

Implementations of UREX & REINFORCE.  Note that these implementations
use a non-parametric baseline to reduce variance.  Thus, multiple
samples with the same seed must be taken from the environment.

"""
import tensorflow as tf
import objective

class Reinforce(objective.Objective):

    def __init__(self, learning_rate, clip_norm, num_samples, tau=0.1, bonus_weight=1.0):
        if False:
            print('Hello World!')
        super(Reinforce, self).__init__(learning_rate, clip_norm=clip_norm)
        self.num_samples = num_samples
        assert self.num_samples > 1
        self.tau = tau
        self.bonus_weight = bonus_weight
        self.eps_lambda = 0.0

    def get_bonus(self, total_rewards, total_log_probs):
        if False:
            while True:
                i = 10
        'Exploration bonus.'
        return -self.tau * total_log_probs

    def get(self, rewards, pads, values, final_values, log_probs, prev_log_probs, target_log_probs, entropies, logits, target_values, final_target_values):
        if False:
            print('Hello World!')
        seq_length = tf.shape(rewards)[0]
        not_pad = tf.reshape(1 - pads, [seq_length, -1, self.num_samples])
        rewards = not_pad * tf.reshape(rewards, [seq_length, -1, self.num_samples])
        log_probs = not_pad * tf.reshape(sum(log_probs), [seq_length, -1, self.num_samples])
        total_rewards = tf.reduce_sum(rewards, 0)
        total_log_probs = tf.reduce_sum(log_probs, 0)
        rewards_and_bonus = total_rewards + self.bonus_weight * self.get_bonus(total_rewards, total_log_probs)
        baseline = tf.reduce_mean(rewards_and_bonus, 1, keep_dims=True)
        loss = -tf.stop_gradient(rewards_and_bonus - baseline) * total_log_probs
        loss = tf.reduce_mean(loss)
        raw_loss = loss
        gradient_ops = self.training_ops(loss, learning_rate=self.learning_rate)
        tf.summary.histogram('log_probs', total_log_probs)
        tf.summary.histogram('rewards', total_rewards)
        tf.summary.scalar('avg_rewards', tf.reduce_mean(total_rewards))
        tf.summary.scalar('loss', loss)
        return (loss, raw_loss, baseline, gradient_ops, tf.summary.merge_all())

class UREX(Reinforce):

    def get_bonus(self, total_rewards, total_log_probs):
        if False:
            i = 10
            return i + 15
        'Exploration bonus.'
        discrepancy = total_rewards / self.tau - total_log_probs
        normalized_d = self.num_samples * tf.nn.softmax(discrepancy)
        return self.tau * normalized_d