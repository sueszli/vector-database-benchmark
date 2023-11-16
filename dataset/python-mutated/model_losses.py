"""Model loss construction."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import xrange
import tensorflow as tf
from losses import losses
FLAGS = tf.app.flags.FLAGS

def create_dis_loss(fake_predictions, real_predictions, targets_present):
    if False:
        return 10
    'Compute Discriminator loss across real/fake.'
    missing = tf.cast(targets_present, tf.int32)
    missing = 1 - missing
    missing = tf.cast(missing, tf.bool)
    real_labels = tf.ones([FLAGS.batch_size, FLAGS.sequence_length])
    dis_loss_real = tf.losses.sigmoid_cross_entropy(real_labels, real_predictions, weights=missing)
    dis_loss_fake = tf.losses.sigmoid_cross_entropy(targets_present, fake_predictions, weights=missing)
    dis_loss = (dis_loss_fake + dis_loss_real) / 2.0
    return (dis_loss, dis_loss_fake, dis_loss_real)

def create_critic_loss(cumulative_rewards, estimated_values, present):
    if False:
        for i in range(10):
            print('nop')
    'Compute Critic loss in estimating the value function.  This should be an\n  estimate only for the missing elements.'
    missing = tf.cast(present, tf.int32)
    missing = 1 - missing
    missing = tf.cast(missing, tf.bool)
    loss = tf.losses.mean_squared_error(labels=cumulative_rewards, predictions=estimated_values, weights=missing)
    return loss

def create_masked_cross_entropy_loss(targets, present, logits):
    if False:
        i = 10
        return i + 15
    'Calculate the cross entropy loss matrices for the masked tokens.'
    cross_entropy_losses = losses.cross_entropy_loss_matrix(targets, logits)
    zeros_losses = tf.zeros(shape=[FLAGS.batch_size, FLAGS.sequence_length], dtype=tf.float32)
    missing_ce_loss = tf.where(present, zeros_losses, cross_entropy_losses)
    return missing_ce_loss

def calculate_reinforce_objective(hparams, log_probs, dis_predictions, present, estimated_values=None):
    if False:
        while True:
            i = 10
    'Calculate the REINFORCE objectives.  The REINFORCE objective should\n  only be on the tokens that were missing.  Specifically, the final Generator\n  reward should be based on the Discriminator predictions on missing tokens.\n  The log probaibilities should be only for missing tokens and the baseline\n  should be calculated only on the missing tokens.\n\n  For this model, we optimize the reward is the log of the *conditional*\n  probability the Discriminator assigns to the distribution.  Specifically, for\n  a Discriminator D which outputs probability of real, given the past context,\n\n    r_t = log D(x_t|x_0,x_1,...x_{t-1})\n\n  And the policy for Generator G is the log-probability of taking action x2\n  given the past context.\n\n\n  Args:\n    hparams:  MaskGAN hyperparameters.\n    log_probs:  tf.float32 Tensor of log probailities of the tokens selected by\n      the Generator.  Shape [batch_size, sequence_length].\n    dis_predictions:  tf.float32 Tensor of the predictions from the\n      Discriminator.  Shape [batch_size, sequence_length].\n    present:  tf.bool Tensor indicating which tokens are present.  Shape\n      [batch_size, sequence_length].\n    estimated_values:  tf.float32 Tensor of estimated state values of tokens.\n      Shape [batch_size, sequence_length]\n\n  Returns:\n    final_gen_objective:  Final REINFORCE objective for the sequence.\n    rewards:  tf.float32 Tensor of rewards for sequence of shape [batch_size,\n      sequence_length]\n    advantages: tf.float32 Tensor of advantages for sequence of shape\n      [batch_size, sequence_length]\n    baselines:  tf.float32 Tensor of baselines for sequence of shape\n      [batch_size, sequence_length]\n    maintain_averages_op:  ExponentialMovingAverage apply average op to\n      maintain the baseline.\n  '
    final_gen_objective = 0.0
    gamma = hparams.rl_discount_rate
    eps = 1e-07
    eps = tf.constant(1e-07, tf.float32)
    dis_predictions = tf.nn.sigmoid(dis_predictions)
    rewards = tf.log(dis_predictions + eps)
    zeros = tf.zeros_like(present, dtype=tf.float32)
    log_probs = tf.where(present, zeros, log_probs)
    rewards = tf.where(present, zeros, rewards)
    rewards_list = tf.unstack(rewards, axis=1)
    log_probs_list = tf.unstack(log_probs, axis=1)
    missing = 1.0 - tf.cast(present, tf.float32)
    missing_list = tf.unstack(missing, axis=1)
    cumulative_rewards = []
    for t in xrange(FLAGS.sequence_length):
        cum_value = tf.zeros(shape=[FLAGS.batch_size])
        for s in xrange(t, FLAGS.sequence_length):
            cum_value += missing_list[s] * np.power(gamma, s - t) * rewards_list[s]
        cumulative_rewards.append(cum_value)
    cumulative_rewards = tf.stack(cumulative_rewards, axis=1)
    if FLAGS.baseline_method == 'critic':
        critic_loss = create_critic_loss(cumulative_rewards, estimated_values, present)
        baselines = tf.unstack(estimated_values, axis=1)
        advantages = []
        for t in xrange(FLAGS.sequence_length):
            log_probability = log_probs_list[t]
            cum_advantage = tf.zeros(shape=[FLAGS.batch_size])
            for s in xrange(t, FLAGS.sequence_length):
                cum_advantage += missing_list[s] * np.power(gamma, s - t) * rewards_list[s]
            cum_advantage -= baselines[t]
            cum_advantage = tf.clip_by_value(cum_advantage, -FLAGS.advantage_clipping, FLAGS.advantage_clipping)
            advantages.append(missing_list[t] * cum_advantage)
            final_gen_objective += tf.multiply(log_probability, missing_list[t] * tf.stop_gradient(cum_advantage))
        maintain_averages_op = None
        baselines = tf.stack(baselines, axis=1)
        advantages = tf.stack(advantages, axis=1)
    elif FLAGS.baseline_method == 'dis_batch':
        [rewards_half, baseline_half] = tf.split(rewards, num_or_size_splits=2, axis=0)
        [log_probs_half, _] = tf.split(log_probs, num_or_size_splits=2, axis=0)
        [reward_present_half, baseline_present_half] = tf.split(present, num_or_size_splits=2, axis=0)
        baseline_list = tf.unstack(baseline_half, axis=1)
        baseline_missing = 1.0 - tf.cast(baseline_present_half, tf.float32)
        baseline_missing_list = tf.unstack(baseline_missing, axis=1)
        baselines = []
        for t in xrange(FLAGS.sequence_length):
            num_missing = tf.reduce_sum(baseline_missing_list[t])
            avg_baseline = tf.reduce_sum(baseline_missing_list[t] * baseline_list[t], keep_dims=True) / (num_missing + eps)
            baseline = tf.tile(avg_baseline, multiples=[FLAGS.batch_size / 2])
            baselines.append(baseline)
        rewards_list = tf.unstack(rewards_half, axis=1)
        log_probs_list = tf.unstack(log_probs_half, axis=1)
        reward_missing = 1.0 - tf.cast(reward_present_half, tf.float32)
        reward_missing_list = tf.unstack(reward_missing, axis=1)
        advantages = []
        for t in xrange(FLAGS.sequence_length):
            log_probability = log_probs_list[t]
            cum_advantage = tf.zeros(shape=[FLAGS.batch_size / 2])
            for s in xrange(t, FLAGS.sequence_length):
                cum_advantage += reward_missing_list[s] * np.power(gamma, s - t) * (rewards_list[s] - baselines[s])
            cum_advantage = tf.clip_by_value(cum_advantage, -FLAGS.advantage_clipping, FLAGS.advantage_clipping)
            advantages.append(reward_missing_list[t] * cum_advantage)
            final_gen_objective += tf.multiply(log_probability, reward_missing_list[t] * tf.stop_gradient(cum_advantage))
        cumulative_rewards = []
        for t in xrange(FLAGS.sequence_length):
            cum_value = tf.zeros(shape=[FLAGS.batch_size / 2])
            for s in xrange(t, FLAGS.sequence_length):
                cum_value += reward_missing_list[s] * np.power(gamma, s - t) * rewards_list[s]
            cumulative_rewards.append(cum_value)
        cumulative_rewards = tf.stack(cumulative_rewards, axis=1)
        rewards = rewards_half
        critic_loss = None
        maintain_averages_op = None
        baselines = tf.stack(baselines, axis=1)
        advantages = tf.stack(advantages, axis=1)
    elif FLAGS.baseline_method == 'ema':
        ema = tf.train.ExponentialMovingAverage(decay=hparams.baseline_decay)
        maintain_averages_op = ema.apply(rewards_list)
        baselines = []
        for r in rewards_list:
            baselines.append(ema.average(r))
        advantages = []
        for t in xrange(FLAGS.sequence_length):
            log_probability = log_probs_list[t]
            cum_advantage = tf.zeros(shape=[FLAGS.batch_size])
            for s in xrange(t, FLAGS.sequence_length):
                cum_advantage += missing_list[s] * np.power(gamma, s - t) * (rewards_list[s] - baselines[s])
            cum_advantage = tf.clip_by_value(cum_advantage, -FLAGS.advantage_clipping, FLAGS.advantage_clipping)
            advantages.append(missing_list[t] * cum_advantage)
            final_gen_objective += tf.multiply(log_probability, missing_list[t] * tf.stop_gradient(cum_advantage))
        critic_loss = None
        baselines = tf.stack(baselines, axis=1)
        advantages = tf.stack(advantages, axis=1)
    elif FLAGS.baseline_method is None:
        num_missing = tf.reduce_sum(missing)
        final_gen_objective += tf.reduce_sum(rewards) / (num_missing + eps)
        baselines = tf.zeros_like(rewards)
        critic_loss = None
        maintain_averages_op = None
        advantages = cumulative_rewards
    else:
        raise NotImplementedError
    return [final_gen_objective, log_probs, rewards, advantages, baselines, maintain_averages_op, critic_loss, cumulative_rewards]

def calculate_log_perplexity(logits, targets, present):
    if False:
        while True:
            i = 10
    'Calculate the average log perplexity per *missing* token.\n\n  Args:\n    logits:  tf.float32 Tensor of the logits of shape [batch_size,\n      sequence_length, vocab_size].\n    targets:  tf.int32 Tensor of the sequence target of shape [batch_size,\n      sequence_length].\n    present:  tf.bool Tensor indicating the presence or absence of the token\n      of shape [batch_size, sequence_length].\n\n  Returns:\n    avg_log_perplexity:  Scalar indicating the average log perplexity per\n      missing token in the batch.\n  '
    eps = 1e-12
    logits = tf.reshape(logits, [-1, FLAGS.vocab_size])
    weights = tf.cast(present, tf.float32)
    weights = 1.0 - weights
    weights = tf.reshape(weights, [-1])
    num_missing = tf.reduce_sum(weights)
    log_perplexity = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(targets, [-1])], [weights])
    avg_log_perplexity = tf.reduce_sum(log_perplexity) / (num_missing + eps)
    return avg_log_perplexity