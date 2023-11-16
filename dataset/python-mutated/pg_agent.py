from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'Language model agent.\n\nAgent outputs code in a sequence just like a language model. Can be trained\nas a language model or using RL, or a combination of the two.\n'
from collections import namedtuple
from math import exp
from math import log
import time
from absl import logging
import numpy as np
from six.moves import xrange
import tensorflow as tf
from common import rollout as rollout_lib
from common import utils
from single_task import misc
MAGIC_LOSS_MULTIPLIER = 64

def rshift_time(tensor_2d, fill=misc.BF_EOS_INT):
    if False:
        i = 10
        return i + 15
    'Right shifts a 2D tensor along the time dimension (axis-1).'
    dim_0 = tf.shape(tensor_2d)[0]
    fill_tensor = tf.fill([dim_0, 1], fill)
    return tf.concat([fill_tensor, tensor_2d[:, :-1]], axis=1)

def join(a, b):
    if False:
        for i in range(10):
            print('nop')
    if a is None or len(a) == 0:
        return b
    if b is None or len(b) == 0:
        return a
    return np.concatenate((a, b))

def make_optimizer(kind, lr):
    if False:
        while True:
            i = 10
    if kind == 'sgd':
        return tf.train.GradientDescentOptimizer(lr)
    elif kind == 'adam':
        return tf.train.AdamOptimizer(lr)
    elif kind == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99)
    else:
        raise ValueError('Optimizer type "%s" not recognized.' % kind)

class LinearWrapper(tf.contrib.rnn.RNNCell):
    """RNNCell wrapper that adds a linear layer to the output."""

    def __init__(self, cell, output_size, dtype=tf.float32, suppress_index=None):
        if False:
            while True:
                i = 10
        self.cell = cell
        self._output_size = output_size
        self._dtype = dtype
        self._suppress_index = suppress_index
        self.smallest_float = -2.4e+38

    def __call__(self, inputs, state, scope=None):
        if False:
            return 10
        with tf.variable_scope(type(self).__name__):
            (outputs, state) = self.cell(inputs, state, scope=scope)
            logits = tf.matmul(outputs, tf.get_variable('w_output', [self.cell.output_size, self.output_size], dtype=self._dtype))
            if self._suppress_index is not None:
                batch_size = tf.shape(logits)[0]
                logits = tf.concat([logits[:, :self._suppress_index], tf.fill([batch_size, 1], self.smallest_float), logits[:, self._suppress_index + 1:]], axis=1)
        return (logits, state)

    @property
    def output_size(self):
        if False:
            print('Hello World!')
        return self._output_size

    @property
    def state_size(self):
        if False:
            return 10
        return self.cell.state_size

    def zero_state(self, batch_size, dtype):
        if False:
            while True:
                i = 10
        return self.cell.zero_state(batch_size, dtype)
UpdateStepResult = namedtuple('UpdateStepResult', ['global_step', 'global_npe', 'summaries_list', 'gradients_dict'])

class AttrDict(dict):
    """Dict with attributes as keys.

  https://stackoverflow.com/a/14620633
  """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class LMAgent(object):
    """Language model agent."""
    action_space = misc.bf_num_tokens()
    observation_space = misc.bf_num_tokens()

    def __init__(self, global_config, task_id=0, logging_file=None, experience_replay_file=None, global_best_reward_fn=None, found_solution_op=None, assign_code_solution_fn=None, program_count=None, do_iw_summaries=False, stop_on_success=True, dtype=tf.float32, verbose_level=0, is_local=True):
        if False:
            return 10
        self.config = config = global_config.agent
        self.logging_file = logging_file
        self.experience_replay_file = experience_replay_file
        self.task_id = task_id
        self.verbose_level = verbose_level
        self.global_best_reward_fn = global_best_reward_fn
        self.found_solution_op = found_solution_op
        self.assign_code_solution_fn = assign_code_solution_fn
        self.parent_scope_name = tf.get_variable_scope().name
        self.dtype = dtype
        self.allow_eos_token = config.eos_token
        self.stop_on_success = stop_on_success
        self.pi_loss_hparam = config.pi_loss_hparam
        self.vf_loss_hparam = config.vf_loss_hparam
        self.is_local = is_local
        self.top_reward = 0.0
        self.embeddings_trainable = True
        self.no_op = tf.no_op()
        self.learning_rate = tf.constant(config.lr, dtype=dtype, name='learning_rate')
        self.initializer = tf.contrib.layers.variance_scaling_initializer(factor=config.param_init_factor, mode='FAN_AVG', uniform=True, dtype=dtype)
        tf.get_variable_scope().set_initializer(self.initializer)
        self.a2c = config.ema_baseline_decay == 0
        if not self.a2c:
            logging.info('Using exponential moving average REINFORCE baselines.')
            self.ema_baseline_decay = config.ema_baseline_decay
            self.ema_by_len = [0.0] * global_config.timestep_limit
        else:
            logging.info('Using advantage (a2c) with learned value function.')
            self.ema_baseline_decay = 0.0
            self.ema_by_len = None
        if config.topk and config.topk_loss_hparam:
            self.topk_loss_hparam = config.topk_loss_hparam
            self.topk_batch_size = config.topk_batch_size
            if self.topk_batch_size <= 0:
                raise ValueError('topk_batch_size must be a positive integer. Got %s', self.topk_batch_size)
            self.top_episodes = utils.MaxUniquePriorityQueue(config.topk)
            logging.info('Made max-priorty-queue with capacity %d', self.top_episodes.capacity)
        else:
            self.top_episodes = None
            self.topk_loss_hparam = 0.0
            logging.info('No max-priorty-queue')
        self.replay_temperature = config.replay_temperature
        self.num_replay_per_batch = int(global_config.batch_size * config.alpha)
        self.num_on_policy_per_batch = global_config.batch_size - self.num_replay_per_batch
        self.replay_alpha = self.num_replay_per_batch / float(global_config.batch_size)
        logging.info('num_replay_per_batch: %d', self.num_replay_per_batch)
        logging.info('num_on_policy_per_batch: %d', self.num_on_policy_per_batch)
        logging.info('replay_alpha: %s', self.replay_alpha)
        if self.num_replay_per_batch > 0:
            start_time = time.time()
            self.experience_replay = utils.RouletteWheel(unique_mode=True, save_file=experience_replay_file)
            logging.info('Took %s sec to load replay buffer from disk.', int(time.time() - start_time))
            logging.info('Replay buffer file location: "%s"', self.experience_replay.save_file)
        else:
            self.experience_replay = None
        if program_count is not None:
            self.program_count = program_count
            self.program_count_add_ph = tf.placeholder(tf.int64, [], 'program_count_add_ph')
            self.program_count_add_op = self.program_count.assign_add(self.program_count_add_ph)
        batch_size = global_config.batch_size
        logging.info('batch_size: %d', batch_size)
        self.policy_cell = LinearWrapper(tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(cell_size) for cell_size in config.policy_lstm_sizes]), self.action_space, dtype=dtype, suppress_index=None if self.allow_eos_token else misc.BF_EOS_INT)
        self.value_cell = LinearWrapper(tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(cell_size) for cell_size in config.value_lstm_sizes]), 1, dtype=dtype)
        obs_embedding_scope = 'obs_embed'
        with tf.variable_scope(obs_embedding_scope, initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0)):
            obs_embeddings = tf.get_variable('embeddings', [self.observation_space, config.obs_embedding_size], dtype=dtype, trainable=self.embeddings_trainable)
            self.obs_embeddings = obs_embeddings
        initial_state = tf.fill([batch_size], misc.BF_EOS_INT)

        def loop_fn(loop_time, cell_output, cell_state, loop_state):
            if False:
                print('Hello World!')
            'Function called by tf.nn.raw_rnn to instantiate body of the while_loop.\n\n      See https://www.tensorflow.org/api_docs/python/tf/nn/raw_rnn for more\n      information.\n\n      When time is 0, and cell_output, cell_state, loop_state are all None,\n      `loop_fn` will create the initial input, internal cell state, and loop\n      state. When time > 0, `loop_fn` will operate on previous cell output,\n      state, and loop state.\n\n      Args:\n        loop_time: A scalar tensor holding the current timestep (zero based\n            counting).\n        cell_output: Output of the raw_rnn cell at the current timestep.\n        cell_state: Cell internal state at the current timestep.\n        loop_state: Additional loop state. These tensors were returned by the\n            previous call to `loop_fn`.\n\n      Returns:\n        elements_finished: Bool tensor of shape [batch_size] which marks each\n            sequence in the batch as being finished or not finished.\n        next_input: A tensor containing input to be fed into the cell at the\n            next timestep.\n        next_cell_state: Cell internal state to be fed into the cell at the\n            next timestep.\n        emit_output: Tensor to be added to the TensorArray returned by raw_rnn\n            as output from the while_loop.\n        next_loop_state: Additional loop state. These tensors will be fed back\n            into the next call to `loop_fn` as `loop_state`.\n      '
            if cell_output is None:
                next_cell_state = self.policy_cell.zero_state(batch_size, dtype)
                elements_finished = tf.zeros([batch_size], tf.bool)
                output_lengths = tf.ones([batch_size], dtype=tf.int32)
                next_input = tf.gather(obs_embeddings, initial_state)
                emit_output = None
                next_loop_state = (tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True), output_lengths, elements_finished)
            else:
                scaled_logits = cell_output * config.softmax_tr
                (prev_chosen, prev_output_lengths, prev_elements_finished) = loop_state
                next_cell_state = cell_state
                chosen_outputs = tf.to_int32(tf.where(tf.logical_not(prev_elements_finished), tf.multinomial(logits=scaled_logits, num_samples=1)[:, 0], tf.zeros([batch_size], dtype=tf.int64)))
                elements_finished = tf.logical_or(tf.equal(chosen_outputs, misc.BF_EOS_INT), loop_time >= global_config.timestep_limit)
                output_lengths = tf.where(elements_finished, prev_output_lengths, tf.tile(tf.expand_dims(loop_time + 1, 0), [batch_size]))
                next_input = tf.gather(obs_embeddings, chosen_outputs)
                emit_output = scaled_logits
                next_loop_state = (prev_chosen.write(loop_time - 1, chosen_outputs), output_lengths, tf.logical_or(prev_elements_finished, elements_finished))
            return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)
        with tf.variable_scope('policy'):
            (decoder_outputs_ta, _, (sampled_output_ta, output_lengths, _)) = tf.nn.raw_rnn(cell=self.policy_cell, loop_fn=loop_fn)
        policy_logits = tf.transpose(decoder_outputs_ta.stack(), (1, 0, 2), name='policy_logits')
        sampled_tokens = tf.transpose(sampled_output_ta.stack(), (1, 0), name='sampled_tokens')
        rshift_sampled_tokens = rshift_time(sampled_tokens, fill=misc.BF_EOS_INT)
        if self.a2c:
            with tf.variable_scope('value'):
                (value_output, _) = tf.nn.dynamic_rnn(self.value_cell, tf.gather(obs_embeddings, rshift_sampled_tokens), sequence_length=output_lengths, dtype=dtype)
            value = tf.squeeze(value_output, axis=[2])
        else:
            value = tf.zeros([], dtype=dtype)
        self.sampled_batch = AttrDict(logits=policy_logits, value=value, tokens=sampled_tokens, episode_lengths=output_lengths, probs=tf.nn.softmax(policy_logits), log_probs=tf.nn.log_softmax(policy_logits))
        self.adjusted_lengths = tf.placeholder(tf.int32, [None], name='adjusted_lengths')
        self.policy_multipliers = tf.placeholder(dtype, [None, None], name='policy_multipliers')
        self.empirical_values = tf.placeholder(dtype, [None, None], name='empirical_values')
        self.off_policy_targets = tf.placeholder(tf.int32, [None, None], name='off_policy_targets')
        self.off_policy_target_lengths = tf.placeholder(tf.int32, [None], name='off_policy_target_lengths')
        self.actions = tf.placeholder(tf.int32, [None, None], name='actions')
        inputs = rshift_time(self.actions, fill=misc.BF_EOS_INT)
        with tf.variable_scope('policy', reuse=True):
            (logits, _) = tf.nn.dynamic_rnn(self.policy_cell, tf.gather(obs_embeddings, inputs), sequence_length=self.adjusted_lengths, dtype=dtype)
        if self.a2c:
            with tf.variable_scope('value', reuse=True):
                (value_output, _) = tf.nn.dynamic_rnn(self.value_cell, tf.gather(obs_embeddings, inputs), sequence_length=self.adjusted_lengths, dtype=dtype)
            value2 = tf.squeeze(value_output, axis=[2])
        else:
            value2 = tf.zeros([], dtype=dtype)
        self.given_batch = AttrDict(logits=logits, value=value2, tokens=sampled_tokens, episode_lengths=self.adjusted_lengths, probs=tf.nn.softmax(logits), log_probs=tf.nn.log_softmax(logits))
        max_episode_length = tf.shape(self.actions)[1]
        range_row = tf.expand_dims(tf.range(max_episode_length), 0)
        episode_masks = tf.cast(tf.less(range_row, tf.expand_dims(self.given_batch.episode_lengths, 1)), dtype=dtype)
        episode_masks_3d = tf.expand_dims(episode_masks, 2)
        self.a_probs = a_probs = self.given_batch.probs * episode_masks_3d
        self.a_log_probs = a_log_probs = self.given_batch.log_probs * episode_masks_3d
        self.a_value = a_value = self.given_batch.value * episode_masks
        self.a_policy_multipliers = a_policy_multipliers = self.policy_multipliers * episode_masks
        if self.a2c:
            self.a_empirical_values = a_empirical_values = self.empirical_values * episode_masks
        acs_onehot = tf.one_hot(self.actions, self.action_space, dtype=dtype)
        self.acs_onehot = acs_onehot
        chosen_masked_log_probs = acs_onehot * a_log_probs
        pi_target = tf.expand_dims(a_policy_multipliers, -1)
        pi_loss_per_step = chosen_masked_log_probs * pi_target
        self.pi_loss = pi_loss = -tf.reduce_mean(tf.reduce_sum(pi_loss_per_step, axis=[1, 2]), axis=0) * MAGIC_LOSS_MULTIPLIER
        assert len(self.pi_loss.shape) == 0
        self.chosen_log_probs = tf.reduce_sum(chosen_masked_log_probs, axis=2)
        self.chosen_probs = tf.reduce_sum(acs_onehot * a_probs, axis=2)
        if self.a2c:
            vf_loss_per_step = tf.square(a_value - a_empirical_values)
            self.vf_loss = vf_loss = tf.reduce_mean(tf.reduce_sum(vf_loss_per_step, axis=1), axis=0) * MAGIC_LOSS_MULTIPLIER
            assert len(self.vf_loss.shape) == 0
        else:
            self.vf_loss = vf_loss = 0.0
        self.entropy = entropy = -tf.reduce_mean(tf.reduce_sum(a_probs * a_log_probs, axis=[1, 2]), axis=0) * MAGIC_LOSS_MULTIPLIER
        self.negentropy = -entropy
        assert len(self.negentropy.shape) == 0
        self.offp_switch = tf.placeholder(dtype, [], name='offp_switch')
        if self.top_episodes is not None:
            offp_inputs = tf.gather(obs_embeddings, rshift_time(self.off_policy_targets, fill=misc.BF_EOS_INT))
            with tf.variable_scope('policy', reuse=True):
                (offp_logits, _) = tf.nn.dynamic_rnn(self.policy_cell, offp_inputs, self.off_policy_target_lengths, dtype=dtype)
            topk_loss_per_step = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.off_policy_targets, logits=offp_logits, name='topk_loss_per_logit')
            topk_loss = tf.reduce_mean(tf.reduce_sum(topk_loss_per_step, axis=1), axis=0)
            assert len(topk_loss.shape) == 0
            self.topk_loss = topk_loss * self.offp_switch
            logging.info('Including off policy loss.')
        else:
            self.topk_loss = topk_loss = 0.0
        self.entropy_hparam = tf.constant(config.entropy_beta, dtype=dtype, name='entropy_beta')
        self.pi_loss_term = pi_loss * self.pi_loss_hparam
        self.vf_loss_term = vf_loss * self.vf_loss_hparam
        self.entropy_loss_term = self.negentropy * self.entropy_hparam
        self.topk_loss_term = self.topk_loss_hparam * topk_loss
        self.loss = self.pi_loss_term + self.vf_loss_term + self.entropy_loss_term + self.topk_loss_term
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.trainable_variables = params
        self.sync_variables = self.trainable_variables
        non_embedding_params = [p for p in params if obs_embedding_scope not in p.name]
        self.non_embedding_params = non_embedding_params
        self.params = params
        if config.regularizer:
            logging.info('Adding L2 regularizer with scale %.2f.', config.regularizer)
            self.regularizer = config.regularizer * sum((tf.nn.l2_loss(w) for w in non_embedding_params))
            self.loss += self.regularizer
        else:
            logging.info('Skipping regularizer.')
            self.regularizer = 0.0
        if self.is_local:
            unclipped_grads = tf.gradients(self.loss, params)
            self.dense_unclipped_grads = [tf.convert_to_tensor(g) for g in unclipped_grads]
            (self.grads, self.global_grad_norm) = tf.clip_by_global_norm(unclipped_grads, config.grad_clip_threshold)
            self.gradients_dict = dict(zip(params, self.grads))
            self.optimizer = make_optimizer(config.optimizer, self.learning_rate)
            self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, tf.get_variable_scope().name)
        self.do_iw_summaries = do_iw_summaries
        if self.do_iw_summaries:
            b = None
            self.log_iw_replay_ph = tf.placeholder(tf.float32, [b], 'log_iw_replay_ph')
            self.log_iw_policy_ph = tf.placeholder(tf.float32, [b], 'log_iw_policy_ph')
            self.log_prob_replay_ph = tf.placeholder(tf.float32, [b], 'log_prob_replay_ph')
            self.log_prob_policy_ph = tf.placeholder(tf.float32, [b], 'log_prob_policy_ph')
            self.log_norm_replay_weights_ph = tf.placeholder(tf.float32, [b], 'log_norm_replay_weights_ph')
            self.iw_summary_op = tf.summary.merge([tf.summary.histogram('is/log_iw_replay', self.log_iw_replay_ph), tf.summary.histogram('is/log_iw_policy', self.log_iw_policy_ph), tf.summary.histogram('is/log_prob_replay', self.log_prob_replay_ph), tf.summary.histogram('is/log_prob_policy', self.log_prob_policy_ph), tf.summary.histogram('is/log_norm_replay_weights', self.log_norm_replay_weights_ph)])

    def make_summary_ops(self):
        if False:
            for i in range(10):
                print('nop')
        'Construct summary ops for the model.'
        size = tf.cast(tf.reduce_sum(self.given_batch.episode_lengths), dtype=self.dtype)
        offp_size = tf.cast(tf.reduce_sum(self.off_policy_target_lengths), dtype=self.dtype)
        scope_prefix = self.parent_scope_name

        def _remove_prefix(prefix, name):
            if False:
                for i in range(10):
                    print('nop')
            assert name.startswith(prefix)
            return name[len(prefix):]
        self.rl_summary_op = tf.summary.merge([tf.summary.scalar('model/policy_loss', self.pi_loss / size), tf.summary.scalar('model/value_loss', self.vf_loss / size), tf.summary.scalar('model/topk_loss', self.topk_loss / offp_size), tf.summary.scalar('model/entropy', self.entropy / size), tf.summary.scalar('model/loss', self.loss / size), tf.summary.scalar('model/grad_norm', tf.global_norm(self.grads)), tf.summary.scalar('model/unclipped_grad_norm', self.global_grad_norm), tf.summary.scalar('model/non_embedding_var_norm', tf.global_norm(self.non_embedding_params)), tf.summary.scalar('hparams/entropy_beta', self.entropy_hparam), tf.summary.scalar('hparams/topk_loss_hparam', self.topk_loss_hparam), tf.summary.scalar('hparams/learning_rate', self.learning_rate), tf.summary.scalar('model/trainable_var_norm', tf.global_norm(self.trainable_variables)), tf.summary.scalar('loss/loss', self.loss), tf.summary.scalar('loss/entropy', self.entropy_loss_term), tf.summary.scalar('loss/vf', self.vf_loss_term), tf.summary.scalar('loss/policy', self.pi_loss_term), tf.summary.scalar('loss/offp', self.topk_loss_term)] + [tf.summary.scalar('param_norms/' + _remove_prefix(scope_prefix + '/', p.name), tf.norm(p)) for p in self.params] + [tf.summary.scalar('grad_norms/' + _remove_prefix(scope_prefix + '/', p.name), tf.norm(g)) for (p, g) in zip(self.params, self.grads)] + [tf.summary.scalar('unclipped_grad_norms/' + _remove_prefix(scope_prefix + '/', p.name), tf.norm(g)) for (p, g) in zip(self.params, self.dense_unclipped_grads)])
        self.text_summary_placeholder = tf.placeholder(tf.string, shape=[])
        self.rl_text_summary_op = tf.summary.text('rl', self.text_summary_placeholder)

    def _rl_text_summary(self, session, step, npe, tot_r, num_steps, input_case, code_output, code, reason):
        if False:
            while True:
                i = 10
        'Logs summary about a single episode and creates a text_summary for TB.\n\n    Args:\n      session: tf.Session instance.\n      step: Global training step.\n      npe: Number of programs executed so far.\n      tot_r: Total reward.\n      num_steps: Number of timesteps in the episode (i.e. code length).\n      input_case: Inputs for test cases.\n      code_output: Outputs produced by running the code on the inputs.\n      code: String representation of the code.\n      reason: Reason for the reward assigned by the task.\n\n    Returns:\n      Serialized text summary data for tensorboard.\n    '
        if not input_case:
            input_case = ' '
        if not code_output:
            code_output = ' '
        if not code:
            code = ' '
        text = 'Tot R: **%.2f**;  Len: **%d**;  Reason: **%s**\n\nInput: **`%s`**; Output: **`%s`**\n\nCode: **`%s`**' % (tot_r, num_steps, reason, input_case, code_output, code)
        text_summary = session.run(self.rl_text_summary_op, {self.text_summary_placeholder: text})
        logging.info('Step %d.\t NPE: %d\t Reason: %s.\t Tot R: %.2f.\t Length: %d. \tInput: %s \tOutput: %s \tProgram: %s', step, npe, reason, tot_r, num_steps, input_case, code_output, code)
        return text_summary

    def _rl_reward_summary(self, total_rewards):
        if False:
            while True:
                i = 10
        'Create summary ops that report on episode rewards.\n\n    Creates summaries for average, median, max, and min rewards in the batch.\n\n    Args:\n      total_rewards: Tensor of shape [batch_size] containing the total reward\n          from each episode in the batch.\n\n    Returns:\n      tf.Summary op.\n    '
        tr = np.asarray(total_rewards)
        reward_summary = tf.Summary(value=[tf.Summary.Value(tag='reward/avg', simple_value=np.mean(tr)), tf.Summary.Value(tag='reward/med', simple_value=np.median(tr)), tf.Summary.Value(tag='reward/max', simple_value=np.max(tr)), tf.Summary.Value(tag='reward/min', simple_value=np.min(tr))])
        return reward_summary

    def _iw_summary(self, session, replay_iw, replay_log_probs, norm_replay_weights, on_policy_iw, on_policy_log_probs):
        if False:
            for i in range(10):
                print('nop')
        'Compute summaries for importance weights at a given batch.\n\n    Args:\n      session: tf.Session instance.\n      replay_iw: Importance weights for episodes from replay buffer.\n      replay_log_probs: Total log probabilities of the replay episodes under the\n          current policy.\n      norm_replay_weights: Normalized replay weights, i.e. values in `replay_iw`\n          divided by the total weight in the entire replay buffer. Note, this is\n          also the probability of selecting each episode from the replay buffer\n          (in a roulette wheel replay buffer).\n      on_policy_iw: Importance weights for episodes sampled from the current\n          policy.\n      on_policy_log_probs: Total log probabilities of the on-policy episodes\n          under the current policy.\n\n    Returns:\n      Serialized TF summaries. Use a summary writer to write these summaries to\n      disk.\n    '
        return session.run(self.iw_summary_op, {self.log_iw_replay_ph: np.log(replay_iw), self.log_iw_policy_ph: np.log(on_policy_iw), self.log_norm_replay_weights_ph: np.log(norm_replay_weights), self.log_prob_replay_ph: replay_log_probs, self.log_prob_policy_ph: on_policy_log_probs})

    def _compute_iw(self, policy_log_probs, replay_weights):
        if False:
            i = 10
            return i + 15
        'Compute importance weights for a batch of episodes.\n\n    Arguments are iterables of length batch_size.\n\n    Args:\n      policy_log_probs: Log probability of each episode under the current\n          policy.\n      replay_weights: Weight of each episode in the replay buffer. 0 for\n          episodes not sampled from the replay buffer (i.e. sampled from the\n          policy).\n\n    Returns:\n      Numpy array of shape [batch_size] containing the importance weight for\n      each episode in the batch.\n    '
        log_total_replay_weight = log(self.experience_replay.total_weight)
        a = float(self.replay_alpha)
        a_com = 1.0 - a
        importance_weights = np.asarray([1.0 / (a_com + a * exp(log(replay_weight) - log_total_replay_weight - log_p)) if replay_weight > 0 else 1.0 / a_com for (log_p, replay_weight) in zip(policy_log_probs, replay_weights)])
        return importance_weights

    def update_step(self, session, rl_batch, train_op, global_step_op, return_gradients=False):
        if False:
            return 10
        'Perform gradient update on the model.\n\n    Args:\n      session: tf.Session instance.\n      rl_batch: RLBatch instance from data.py. Use DataManager to create a\n          RLBatch for each call to update_step. RLBatch contains a batch of\n          tasks.\n      train_op: A TF op which will perform the gradient update. LMAgent does not\n          own its training op, so that trainers can do distributed training\n          and construct a specialized training op.\n      global_step_op: A TF op which will return the current global step when\n          run (should not increment it).\n      return_gradients: If True, the gradients will be saved and returned from\n          this method call. This is useful for testing.\n\n    Returns:\n      Results from the update step in a UpdateStepResult namedtuple, including\n      global step, global NPE, serialized summaries, and optionally gradients.\n    '
        assert self.is_local
        if self.experience_replay is None:
            num_programs_from_policy = rl_batch.batch_size
            (batch_actions, batch_values, episode_lengths) = session.run([self.sampled_batch.tokens, self.sampled_batch.value, self.sampled_batch.episode_lengths])
            if episode_lengths.size == 0:
                logging.warn('Shapes:\nbatch_actions.shape: %s\nbatch_values.shape: %s\nepisode_lengths.shape: %s\n', batch_actions.shape, batch_values.shape, episode_lengths.shape)
            code_scores = compute_rewards(rl_batch, batch_actions, episode_lengths)
            code_strings = code_scores.code_strings
            batch_tot_r = code_scores.total_rewards
            test_cases = code_scores.test_cases
            code_outputs = code_scores.code_outputs
            reasons = code_scores.reasons
            (batch_targets, batch_returns) = process_episodes(code_scores.batch_rewards, episode_lengths, a2c=self.a2c, baselines=self.ema_by_len, batch_values=batch_values)
            batch_policy_multipliers = batch_targets
            batch_emp_values = batch_returns if self.a2c else [[]]
            adjusted_lengths = episode_lengths
            if self.top_episodes:
                assert len(self.top_episodes) > 0
                off_policy_targets = [item for (item, _) in self.top_episodes.random_sample(self.topk_batch_size)]
                off_policy_target_lengths = [len(t) for t in off_policy_targets]
                off_policy_targets = utils.stack_pad(off_policy_targets, pad_axes=0, dtype=np.int32)
                offp_switch = 1
            else:
                off_policy_targets = [[0]]
                off_policy_target_lengths = [1]
                offp_switch = 0
            fetches = {'global_step': global_step_op, 'program_count': self.program_count, 'summaries': self.rl_summary_op, 'train_op': train_op, 'gradients': self.gradients_dict if return_gradients else self.no_op}
            fetched = session.run(fetches, {self.actions: batch_actions, self.empirical_values: batch_emp_values, self.policy_multipliers: batch_policy_multipliers, self.adjusted_lengths: adjusted_lengths, self.off_policy_targets: off_policy_targets, self.off_policy_target_lengths: off_policy_target_lengths, self.offp_switch: offp_switch})
            combined_adjusted_lengths = adjusted_lengths
            combined_returns = batch_returns
        else:
            (batch_actions, batch_values, episode_lengths, log_probs) = session.run([self.sampled_batch.tokens, self.sampled_batch.value, self.sampled_batch.episode_lengths, self.sampled_batch.log_probs])
            if episode_lengths.size == 0:
                logging.warn('Shapes:\nbatch_actions.shape: %s\nbatch_values.shape: %s\nepisode_lengths.shape: %s\n', batch_actions.shape, batch_values.shape, episode_lengths.shape)
            empty_replay_buffer = self.experience_replay.is_empty() if self.experience_replay is not None else True
            num_programs_from_replay_buff = self.num_replay_per_batch if not empty_replay_buffer else 0
            num_programs_from_policy = rl_batch.batch_size - num_programs_from_replay_buff
            if not empty_replay_buffer and num_programs_from_replay_buff:
                result = self.experience_replay.sample_many(num_programs_from_replay_buff)
                (experience_samples, replay_weights) = zip(*result)
                (replay_actions, replay_rewards, _, replay_adjusted_lengths) = zip(*experience_samples)
                replay_batch_actions = utils.stack_pad(replay_actions, pad_axes=0, dtype=np.int32)
                (all_replay_log_probs,) = session.run([self.given_batch.log_probs], {self.actions: replay_batch_actions, self.adjusted_lengths: replay_adjusted_lengths})
                replay_log_probs = [np.choose(replay_actions[i], all_replay_log_probs[i, :l].T).sum() for (i, l) in enumerate(replay_adjusted_lengths)]
            else:
                replay_actions = None
                replay_policy_multipliers = None
                replay_adjusted_lengths = None
                replay_log_probs = None
                replay_weights = None
                replay_returns = None
                on_policy_weights = [0] * num_programs_from_replay_buff
            assert not self.a2c
            code_scores = compute_rewards(rl_batch, batch_actions, episode_lengths, batch_size=num_programs_from_policy)
            code_strings = code_scores.code_strings
            batch_tot_r = code_scores.total_rewards
            test_cases = code_scores.test_cases
            code_outputs = code_scores.code_outputs
            reasons = code_scores.reasons
            p = num_programs_from_policy
            (batch_targets, batch_returns) = process_episodes(code_scores.batch_rewards, episode_lengths[:p], a2c=False, baselines=self.ema_by_len)
            batch_policy_multipliers = batch_targets
            batch_emp_values = [[]]
            on_policy_returns = batch_returns
            if not empty_replay_buffer and num_programs_from_replay_buff:
                offp_batch_rewards = [[0.0] * (l - 1) + [r] for (l, r) in zip(replay_adjusted_lengths, replay_rewards)]
                assert len(offp_batch_rewards) == num_programs_from_replay_buff
                assert len(replay_adjusted_lengths) == num_programs_from_replay_buff
                (replay_batch_targets, replay_returns) = process_episodes(offp_batch_rewards, replay_adjusted_lengths, a2c=False, baselines=self.ema_by_len)
                replay_policy_multipliers = [replay_batch_targets[i, :l] for (i, l) in enumerate(replay_adjusted_lengths[:num_programs_from_replay_buff])]
            adjusted_lengths = episode_lengths[:num_programs_from_policy]
            if self.top_episodes:
                assert len(self.top_episodes) > 0
                off_policy_targets = [item for (item, _) in self.top_episodes.random_sample(self.topk_batch_size)]
                off_policy_target_lengths = [len(t) for t in off_policy_targets]
                off_policy_targets = utils.stack_pad(off_policy_targets, pad_axes=0, dtype=np.int32)
                offp_switch = 1
            else:
                off_policy_targets = [[0]]
                off_policy_target_lengths = [1]
                offp_switch = 0
            if num_programs_from_policy:
                separate_actions = [batch_actions[i, :l] for (i, l) in enumerate(adjusted_lengths)]
                chosen_log_probs = [np.choose(separate_actions[i], log_probs[i, :l].T) for (i, l) in enumerate(adjusted_lengths)]
                new_experiences = [(separate_actions[i], batch_tot_r[i], chosen_log_probs[i].sum(), l) for (i, l) in enumerate(adjusted_lengths)]
                on_policy_policy_multipliers = [batch_policy_multipliers[i, :l] for (i, l) in enumerate(adjusted_lengths)]
                (on_policy_actions, _, on_policy_log_probs, on_policy_adjusted_lengths) = zip(*new_experiences)
            else:
                new_experiences = []
                on_policy_policy_multipliers = []
                on_policy_actions = []
                on_policy_log_probs = []
                on_policy_adjusted_lengths = []
            if not empty_replay_buffer and num_programs_from_replay_buff:
                on_policy_weights = [0] * num_programs_from_policy
                for (i, cs) in enumerate(code_strings):
                    if self.experience_replay.has_key(cs):
                        on_policy_weights[i] = self.experience_replay.get_weight(cs)
            combined_actions = join(replay_actions, on_policy_actions)
            combined_policy_multipliers = join(replay_policy_multipliers, on_policy_policy_multipliers)
            combined_adjusted_lengths = join(replay_adjusted_lengths, on_policy_adjusted_lengths)
            combined_returns = join(replay_returns, on_policy_returns)
            combined_actions = utils.stack_pad(combined_actions, pad_axes=0)
            combined_policy_multipliers = utils.stack_pad(combined_policy_multipliers, pad_axes=0)
            combined_on_policy_log_probs = join(replay_log_probs, on_policy_log_probs)
            combined_q_weights = join(replay_weights, on_policy_weights)
            if empty_replay_buffer:
                combined_policy_multipliers *= 0
            elif not num_programs_from_replay_buff:
                combined_policy_multipliers = np.ones([len(combined_actions), 1], dtype=np.float32)
            else:
                importance_weights = self._compute_iw(combined_on_policy_log_probs, combined_q_weights)
                if self.config.iw_normalize:
                    importance_weights *= float(rl_batch.batch_size) / importance_weights.sum()
                combined_policy_multipliers *= importance_weights.reshape(-1, 1)
            assert self.program_count is not None
            fetches = {'global_step': global_step_op, 'program_count': self.program_count, 'summaries': self.rl_summary_op, 'train_op': train_op, 'gradients': self.gradients_dict if return_gradients else self.no_op}
            fetched = session.run(fetches, {self.actions: combined_actions, self.empirical_values: [[]], self.policy_multipliers: combined_policy_multipliers, self.adjusted_lengths: combined_adjusted_lengths, self.off_policy_targets: off_policy_targets, self.off_policy_target_lengths: off_policy_target_lengths, self.offp_switch: offp_switch})
            self.experience_replay.add_many(objs=new_experiences, weights=[exp(r / self.replay_temperature) for r in batch_tot_r], keys=code_strings)
        session.run([self.program_count_add_op], {self.program_count_add_ph: num_programs_from_policy})
        if not self.a2c:
            for i in xrange(rl_batch.batch_size):
                episode_length = combined_adjusted_lengths[i]
                empirical_returns = combined_returns[i, :episode_length]
                for j in xrange(episode_length):
                    self.ema_by_len[j] = self.ema_baseline_decay * self.ema_by_len[j] + (1 - self.ema_baseline_decay) * empirical_returns[j]
        global_step = fetched['global_step']
        global_npe = fetched['program_count']
        core_summaries = fetched['summaries']
        summaries_list = [core_summaries]
        if num_programs_from_policy:
            s_i = 0
            text_summary = self._rl_text_summary(session, global_step, global_npe, batch_tot_r[s_i], episode_lengths[s_i], test_cases[s_i], code_outputs[s_i], code_strings[s_i], reasons[s_i])
            reward_summary = self._rl_reward_summary(batch_tot_r)
            is_best = False
            if self.global_best_reward_fn:
                best_reward = np.max(batch_tot_r)
                is_best = self.global_best_reward_fn(session, best_reward)
            if self.found_solution_op is not None and 'correct' in reasons:
                session.run(self.found_solution_op)
                if self.stop_on_success:
                    solutions = [{'code': code_strings[i], 'reward': batch_tot_r[i], 'npe': global_npe} for i in xrange(len(reasons)) if reasons[i] == 'correct']
                elif is_best:
                    solutions = [{'code': code_strings[np.argmax(batch_tot_r)], 'reward': np.max(batch_tot_r), 'npe': global_npe}]
                else:
                    solutions = []
                if solutions:
                    if self.assign_code_solution_fn:
                        self.assign_code_solution_fn(session, solutions[0]['code'])
                    with tf.gfile.FastGFile(self.logging_file, 'a') as writer:
                        for solution_dict in solutions:
                            writer.write(str(solution_dict) + '\n')
            max_i = np.argmax(batch_tot_r)
            max_tot_r = batch_tot_r[max_i]
            if max_tot_r >= self.top_reward:
                if max_tot_r >= self.top_reward:
                    self.top_reward = max_tot_r
                logging.info('Top code: r=%.2f, \t%s', max_tot_r, code_strings[max_i])
            if self.top_episodes is not None:
                self.top_episodes.push(max_tot_r, tuple(batch_actions[max_i, :episode_lengths[max_i]]))
            summaries_list += [text_summary, reward_summary]
            if self.do_iw_summaries and (not empty_replay_buffer):
                norm_replay_weights = [w / self.experience_replay.total_weight for w in replay_weights]
                replay_iw = self._compute_iw(replay_log_probs, replay_weights)
                on_policy_iw = self._compute_iw(on_policy_log_probs, on_policy_weights)
                summaries_list.append(self._iw_summary(session, replay_iw, replay_log_probs, norm_replay_weights, on_policy_iw, on_policy_log_probs))
        return UpdateStepResult(global_step=global_step, global_npe=global_npe, summaries_list=summaries_list, gradients_dict=fetched['gradients'])

def io_to_text(io_case, io_type):
    if False:
        while True:
            i = 10
    if isinstance(io_case, misc.IOTuple):
        return ','.join([io_to_text(e, io_type) for e in io_case])
    if io_type == misc.IOType.string:
        return misc.tokens_to_text(io_case)
    if io_type == misc.IOType.integer or io_type == misc.IOType.boolean:
        if len(io_case) == 1:
            return str(io_case[0])
        return str(io_case)
CodeScoreInfo = namedtuple('CodeScoreInfo', ['code_strings', 'batch_rewards', 'total_rewards', 'test_cases', 'code_outputs', 'reasons'])

def compute_rewards(rl_batch, batch_actions, episode_lengths, batch_size=None):
    if False:
        return 10
    'Compute rewards for each episode in the batch.\n\n  Args:\n    rl_batch: A data.RLBatch instance. This holds information about the task\n        each episode is solving, and a reward function for each episode.\n    batch_actions: Contains batch of episodes. Each sequence of actions will be\n        converted into a BF program and then scored. A numpy array of shape\n        [batch_size, max_sequence_length].\n    episode_lengths: The sequence length of each episode in the batch. Iterable\n        of length batch_size.\n    batch_size: (optional) number of programs to score. Use this to limit the\n        number of programs executed from this batch. For example, when doing\n        importance sampling some of the on-policy episodes will be discarded\n        and they should not be executed. `batch_size` can be less than or equal\n        to the size of the input batch.\n\n  Returns:\n    CodeScoreInfo namedtuple instance. This holds not just the computed rewards,\n    but additional information computed during code execution which can be used\n    for debugging and monitoring. this includes: BF code strings, test cases\n    the code was executed on, code outputs from those test cases, and reasons\n    for success or failure.\n  '
    code_strings = [''.join([misc.bf_int2char(a) for a in action_sequence[:l]]) for (action_sequence, l) in zip(batch_actions, episode_lengths)]
    if batch_size is None:
        batch_size = len(code_strings)
    else:
        assert batch_size <= len(code_strings)
        code_strings = code_strings[:batch_size]
    if isinstance(rl_batch.reward_fns, (list, tuple)):
        assert len(rl_batch.reward_fns) >= batch_size
        r_fn_results = [rl_batch.reward_fns[i](code_strings[i]) for i in xrange(batch_size)]
    else:
        r_fn_results = rl_batch.reward_fns(code_strings)
    batch_rewards = [r.episode_rewards for r in r_fn_results]
    total_rewards = [sum(b) for b in batch_rewards]
    test_cases = [io_to_text(r.input_case, r.input_type) for r in r_fn_results]
    code_outputs = [io_to_text(r.code_output, r.output_type) for r in r_fn_results]
    reasons = [r.reason for r in r_fn_results]
    return CodeScoreInfo(code_strings=code_strings, batch_rewards=batch_rewards, total_rewards=total_rewards, test_cases=test_cases, code_outputs=code_outputs, reasons=reasons)

def process_episodes(batch_rewards, episode_lengths, a2c=False, baselines=None, batch_values=None):
    if False:
        return 10
    'Compute REINFORCE targets.\n\n  REINFORCE here takes the form:\n  grad_t = grad[log(pi(a_t|c_t))*target_t]\n  where c_t is context: i.e. RNN state or environment state (or both).\n\n  Two types of targets are supported:\n  1) Advantage actor critic (a2c).\n  2) Vanilla REINFORCE with baseline.\n\n  Args:\n    batch_rewards: Rewards received in each episode in the batch. A numpy array\n        of shape [batch_size, max_sequence_length]. Note, these are per-timestep\n        rewards, not total reward.\n    episode_lengths: Length of each episode. An iterable of length batch_size.\n    a2c: A bool. Whether to compute a2c targets (True) or vanilla targets\n        (False).\n    baselines: If a2c is False, provide baselines for each timestep. This is a\n        list (or indexable container) of length max_time. Note: baselines are\n        shared across all episodes, which is why there is no batch dimension.\n        It is up to the caller to update baselines accordingly.\n    batch_values: If a2c is True, provide values computed by a value estimator.\n        A numpy array of shape [batch_size, max_sequence_length].\n\n  Returns:\n    batch_targets: REINFORCE targets for each episode and timestep. A numpy\n        array of shape [batch_size, max_sequence_length].\n    batch_returns: Returns computed for each episode and timestep. This is for\n        reference, and is not used in the REINFORCE gradient update (but was\n        used to compute the targets). A numpy array of shape\n        [batch_size, max_sequence_length].\n  '
    num_programs = len(batch_rewards)
    assert num_programs <= len(episode_lengths)
    batch_returns = [None] * num_programs
    batch_targets = [None] * num_programs
    for i in xrange(num_programs):
        episode_length = episode_lengths[i]
        assert len(batch_rewards[i]) == episode_length
        if a2c:
            assert batch_values is not None
            episode_values = batch_values[i, :episode_length]
            episode_rewards = batch_rewards[i]
            (emp_val, gen_adv) = rollout_lib.discounted_advantage_and_rewards(episode_rewards, episode_values, gamma=1.0, lambda_=1.0)
            batch_returns[i] = emp_val
            batch_targets[i] = gen_adv
        else:
            assert baselines is not None
            empirical_returns = rollout_lib.discount(batch_rewards[i], gamma=1.0)
            targets = [None] * episode_length
            for j in xrange(episode_length):
                targets[j] = empirical_returns[j] - baselines[j]
            batch_returns[i] = empirical_returns
            batch_targets[i] = targets
    batch_returns = utils.stack_pad(batch_returns, 0)
    if num_programs:
        batch_targets = utils.stack_pad(batch_targets, 0)
    else:
        batch_targets = np.array([], dtype=np.float32)
    return (batch_targets, batch_returns)