"""TensorFlow policy class used for R2D2."""
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import ray
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS, build_q_model, clip_gradients, compute_q_values, postprocess_nstep_and_prio
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import LearningRateSchedule, TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import huber_loss
from ray.rllib.utils.typing import AlgorithmConfigDict, ModelInputDict, TensorType
(tf1, tf, tfv) = try_import_tf()

def build_r2d2_model(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> Tuple[ModelV2, ActionDistribution]:
    if False:
        i = 10
        return i + 15
    "Build q_model and target_model for DQN\n\n    Args:\n        policy: The policy, which will use the model for optimization.\n        obs_space (gym.spaces.Space): The policy's observation space.\n        action_space (gym.spaces.Space): The policy's action space.\n        config (AlgorithmConfigDict):\n\n    Returns:\n        q_model\n            Note: The target q model will not be returned, just assigned to\n            `policy.target_model`.\n    "
    model = build_q_model(policy, obs_space, action_space, config)
    assert model.get_initial_state() != [] or model.view_requirements.get('state_in_0') is not None, 'R2D2 requires its model to be a recurrent one! Try using `model.use_lstm` or `model.use_attention` in your config to auto-wrap your model with an LSTM- or attention net.'
    return model

def r2d2_loss(policy: Policy, model, _, train_batch: SampleBatch) -> TensorType:
    if False:
        for i in range(10):
            print('nop')
    'Constructs the loss for R2D2TFPolicy.\n\n    Args:\n        policy: The Policy to calculate the loss for.\n        model (ModelV2): The Model to calculate the loss for.\n        train_batch: The training data.\n\n    Returns:\n        TensorType: A single loss tensor.\n    '
    config = policy.config
    i = 0
    state_batches = []
    while 'state_in_{}'.format(i) in train_batch:
        state_batches.append(train_batch['state_in_{}'.format(i)])
        i += 1
    assert state_batches
    (q, _, _, _) = compute_q_values(policy, model, train_batch, state_batches=state_batches, seq_lens=train_batch.get(SampleBatch.SEQ_LENS), explore=False, is_training=True)
    (q_target, _, _, _) = compute_q_values(policy, policy.target_model, train_batch, state_batches=state_batches, seq_lens=train_batch.get(SampleBatch.SEQ_LENS), explore=False, is_training=True)
    if not hasattr(policy, 'target_q_func_vars'):
        policy.target_q_func_vars = policy.target_model.variables()
    actions = tf.cast(train_batch[SampleBatch.ACTIONS], tf.int64)
    dones = tf.cast(train_batch[SampleBatch.TERMINATEDS], tf.float32)
    rewards = train_batch[SampleBatch.REWARDS]
    weights = tf.cast(train_batch[PRIO_WEIGHTS], tf.float32)
    B = tf.shape(state_batches[0])[0]
    T = tf.shape(q)[0] // B
    one_hot_selection = tf.one_hot(actions, policy.action_space.n)
    q_selected = tf.reduce_sum(tf.where(q > tf.float32.min, q, tf.zeros_like(q)) * one_hot_selection, axis=1)
    if config['double_q']:
        best_actions = tf.argmax(q, axis=1)
    else:
        best_actions = tf.argmax(q_target, axis=1)
    best_actions_one_hot = tf.one_hot(best_actions, policy.action_space.n)
    q_target_best = tf.reduce_sum(tf.where(q_target > tf.float32.min, q_target, tf.zeros_like(q_target)) * best_actions_one_hot, axis=1)
    if config['num_atoms'] > 1:
        raise ValueError('Distributional R2D2 not supported yet!')
    else:
        q_target_best_masked_tp1 = (1.0 - dones) * tf.concat([q_target_best[1:], tf.constant([0.0])], axis=0)
        if config['use_h_function']:
            h_inv = h_inverse(q_target_best_masked_tp1, config['h_function_epsilon'])
            target = h_function(rewards + config['gamma'] ** config['n_step'] * h_inv, config['h_function_epsilon'])
        else:
            target = rewards + config['gamma'] ** config['n_step'] * q_target_best_masked_tp1
        seq_mask = tf.sequence_mask(train_batch[SampleBatch.SEQ_LENS], T)[:, :-1]
        burn_in = policy.config['replay_buffer_config']['replay_burn_in']
        if burn_in > 0:
            seq_mask = tf.cond(pred=tf.convert_to_tensor(burn_in, tf.int32) < T, true_fn=lambda : tf.concat([tf.fill([B, burn_in], False), seq_mask[:, burn_in:]], 1), false_fn=lambda : seq_mask)

        def reduce_mean_valid(t):
            if False:
                for i in range(10):
                    print('nop')
            return tf.reduce_mean(tf.boolean_mask(t, seq_mask))
        q_selected = tf.reshape(q_selected, [B, T])[:, :-1]
        td_error = q_selected - tf.stop_gradient(tf.reshape(target, [B, T])[:, :-1])
        td_error = td_error * tf.cast(seq_mask, tf.float32)
        weights = tf.reshape(weights, [B, T])[:, :-1]
        policy._total_loss = reduce_mean_valid(weights * huber_loss(td_error))
        policy._td_error = tf.reduce_mean(td_error, axis=-1)
        policy._loss_stats = {'mean_q': reduce_mean_valid(q_selected), 'min_q': tf.reduce_min(q_selected), 'max_q': tf.reduce_max(q_selected), 'mean_td_error': reduce_mean_valid(td_error)}
    return policy._total_loss

def h_function(x, epsilon=1.0):
    if False:
        while True:
            i = 10
    'h-function to normalize target Qs, described in the paper [1].\n\n    h(x) = sign(x) * [sqrt(abs(x) + 1) - 1] + epsilon * x\n\n    Used in [1] in combination with h_inverse:\n      targets = h(r + gamma * h_inverse(Q^))\n    '
    return tf.sign(x) * (tf.sqrt(tf.abs(x) + 1.0) - 1.0) + epsilon * x

def h_inverse(x, epsilon=1.0):
    if False:
        while True:
            i = 10
    'Inverse if the above h-function, described in the paper [1].\n\n    If x > 0.0:\n    h-1(x) = [2eps * x + (2eps + 1) - sqrt(4eps x + (2eps + 1)^2)] /\n        (2 * eps^2)\n\n    If x < 0.0:\n    h-1(x) = [2eps * x + (2eps + 1) + sqrt(-4eps x + (2eps + 1)^2)] /\n        (2 * eps^2)\n    '
    two_epsilon = epsilon * 2
    if_x_pos = (two_epsilon * x + (two_epsilon + 1.0) - tf.sqrt(4.0 * epsilon * x + (two_epsilon + 1.0) ** 2)) / (2.0 * epsilon ** 2)
    if_x_neg = (two_epsilon * x - (two_epsilon + 1.0) + tf.sqrt(-4.0 * epsilon * x + (two_epsilon + 1.0) ** 2)) / (2.0 * epsilon ** 2)
    return tf.where(x < 0.0, if_x_neg, if_x_pos)

class ComputeTDErrorMixin:
    """Assign the `compute_td_error` method to the R2D2TFPolicy

    This allows us to prioritize on the worker side.
    """

    def __init__(self):
        if False:
            while True:
                i = 10

        def compute_td_error(obs_t, act_t, rew_t, obs_tp1, terminateds_mask, importance_weights):
            if False:
                return 10
            input_dict = self._lazy_tensor_dict({SampleBatch.CUR_OBS: obs_t})
            input_dict[SampleBatch.ACTIONS] = act_t
            input_dict[SampleBatch.REWARDS] = rew_t
            input_dict[SampleBatch.NEXT_OBS] = obs_tp1
            input_dict[SampleBatch.TERMINATEDS] = terminateds_mask
            input_dict[PRIO_WEIGHTS] = importance_weights
            r2d2_loss(self, self.model, None, input_dict)
            return self._td_error
        self.compute_td_error = compute_td_error

def get_distribution_inputs_and_class(policy: Policy, model: ModelV2, *, input_dict: ModelInputDict, state_batches: Optional[List[TensorType]]=None, seq_lens: Optional[TensorType]=None, explore: bool=True, is_training: bool=False, **kwargs) -> Tuple[TensorType, type, List[TensorType]]:
    if False:
        for i in range(10):
            print('nop')
    if policy.config['framework'] == 'torch':
        from ray.rllib.algorithms.r2d2.r2d2_torch_policy import compute_q_values as torch_compute_q_values
        func = torch_compute_q_values
    else:
        func = compute_q_values
    (q_vals, logits, probs_or_logits, state_out) = func(policy, model, input_dict, state_batches, seq_lens, explore, is_training)
    policy.q_values = q_vals
    if not hasattr(policy, 'q_func_vars'):
        policy.q_func_vars = model.variables()
    action_dist_class = TorchCategorical if policy.config['framework'] == 'torch' else Categorical
    return (policy.q_values, action_dist_class, state_out)

def adam_optimizer(policy: Policy, config: AlgorithmConfigDict) -> 'tf.keras.optimizers.Optimizer':
    if False:
        for i in range(10):
            print('nop')
    return tf1.train.AdamOptimizer(learning_rate=policy.cur_lr, epsilon=config['adam_epsilon'])

def build_q_stats(policy: Policy, batch) -> Dict[str, TensorType]:
    if False:
        return 10
    return dict({'cur_lr': policy.cur_lr}, **policy._loss_stats)

def setup_early_mixins(policy: Policy, obs_space, action_space, config: AlgorithmConfigDict) -> None:
    if False:
        print('Hello World!')
    LearningRateSchedule.__init__(policy, config['lr'], config['lr_schedule'])

def before_loss_init(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> None:
    if False:
        while True:
            i = 10
    ComputeTDErrorMixin.__init__(policy)

def setup_late_mixins(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> None:
    if False:
        return 10
    TargetNetworkMixin.__init__(policy)
R2D2TFPolicy = build_tf_policy(name='R2D2TFPolicy', loss_fn=r2d2_loss, get_default_config=lambda : ray.rllib.algorithms.r2d2.r2d2.R2D2Config(), postprocess_fn=postprocess_nstep_and_prio, stats_fn=build_q_stats, make_model=build_r2d2_model, action_distribution_fn=get_distribution_inputs_and_class, optimizer_fn=adam_optimizer, extra_action_out_fn=lambda policy: {'q_values': policy.q_values}, compute_gradients_fn=clip_gradients, extra_learn_fetches_fn=lambda policy: {'td_error': policy._td_error}, before_init=setup_early_mixins, before_loss_init=before_loss_init, after_init=setup_late_mixins, mixins=[TargetNetworkMixin, ComputeTDErrorMixin, LearningRateSchedule])