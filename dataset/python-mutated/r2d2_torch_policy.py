"""PyTorch policy class used for R2D2."""
from typing import Dict, Tuple
import gymnasium as gym
import ray
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS, postprocess_nstep_and_prio
from ray.rllib.algorithms.dqn.dqn_torch_policy import adam_optimizer, build_q_model_and_distribution, compute_q_values
from ray.rllib.algorithms.r2d2.r2d2_tf_policy import get_distribution_inputs_and_class
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import LearningRateSchedule, TargetNetworkMixin
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN, apply_grad_clipping, concat_multi_gpu_td_errors, huber_loss, sequence_mask
from ray.rllib.utils.typing import AlgorithmConfigDict, TensorType
(torch, nn) = try_import_torch()
F = None
if nn:
    F = nn.functional

def build_r2d2_model_and_distribution(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> Tuple[ModelV2, TorchDistributionWrapper]:
    if False:
        for i in range(10):
            print('nop')
    "Build q_model and target_model for DQN\n\n    Args:\n        policy: The policy, which will use the model for optimization.\n        obs_space (gym.spaces.Space): The policy's observation space.\n        action_space (gym.spaces.Space): The policy's action space.\n        config (AlgorithmConfigDict):\n\n    Returns:\n        (q_model, TorchCategorical)\n            Note: The target q model will not be returned, just assigned to\n            `policy.target_model`.\n    "
    (model, distribution_cls) = build_q_model_and_distribution(policy, obs_space, action_space, config)
    assert model.get_initial_state() != [] or model.view_requirements.get('state_in_0') is not None, 'R2D2 requires its model to be a recurrent one! Try using `model.use_lstm` or `model.use_attention` in your config to auto-wrap your model with an LSTM- or attention net.'
    return (model, distribution_cls)

def r2d2_loss(policy: Policy, model, _, train_batch: SampleBatch) -> TensorType:
    if False:
        i = 10
        return i + 15
    'Constructs the loss for R2D2TorchPolicy.\n\n    Args:\n        policy: The Policy to calculate the loss for.\n        model (ModelV2): The Model to calculate the loss for.\n        train_batch: The training data.\n\n    Returns:\n        TensorType: A single loss tensor.\n    '
    target_model = policy.target_models[model]
    config = policy.config
    i = 0
    state_batches = []
    while 'state_in_{}'.format(i) in train_batch:
        state_batches.append(train_batch['state_in_{}'.format(i)])
        i += 1
    assert state_batches
    (q, _, _, _) = compute_q_values(policy, model, train_batch, state_batches=state_batches, seq_lens=train_batch.get(SampleBatch.SEQ_LENS), explore=False, is_training=True)
    (q_target, _, _, _) = compute_q_values(policy, target_model, train_batch, state_batches=state_batches, seq_lens=train_batch.get(SampleBatch.SEQ_LENS), explore=False, is_training=True)
    actions = train_batch[SampleBatch.ACTIONS].long()
    dones = train_batch[SampleBatch.TERMINATEDS].float()
    rewards = train_batch[SampleBatch.REWARDS]
    weights = train_batch[PRIO_WEIGHTS]
    B = state_batches[0].shape[0]
    T = q.shape[0] // B
    one_hot_selection = F.one_hot(actions, policy.action_space.n)
    q_selected = torch.sum(torch.where(q > FLOAT_MIN, q, torch.tensor(0.0, device=q.device)) * one_hot_selection, 1)
    if config['double_q']:
        best_actions = torch.argmax(q, dim=1)
    else:
        best_actions = torch.argmax(q_target, dim=1)
    best_actions_one_hot = F.one_hot(best_actions, policy.action_space.n)
    q_target_best = torch.sum(torch.where(q_target > FLOAT_MIN, q_target, torch.tensor(0.0, device=q_target.device)) * best_actions_one_hot, dim=1)
    if config['num_atoms'] > 1:
        raise ValueError('Distributional R2D2 not supported yet!')
    else:
        q_target_best_masked_tp1 = (1.0 - dones) * torch.cat([q_target_best[1:], torch.tensor([0.0], device=q_target_best.device)])
        if config['use_h_function']:
            h_inv = h_inverse(q_target_best_masked_tp1, config['h_function_epsilon'])
            target = h_function(rewards + config['gamma'] ** config['n_step'] * h_inv, config['h_function_epsilon'])
        else:
            target = rewards + config['gamma'] ** config['n_step'] * q_target_best_masked_tp1
        seq_mask = sequence_mask(train_batch[SampleBatch.SEQ_LENS], T)[:, :-1]
        burn_in = policy.config['replay_buffer_config']['replay_burn_in']
        if burn_in > 0 and burn_in < T:
            seq_mask[:, :burn_in] = False
        num_valid = torch.sum(seq_mask)

        def reduce_mean_valid(t):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sum(t[seq_mask]) / num_valid
        q_selected = q_selected.reshape([B, T])[:, :-1]
        td_error = q_selected - target.reshape([B, T])[:, :-1].detach()
        td_error = td_error * seq_mask
        weights = weights.reshape([B, T])[:, :-1]
        total_loss = reduce_mean_valid(weights * huber_loss(td_error))
        model.tower_stats['total_loss'] = total_loss
        model.tower_stats['mean_q'] = reduce_mean_valid(q_selected)
        model.tower_stats['min_q'] = torch.min(q_selected)
        model.tower_stats['max_q'] = torch.max(q_selected)
        model.tower_stats['mean_td_error'] = reduce_mean_valid(td_error)
        model.tower_stats['td_error'] = torch.mean(td_error, dim=-1)
    return total_loss

def h_function(x, epsilon=1.0):
    if False:
        i = 10
        return i + 15
    'h-function to normalize target Qs, described in the paper [1].\n\n    h(x) = sign(x) * [sqrt(abs(x) + 1) - 1] + epsilon * x\n\n    Used in [1] in combination with h_inverse:\n      targets = h(r + gamma * h_inverse(Q^))\n    '
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + epsilon * x

def h_inverse(x, epsilon=1.0):
    if False:
        i = 10
        return i + 15
    'Inverse if the above h-function, described in the paper [1].\n\n    If x > 0.0:\n    h-1(x) = [2eps * x + (2eps + 1) - sqrt(4eps x + (2eps + 1)^2)] /\n        (2 * eps^2)\n\n    If x < 0.0:\n    h-1(x) = [2eps * x + (2eps + 1) + sqrt(-4eps x + (2eps + 1)^2)] /\n        (2 * eps^2)\n    '
    two_epsilon = epsilon * 2
    if_x_pos = (two_epsilon * x + (two_epsilon + 1.0) - torch.sqrt(4.0 * epsilon * x + (two_epsilon + 1.0) ** 2)) / (2.0 * epsilon ** 2)
    if_x_neg = (two_epsilon * x - (two_epsilon + 1.0) + torch.sqrt(-4.0 * epsilon * x + (two_epsilon + 1.0) ** 2)) / (2.0 * epsilon ** 2)
    return torch.where(x < 0.0, if_x_neg, if_x_pos)

class ComputeTDErrorMixin:
    """Assign the `compute_td_error` method to the R2D2TorchPolicy

    This allows us to prioritize on the worker side.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')

        def compute_td_error(obs_t, act_t, rew_t, obs_tp1, terminateds_mask, importance_weights):
            if False:
                i = 10
                return i + 15
            input_dict = self._lazy_tensor_dict({SampleBatch.CUR_OBS: obs_t})
            input_dict[SampleBatch.ACTIONS] = act_t
            input_dict[SampleBatch.REWARDS] = rew_t
            input_dict[SampleBatch.NEXT_OBS] = obs_tp1
            input_dict[SampleBatch.TERMINATEDS] = terminateds_mask
            input_dict[PRIO_WEIGHTS] = importance_weights
            r2d2_loss(self, self.model, None, input_dict)
            return self.model.tower_stats['td_error']
        self.compute_td_error = compute_td_error

def build_q_stats(policy: Policy, batch: SampleBatch) -> Dict[str, TensorType]:
    if False:
        while True:
            i = 10
    return {'cur_lr': policy.cur_lr, 'total_loss': torch.mean(torch.stack(policy.get_tower_stats('total_loss'))), 'mean_q': torch.mean(torch.stack(policy.get_tower_stats('mean_q'))), 'min_q': torch.mean(torch.stack(policy.get_tower_stats('min_q'))), 'max_q': torch.mean(torch.stack(policy.get_tower_stats('max_q'))), 'mean_td_error': torch.mean(torch.stack(policy.get_tower_stats('mean_td_error')))}

def setup_early_mixins(policy: Policy, obs_space, action_space, config: AlgorithmConfigDict) -> None:
    if False:
        for i in range(10):
            print('nop')
    LearningRateSchedule.__init__(policy, config['lr'], config['lr_schedule'])

def before_loss_init(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> None:
    if False:
        while True:
            i = 10
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy)

def grad_process_and_td_error_fn(policy: Policy, optimizer: 'torch.optim.Optimizer', loss: TensorType) -> Dict[str, TensorType]:
    if False:
        while True:
            i = 10
    return apply_grad_clipping(policy, optimizer, loss)

def extra_action_out_fn(policy: Policy, input_dict, state_batches, model, action_dist) -> Dict[str, TensorType]:
    if False:
        return 10
    return {'q_values': policy.q_values}
R2D2TorchPolicy = build_policy_class(name='R2D2TorchPolicy', framework='torch', loss_fn=r2d2_loss, get_default_config=lambda : ray.rllib.algorithms.r2d2.r2d2.R2D2Config(), make_model_and_action_dist=build_r2d2_model_and_distribution, action_distribution_fn=get_distribution_inputs_and_class, stats_fn=build_q_stats, postprocess_fn=postprocess_nstep_and_prio, optimizer_fn=adam_optimizer, extra_grad_process_fn=grad_process_and_td_error_fn, extra_learn_fetches_fn=concat_multi_gpu_td_errors, extra_action_out_fn=extra_action_out_fn, before_init=setup_early_mixins, before_loss_init=before_loss_init, mixins=[TargetNetworkMixin, ComputeTDErrorMixin, LearningRateSchedule])