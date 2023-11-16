import gymnasium as gym
import numpy as np
from typing import List, Optional, Tuple, Type, Union
import copy
import ray
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.algorithms.sac import SACTorchPolicy
from ray.rllib.algorithms.sac.rnnsac_torch_model import RNNSACTorchModel
from ray.rllib.algorithms.sac.sac_torch_policy import _get_dist_class
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import huber_loss, sequence_mask
from ray.rllib.utils.typing import ModelInputDict, TensorType, AlgorithmConfigDict
(torch, nn) = try_import_torch()
F = None
if nn:
    F = nn.functional

def build_rnnsac_model(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> ModelV2:
    if False:
        i = 10
        return i + 15
    "Constructs the necessary ModelV2 for the Policy and returns it.\n\n    Args:\n        policy: The TFPolicy that will use the models.\n        obs_space (gym.spaces.Space): The observation space.\n        action_space (gym.spaces.Space): The action space.\n        config: The SAC's config dict.\n\n    Returns:\n        ModelV2: The ModelV2 to be used by the Policy. Note: An additional\n            target model will be created in this function and assigned to\n            `policy.target_model`.\n    "
    num_outputs = int(np.product(obs_space.shape))
    policy_model_config = copy.deepcopy(config['model'])
    policy_model_config.update(config['policy_model_config'])
    q_model_config = copy.deepcopy(config['model'])
    q_model_config.update(config['q_model_config'])
    default_model_cls = RNNSACTorchModel
    model = ModelCatalog.get_model_v2(obs_space=obs_space, action_space=action_space, num_outputs=num_outputs, model_config=config['model'], framework=config['framework'], default_model=default_model_cls, name='sac_model', policy_model_config=policy_model_config, q_model_config=q_model_config, twin_q=config['twin_q'], initial_alpha=config['initial_alpha'], target_entropy=config['target_entropy'])
    assert isinstance(model, default_model_cls)
    policy.target_model = ModelCatalog.get_model_v2(obs_space=obs_space, action_space=action_space, num_outputs=num_outputs, model_config=config['model'], framework=config['framework'], default_model=default_model_cls, name='target_sac_model', policy_model_config=policy_model_config, q_model_config=q_model_config, twin_q=config['twin_q'], initial_alpha=config['initial_alpha'], target_entropy=config['target_entropy'])
    assert isinstance(policy.target_model, default_model_cls)
    return model

def build_sac_model_and_action_dist(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
    if False:
        while True:
            i = 10
    "Constructs the necessary ModelV2 and action dist class for the Policy.\n\n    Args:\n        policy: The TFPolicy that will use the models.\n        obs_space (gym.spaces.Space): The observation space.\n        action_space (gym.spaces.Space): The action space.\n        config: The SAC's config dict.\n\n    Returns:\n        ModelV2: The ModelV2 to be used by the Policy. Note: An additional\n            target model will be created in this function and assigned to\n            `policy.target_model`.\n    "
    model = build_rnnsac_model(policy, obs_space, action_space, config)
    assert model.get_initial_state() != [], 'RNNSAC requires its model to be a recurrent one!'
    action_dist_class = _get_dist_class(policy, config, action_space)
    return (model, action_dist_class)

def action_distribution_fn(policy: Policy, model: ModelV2, input_dict: ModelInputDict, *, state_batches: Optional[List[TensorType]]=None, seq_lens: Optional[TensorType]=None, prev_action_batch: Optional[TensorType]=None, prev_reward_batch=None, explore: Optional[bool]=None, timestep: Optional[int]=None, is_training: Optional[bool]=None) -> Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
    if False:
        while True:
            i = 10
    'The action distribution function to be used the algorithm.\n\n    An action distribution function is used to customize the choice of action\n    distribution class and the resulting action distribution inputs (to\n    parameterize the distribution object).\n    After parameterizing the distribution, a `sample()` call\n    will be made on it to generate actions.\n\n    Args:\n        policy: The Policy being queried for actions and calling this\n            function.\n        model (TorchModelV2): The SAC specific Model to use to generate the\n            distribution inputs (see sac_tf|torch_model.py). Must support the\n            `get_action_model_outputs` method.\n        input_dict: The input-dict to be used for the model\n            call.\n        state_batches (Optional[List[TensorType]]): The list of internal state\n            tensor batches.\n        seq_lens (Optional[TensorType]): The tensor of sequence lengths used\n            in RNNs.\n        prev_action_batch (Optional[TensorType]): Optional batch of prev\n            actions used by the model.\n        prev_reward_batch (Optional[TensorType]): Optional batch of prev\n            rewards used by the model.\n        explore (Optional[bool]): Whether to activate exploration or not. If\n            None, use value of `config.explore`.\n        timestep (Optional[int]): An optional timestep.\n        is_training (Optional[bool]): An optional is-training flag.\n\n    Returns:\n        Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:\n            The dist inputs, dist class, and a list of internal state outputs\n            (in the RNN case).\n    '
    (model_out, state_in) = model(input_dict, state_batches, seq_lens)
    states_in = model.select_state(state_in, ['policy', 'q', 'twin_q'])
    (action_dist_inputs, policy_state_out) = model.get_action_model_outputs(model_out, states_in['policy'], seq_lens)
    (_, q_state_out) = model.get_q_values(model_out, states_in['q'], seq_lens)
    if model.twin_q_net:
        (_, twin_q_state_out) = model.get_twin_q_values(model_out, states_in['twin_q'], seq_lens)
    else:
        twin_q_state_out = []
    action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
    states_out = policy_state_out + q_state_out + twin_q_state_out
    return (action_dist_inputs, action_dist_class, states_out)

def actor_critic_loss(policy: Policy, model: ModelV2, dist_class: Type[TorchDistributionWrapper], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    if False:
        i = 10
        return i + 15
    'Constructs the loss for the Soft Actor Critic.\n\n    Args:\n        policy: The Policy to calculate the loss for.\n        model (ModelV2): The Model to calculate the loss for.\n        dist_class (Type[TorchDistributionWrapper]: The action distr. class.\n        train_batch: The training data.\n\n    Returns:\n        Union[TensorType, List[TensorType]]: A single loss tensor or a list\n            of loss tensors.\n    '
    target_model = policy.target_models[model]
    deterministic = policy.config['_deterministic_loss']
    i = 0
    state_batches = []
    while 'state_in_{}'.format(i) in train_batch:
        state_batches.append(train_batch['state_in_{}'.format(i)])
        i += 1
    assert state_batches
    seq_lens = train_batch.get(SampleBatch.SEQ_LENS)
    (model_out_t, state_in_t) = model(SampleBatch(obs=train_batch[SampleBatch.CUR_OBS], prev_actions=train_batch[SampleBatch.PREV_ACTIONS], prev_rewards=train_batch[SampleBatch.PREV_REWARDS], _is_training=True), state_batches, seq_lens)
    states_in_t = model.select_state(state_in_t, ['policy', 'q', 'twin_q'])
    (model_out_tp1, state_in_tp1) = model(SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], prev_actions=train_batch[SampleBatch.ACTIONS], prev_rewards=train_batch[SampleBatch.REWARDS], _is_training=True), state_batches, seq_lens)
    states_in_tp1 = model.select_state(state_in_tp1, ['policy', 'q', 'twin_q'])
    (target_model_out_tp1, target_state_in_tp1) = target_model(SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], prev_actions=train_batch[SampleBatch.ACTIONS], prev_rewards=train_batch[SampleBatch.REWARDS], _is_training=True), state_batches, seq_lens)
    target_states_in_tp1 = target_model.select_state(state_in_tp1, ['policy', 'q', 'twin_q'])
    alpha = torch.exp(model.log_alpha)
    if model.discrete:
        (action_dist_inputs_t, _) = model.get_action_model_outputs(model_out_t, states_in_t['policy'], seq_lens)
        log_pis_t = F.log_softmax(action_dist_inputs_t, dim=-1)
        policy_t = torch.exp(log_pis_t)
        (action_dist_inputs_tp1, _) = model.get_action_model_outputs(model_out_tp1, states_in_tp1['policy'], seq_lens)
        log_pis_tp1 = F.log_softmax(action_dist_inputs_tp1, -1)
        policy_tp1 = torch.exp(log_pis_tp1)
        (q_t, _) = model.get_q_values(model_out_t, states_in_t['q'], seq_lens)
        (q_tp1, _) = target_model.get_q_values(target_model_out_tp1, target_states_in_tp1['q'], seq_lens)
        if policy.config['twin_q']:
            (twin_q_t, _) = model.get_twin_q_values(model_out_t, states_in_t['twin_q'], seq_lens)
            (twin_q_tp1, _) = target_model.get_twin_q_values(target_model_out_tp1, target_states_in_tp1['twin_q'], seq_lens)
            q_tp1 = torch.min(q_tp1, twin_q_tp1)
        q_tp1 -= alpha * log_pis_tp1
        one_hot = F.one_hot(train_batch[SampleBatch.ACTIONS].long(), num_classes=q_t.size()[-1])
        q_t_selected = torch.sum(q_t * one_hot, dim=-1)
        if policy.config['twin_q']:
            twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)
        q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.TERMINATEDS].float()) * q_tp1_best
    else:
        action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
        (action_dist_inputs_t, _) = model.get_action_model_outputs(model_out_t, states_in_t['policy'], seq_lens)
        action_dist_t = action_dist_class(action_dist_inputs_t, model)
        policy_t = action_dist_t.sample() if not deterministic else action_dist_t.deterministic_sample()
        log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)
        (action_dist_inputs_t, _) = model.get_action_model_outputs(model_out_tp1, states_in_tp1['policy'], seq_lens)
        action_dist_tp1 = action_dist_class(action_dist_inputs_t, model)
        policy_tp1 = action_dist_tp1.sample() if not deterministic else action_dist_tp1.deterministic_sample()
        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)
        (q_t, _) = model.get_q_values(model_out_t, states_in_t['q'], seq_lens, train_batch[SampleBatch.ACTIONS])
        if policy.config['twin_q']:
            (twin_q_t, _) = model.get_twin_q_values(model_out_t, states_in_t['twin_q'], seq_lens, train_batch[SampleBatch.ACTIONS])
        (q_t_det_policy, _) = model.get_q_values(model_out_t, states_in_t['q'], seq_lens, policy_t)
        if policy.config['twin_q']:
            (twin_q_t_det_policy, _) = model.get_twin_q_values(model_out_t, states_in_t['twin_q'], seq_lens, policy_t)
            q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)
        q_tp1 = target_model.get_q_values(target_model_out_tp1, target_states_in_tp1['q'], seq_lens, policy_tp1)
        if policy.config['twin_q']:
            (twin_q_tp1, _) = target_model.get_twin_q_values(target_model_out_tp1, target_states_in_tp1['twin_q'], seq_lens, policy_tp1)
            q_tp1 = torch.min(q_tp1, twin_q_tp1)
        q_t_selected = torch.squeeze(q_t, dim=-1)
        if policy.config['twin_q']:
            twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)
        q_tp1 -= alpha * log_pis_tp1
        q_tp1_best = torch.squeeze(input=q_tp1, dim=-1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.TERMINATEDS].float()) * q_tp1_best
    q_t_selected_target = (train_batch[SampleBatch.REWARDS] + policy.config['gamma'] ** policy.config['n_step'] * q_tp1_best_masked).detach()
    B = state_batches[0].shape[0]
    T = q_t_selected.shape[0] // B
    seq_mask = sequence_mask(train_batch[SampleBatch.SEQ_LENS], T)
    burn_in = policy.config['burn_in']
    if burn_in > 0 and burn_in < T:
        seq_mask[:, :burn_in] = False
    seq_mask = seq_mask.reshape(-1)
    num_valid = torch.sum(seq_mask)

    def reduce_mean_valid(t):
        if False:
            for i in range(10):
                print('nop')
        return torch.sum(t[seq_mask]) / num_valid
    base_td_error = torch.abs(q_t_selected - q_t_selected_target)
    if policy.config['twin_q']:
        twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error
    critic_loss = [reduce_mean_valid(train_batch[PRIO_WEIGHTS] * huber_loss(base_td_error))]
    if policy.config['twin_q']:
        critic_loss.append(reduce_mean_valid(train_batch[PRIO_WEIGHTS] * huber_loss(twin_td_error)))
    td_error = td_error * seq_mask
    if model.discrete:
        weighted_log_alpha_loss = policy_t.detach() * (-model.log_alpha * (log_pis_t + model.target_entropy).detach())
        alpha_loss = reduce_mean_valid(torch.sum(weighted_log_alpha_loss, dim=-1))
        actor_loss = reduce_mean_valid(torch.sum(torch.mul(policy_t, alpha.detach() * log_pis_t - q_t.detach()), dim=-1))
    else:
        alpha_loss = -reduce_mean_valid(model.log_alpha * (log_pis_t + model.target_entropy).detach())
        actor_loss = reduce_mean_valid(alpha.detach() * log_pis_t - q_t_det_policy)
    model.tower_stats['q_t'] = q_t * seq_mask[..., None]
    model.tower_stats['policy_t'] = policy_t * seq_mask[..., None]
    model.tower_stats['log_pis_t'] = log_pis_t * seq_mask[..., None]
    model.tower_stats['actor_loss'] = actor_loss
    model.tower_stats['critic_loss'] = critic_loss
    model.tower_stats['alpha_loss'] = alpha_loss
    model.tower_stats['td_error'] = torch.mean(td_error.reshape([-1, T]), dim=-1)
    return tuple([actor_loss] + critic_loss + [alpha_loss])
RNNSACTorchPolicy = SACTorchPolicy.with_updates(name='RNNSACPolicy', get_default_config=lambda : ray.rllib.algorithms.sac.rnnsac.RNNSACConfig(), action_distribution_fn=action_distribution_fn, make_model_and_action_dist=build_sac_model_and_action_dist, loss_fn=actor_critic_loss)