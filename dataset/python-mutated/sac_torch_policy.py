"""
PyTorch policy class used for SAC.
"""
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import logging
import tree
from typing import Dict, List, Optional, Tuple, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.sac.sac_tf_policy import build_sac_model, postprocess_trajectory, validate_spaces
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, TorchDistributionWrapper, TorchDirichlet, TorchSquashedGaussian, TorchDiagGaussian, TorchBeta
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.policy.torch_mixins import TargetNetworkMixin
from ray.rllib.utils.torch_utils import apply_grad_clipping, concat_multi_gpu_td_errors, huber_loss
from ray.rllib.utils.typing import LocalOptimizer, ModelInputDict, TensorType, AlgorithmConfigDict
(torch, nn) = try_import_torch()
F = nn.functional
logger = logging.getLogger(__name__)

def _get_dist_class(policy: Policy, config: AlgorithmConfigDict, action_space: gym.spaces.Space) -> Type[TorchDistributionWrapper]:
    if False:
        print('Hello World!')
    "Helper function to return a dist class based on config and action space.\n\n    Args:\n        policy: The policy for which to return the action\n            dist class.\n        config: The Algorithm's config dict.\n        action_space (gym.spaces.Space): The action space used.\n\n    Returns:\n        Type[TFActionDistribution]: A TF distribution class.\n    "
    if hasattr(policy, 'dist_class') and policy.dist_class is not None:
        return policy.dist_class
    elif config['model'].get('custom_action_dist'):
        (action_dist_class, _) = ModelCatalog.get_action_dist(action_space, config['model'], framework='torch')
        return action_dist_class
    elif isinstance(action_space, Discrete):
        return TorchCategorical
    elif isinstance(action_space, Simplex):
        return TorchDirichlet
    else:
        assert isinstance(action_space, Box)
        if config['normalize_actions']:
            return TorchSquashedGaussian if not config['_use_beta_distribution'] else TorchBeta
        else:
            return TorchDiagGaussian

def build_sac_model_and_action_dist(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
    if False:
        print('Hello World!')
    'Constructs the necessary ModelV2 and action dist class for the Policy.\n\n    Args:\n        policy: The TFPolicy that will use the models.\n        obs_space (gym.spaces.Space): The observation space.\n        action_space (gym.spaces.Space): The action space.\n        config: The SACConfig object.\n\n    Returns:\n        ModelV2: The ModelV2 to be used by the Policy. Note: An additional\n            target model will be created in this function and assigned to\n            `policy.target_model`.\n    '
    model = build_sac_model(policy, obs_space, action_space, config)
    action_dist_class = _get_dist_class(policy, config, action_space)
    return (model, action_dist_class)

def action_distribution_fn(policy: Policy, model: ModelV2, input_dict: ModelInputDict, *, state_batches: Optional[List[TensorType]]=None, seq_lens: Optional[TensorType]=None, prev_action_batch: Optional[TensorType]=None, prev_reward_batch=None, explore: Optional[bool]=None, timestep: Optional[int]=None, is_training: Optional[bool]=None) -> Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
    if False:
        i = 10
        return i + 15
    'The action distribution function to be used the algorithm.\n\n    An action distribution function is used to customize the choice of action\n    distribution class and the resulting action distribution inputs (to\n    parameterize the distribution object).\n    After parameterizing the distribution, a `sample()` call\n    will be made on it to generate actions.\n\n    Args:\n        policy: The Policy being queried for actions and calling this\n            function.\n        model (TorchModelV2): The SAC specific model to use to generate the\n            distribution inputs (see sac_tf|torch_model.py). Must support the\n            `get_action_model_outputs` method.\n        input_dict: The input-dict to be used for the model\n            call.\n        state_batches (Optional[List[TensorType]]): The list of internal state\n            tensor batches.\n        seq_lens (Optional[TensorType]): The tensor of sequence lengths used\n            in RNNs.\n        prev_action_batch (Optional[TensorType]): Optional batch of prev\n            actions used by the model.\n        prev_reward_batch (Optional[TensorType]): Optional batch of prev\n            rewards used by the model.\n        explore (Optional[bool]): Whether to activate exploration or not. If\n            None, use value of `config.explore`.\n        timestep (Optional[int]): An optional timestep.\n        is_training (Optional[bool]): An optional is-training flag.\n\n    Returns:\n        Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:\n            The dist inputs, dist class, and a list of internal state outputs\n            (in the RNN case).\n    '
    (model_out, _) = model(input_dict, [], None)
    (action_dist_inputs, _) = model.get_action_model_outputs(model_out)
    action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
    return (action_dist_inputs, action_dist_class, [])

def actor_critic_loss(policy: Policy, model: ModelV2, dist_class: Type[TorchDistributionWrapper], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    if False:
        print('Hello World!')
    'Constructs the loss for the Soft Actor Critic.\n\n    Args:\n        policy: The Policy to calculate the loss for.\n        model (ModelV2): The Model to calculate the loss for.\n        dist_class (Type[TorchDistributionWrapper]: The action distr. class.\n        train_batch: The training data.\n\n    Returns:\n        Union[TensorType, List[TensorType]]: A single loss tensor or a list\n            of loss tensors.\n    '
    target_model = policy.target_models[model]
    deterministic = policy.config['_deterministic_loss']
    (model_out_t, _) = model(SampleBatch(obs=train_batch[SampleBatch.CUR_OBS], _is_training=True), [], None)
    (model_out_tp1, _) = model(SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True), [], None)
    (target_model_out_tp1, _) = target_model(SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True), [], None)
    alpha = torch.exp(model.log_alpha)
    if model.discrete:
        (action_dist_inputs_t, _) = model.get_action_model_outputs(model_out_t)
        log_pis_t = F.log_softmax(action_dist_inputs_t, dim=-1)
        policy_t = torch.exp(log_pis_t)
        (action_dist_inputs_tp1, _) = model.get_action_model_outputs(model_out_tp1)
        log_pis_tp1 = F.log_softmax(action_dist_inputs_tp1, -1)
        policy_tp1 = torch.exp(log_pis_tp1)
        (q_t, _) = model.get_q_values(model_out_t)
        (q_tp1, _) = target_model.get_q_values(target_model_out_tp1)
        if policy.config['twin_q']:
            (twin_q_t, _) = model.get_twin_q_values(model_out_t)
            (twin_q_tp1, _) = target_model.get_twin_q_values(target_model_out_tp1)
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
        (action_dist_inputs_t, _) = model.get_action_model_outputs(model_out_t)
        action_dist_t = action_dist_class(action_dist_inputs_t, model)
        policy_t = action_dist_t.sample() if not deterministic else action_dist_t.deterministic_sample()
        log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)
        (action_dist_inputs_tp1, _) = model.get_action_model_outputs(model_out_tp1)
        action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, model)
        policy_tp1 = action_dist_tp1.sample() if not deterministic else action_dist_tp1.deterministic_sample()
        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)
        (q_t, _) = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        if policy.config['twin_q']:
            (twin_q_t, _) = model.get_twin_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        (q_t_det_policy, _) = model.get_q_values(model_out_t, policy_t)
        if policy.config['twin_q']:
            (twin_q_t_det_policy, _) = model.get_twin_q_values(model_out_t, policy_t)
            q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)
        (q_tp1, _) = target_model.get_q_values(target_model_out_tp1, policy_tp1)
        if policy.config['twin_q']:
            (twin_q_tp1, _) = target_model.get_twin_q_values(target_model_out_tp1, policy_tp1)
            q_tp1 = torch.min(q_tp1, twin_q_tp1)
        q_t_selected = torch.squeeze(q_t, dim=-1)
        if policy.config['twin_q']:
            twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)
        q_tp1 -= alpha * log_pis_tp1
        q_tp1_best = torch.squeeze(input=q_tp1, dim=-1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.TERMINATEDS].float()) * q_tp1_best
    q_t_selected_target = (train_batch[SampleBatch.REWARDS] + policy.config['gamma'] ** policy.config['n_step'] * q_tp1_best_masked).detach()
    base_td_error = torch.abs(q_t_selected - q_t_selected_target)
    if policy.config['twin_q']:
        twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error
    critic_loss = [torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(base_td_error))]
    if policy.config['twin_q']:
        critic_loss.append(torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(twin_td_error)))
    if model.discrete:
        weighted_log_alpha_loss = policy_t.detach() * (-model.log_alpha * (log_pis_t + model.target_entropy).detach())
        alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))
        actor_loss = torch.mean(torch.sum(torch.mul(policy_t, alpha.detach() * log_pis_t - q_t.detach()), dim=-1))
    else:
        alpha_loss = -torch.mean(model.log_alpha * (log_pis_t + model.target_entropy).detach())
        actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_det_policy)
    model.tower_stats['q_t'] = q_t
    model.tower_stats['policy_t'] = policy_t
    model.tower_stats['log_pis_t'] = log_pis_t
    model.tower_stats['actor_loss'] = actor_loss
    model.tower_stats['critic_loss'] = critic_loss
    model.tower_stats['alpha_loss'] = alpha_loss
    model.tower_stats['td_error'] = td_error
    return tuple([actor_loss] + critic_loss + [alpha_loss])

def stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    if False:
        return 10
    'Stats function for SAC. Returns a dict with important loss stats.\n\n    Args:\n        policy: The Policy to generate stats for.\n        train_batch: The SampleBatch (already) used for training.\n\n    Returns:\n        Dict[str, TensorType]: The stats dict.\n    '
    q_t = torch.stack(policy.get_tower_stats('q_t'))
    return {'actor_loss': torch.mean(torch.stack(policy.get_tower_stats('actor_loss'))), 'critic_loss': torch.mean(torch.stack(tree.flatten(policy.get_tower_stats('critic_loss')))), 'alpha_loss': torch.mean(torch.stack(policy.get_tower_stats('alpha_loss'))), 'alpha_value': torch.exp(policy.model.log_alpha), 'log_alpha_value': policy.model.log_alpha, 'target_entropy': policy.model.target_entropy, 'policy_t': torch.mean(torch.stack(policy.get_tower_stats('policy_t'))), 'mean_q': torch.mean(q_t), 'max_q': torch.max(q_t), 'min_q': torch.min(q_t)}

def optimizer_fn(policy: Policy, config: AlgorithmConfigDict) -> Tuple[LocalOptimizer]:
    if False:
        print('Hello World!')
    "Creates all necessary optimizers for SAC learning.\n\n    The 3 or 4 (twin_q=True) optimizers returned here correspond to the\n    number of loss terms returned by the loss function.\n\n    Args:\n        policy: The policy object to be trained.\n        config: The Algorithm's config dict.\n\n    Returns:\n        Tuple[LocalOptimizer]: The local optimizers to use for policy training.\n    "
    policy.actor_optim = torch.optim.Adam(params=policy.model.policy_variables(), lr=config['optimization']['actor_learning_rate'], eps=1e-07)
    critic_split = len(policy.model.q_variables())
    if config['twin_q']:
        critic_split //= 2
    policy.critic_optims = [torch.optim.Adam(params=policy.model.q_variables()[:critic_split], lr=config['optimization']['critic_learning_rate'], eps=1e-07)]
    if config['twin_q']:
        policy.critic_optims.append(torch.optim.Adam(params=policy.model.q_variables()[critic_split:], lr=config['optimization']['critic_learning_rate'], eps=1e-07))
    policy.alpha_optim = torch.optim.Adam(params=[policy.model.log_alpha], lr=config['optimization']['entropy_learning_rate'], eps=1e-07)
    return tuple([policy.actor_optim] + policy.critic_optims + [policy.alpha_optim])

class ComputeTDErrorMixin:
    """Mixin class calculating TD-error (part of critic loss) per batch item.

    - Adds `policy.compute_td_error()` method for TD-error calculation from a
      batch of observations/actions/rewards/etc..
    """

    def __init__(self):
        if False:
            while True:
                i = 10

        def compute_td_error(obs_t, act_t, rew_t, obs_tp1, terminateds_mask, importance_weights):
            if False:
                i = 10
                return i + 15
            input_dict = self._lazy_tensor_dict({SampleBatch.CUR_OBS: obs_t, SampleBatch.ACTIONS: act_t, SampleBatch.REWARDS: rew_t, SampleBatch.NEXT_OBS: obs_tp1, SampleBatch.TERMINATEDS: terminateds_mask, PRIO_WEIGHTS: importance_weights})
            actor_critic_loss(self, self.model, None, input_dict)
            return self.model.tower_stats['td_error']
        self.compute_td_error = compute_td_error

def setup_late_mixins(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> None:
    if False:
        return 10
    'Call mixin classes\' constructors after Policy initialization.\n\n    - Moves the target model(s) to the GPU, if necessary.\n    - Adds the `compute_td_error` method to the given policy.\n    Calling `compute_td_error` with batch data will re-calculate the loss\n    on that batch AND return the per-batch-item TD-error for prioritized\n    replay buffer record weight updating (in case a prioritized replay buffer\n    is used).\n    - Also adds the `update_target` method to the given policy.\n    Calling `update_target` updates all target Q-networks\' weights from their\n    respective "main" Q-metworks, based on tau (smooth, partial updating).\n\n    Args:\n        policy: The Policy object.\n        obs_space (gym.spaces.Space): The Policy\'s observation space.\n        action_space (gym.spaces.Space): The Policy\'s action space.\n        config: The Policy\'s config.\n    '
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy)
SACTorchPolicy = build_policy_class(name='SACTorchPolicy', framework='torch', loss_fn=actor_critic_loss, get_default_config=lambda : ray.rllib.algorithms.sac.sac.SACConfig(), stats_fn=stats, postprocess_fn=postprocess_trajectory, extra_grad_process_fn=apply_grad_clipping, optimizer_fn=optimizer_fn, validate_spaces=validate_spaces, before_loss_init=setup_late_mixins, make_model_and_action_dist=build_sac_model_and_action_dist, extra_learn_fetches_fn=concat_multi_gpu_td_errors, mixins=[TargetNetworkMixin, ComputeTDErrorMixin], action_distribution_fn=action_distribution_fn)