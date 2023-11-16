import gymnasium as gym
import logging
import numpy as np
from typing import Dict, List, Optional, Type, Union
import ray
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import compute_bootstrap_value
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import EntropyCoeffSchedule, LearningRateSchedule, ValueNetworkMixin
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import apply_grad_clipping, explained_variance, global_norm, sequence_mask
from ray.rllib.utils.typing import TensorType
(torch, nn) = try_import_torch()
logger = logging.getLogger(__name__)

class VTraceLoss:

    def __init__(self, actions, actions_logp, actions_entropy, dones, behaviour_action_logp, behaviour_logits, target_logits, discount, rewards, values, bootstrap_value, dist_class, model, valid_mask, config, vf_loss_coeff=0.5, entropy_coeff=0.01, clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):
        if False:
            while True:
                i = 10
        'Policy gradient loss with vtrace importance weighting.\n\n        VTraceLoss takes tensors of shape [T, B, ...], where `B` is the\n        batch_size. The reason we need to know `B` is for V-trace to properly\n        handle episode cut boundaries.\n\n        Args:\n            actions: An int|float32 tensor of shape [T, B, ACTION_SPACE].\n            actions_logp: A float32 tensor of shape [T, B].\n            actions_entropy: A float32 tensor of shape [T, B].\n            dones: A bool tensor of shape [T, B].\n            behaviour_action_logp: Tensor of shape [T, B].\n            behaviour_logits: A list with length of ACTION_SPACE of float32\n                tensors of shapes\n                [T, B, ACTION_SPACE[0]],\n                ...,\n                [T, B, ACTION_SPACE[-1]]\n            target_logits: A list with length of ACTION_SPACE of float32\n                tensors of shapes\n                [T, B, ACTION_SPACE[0]],\n                ...,\n                [T, B, ACTION_SPACE[-1]]\n            discount: A float32 scalar.\n            rewards: A float32 tensor of shape [T, B].\n            values: A float32 tensor of shape [T, B].\n            bootstrap_value: A float32 tensor of shape [B].\n            dist_class: action distribution class for logits.\n            valid_mask: A bool tensor of valid RNN input elements (#2992).\n            config: Algorithm config dict.\n        '
        import ray.rllib.algorithms.impala.vtrace_torch as vtrace
        if valid_mask is None:
            valid_mask = torch.ones_like(actions_logp)
        device = behaviour_action_logp[0].device
        self.vtrace_returns = vtrace.multi_from_logits(behaviour_action_log_probs=behaviour_action_logp, behaviour_policy_logits=behaviour_logits, target_policy_logits=target_logits, actions=torch.unbind(actions, dim=2), discounts=(1.0 - dones.float()) * discount, rewards=rewards, values=values, bootstrap_value=bootstrap_value, dist_class=dist_class, model=model, clip_rho_threshold=clip_rho_threshold, clip_pg_rho_threshold=clip_pg_rho_threshold)
        self.value_targets = self.vtrace_returns.vs.to(device)
        self.pi_loss = -torch.sum(actions_logp * self.vtrace_returns.pg_advantages.to(device) * valid_mask)
        delta = (values - self.value_targets) * valid_mask
        self.vf_loss = 0.5 * torch.sum(torch.pow(delta, 2.0))
        self.entropy = torch.sum(actions_entropy * valid_mask)
        self.mean_entropy = self.entropy / torch.sum(valid_mask)
        self.total_loss = self.pi_loss - self.entropy * entropy_coeff
        self.loss_wo_vf = self.total_loss
        if not config['_separate_vf_optimizer']:
            self.total_loss += self.vf_loss * vf_loss_coeff

def make_time_major(policy, seq_lens, tensor):
    if False:
        return 10
    'Swaps batch and trajectory axis.\n\n    Args:\n        policy: Policy reference\n        seq_lens: Sequence lengths if recurrent or None\n        tensor: A tensor or list of tensors to reshape.\n\n    Returns:\n        res: A tensor with swapped axes or a list of tensors with\n        swapped axes.\n    '
    if isinstance(tensor, (list, tuple)):
        return [make_time_major(policy, seq_lens, t) for t in tensor]
    if policy.is_recurrent():
        B = seq_lens.shape[0]
        T = tensor.shape[0] // B
    else:
        T = policy.config['rollout_fragment_length']
        B = tensor.shape[0] // T
    rs = torch.reshape(tensor, [B, T] + list(tensor.shape[1:]))
    res = torch.transpose(rs, 1, 0)
    return res

class VTraceOptimizer:
    """Optimizer function for VTrace torch policies."""

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    @override(TorchPolicyV2)
    def optimizer(self) -> Union[List['torch.optim.Optimizer'], 'torch.optim.Optimizer']:
        if False:
            for i in range(10):
                print('nop')
        if self.config['_separate_vf_optimizer']:
            dummy_batch = self._lazy_tensor_dict(self._get_dummy_batch_from_view_requirements())
            for param in self.model.parameters():
                param.grad = None
            out = self.model(dummy_batch)
            torch.sum(out[0]).backward()
            policy_params = []
            value_params = []
            for param in self.model.parameters():
                if param.grad is None:
                    value_params.append(param)
                else:
                    policy_params.append(param)
            if self.config['opt_type'] == 'adam':
                return (torch.optim.Adam(params=policy_params, lr=self.cur_lr), torch.optim.Adam(params=value_params, lr=self.config['_lr_vf']))
            else:
                raise NotImplementedError
        if self.config['opt_type'] == 'adam':
            return torch.optim.Adam(params=self.model.parameters(), lr=self.cur_lr)
        else:
            return torch.optim.RMSprop(params=self.model.parameters(), lr=self.cur_lr, weight_decay=self.config['decay'], momentum=self.config['momentum'], eps=self.config['epsilon'])

class ImpalaTorchPolicy(VTraceOptimizer, LearningRateSchedule, EntropyCoeffSchedule, ValueNetworkMixin, TorchPolicyV2):
    """PyTorch policy class used with Impala."""

    def __init__(self, observation_space, action_space, config):
        if False:
            for i in range(10):
                print('nop')
        config = dict(ray.rllib.algorithms.impala.impala.ImpalaConfig().to_dict(), **config)
        if not config.get('_enable_new_api_stack'):
            VTraceOptimizer.__init__(self)
            LearningRateSchedule.__init__(self, config['lr'], config['lr_schedule'])
            EntropyCoeffSchedule.__init__(self, config['entropy_coeff'], config['entropy_coeff_schedule'])
        TorchPolicyV2.__init__(self, observation_space, action_space, config, max_seq_len=config['model']['max_seq_len'])
        ValueNetworkMixin.__init__(self, config)
        self._initialize_loss_from_dummy_batch()

    @override(TorchPolicyV2)
    def loss(self, model: ModelV2, dist_class: Type[ActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        if False:
            while True:
                i = 10
        (model_out, _) = model(train_batch)
        action_dist = dist_class(model_out, model)
        if isinstance(self.action_space, gym.spaces.Discrete):
            is_multidiscrete = False
            output_hidden_shape = [self.action_space.n]
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            is_multidiscrete = True
            output_hidden_shape = self.action_space.nvec.astype(np.int32)
        else:
            is_multidiscrete = False
            output_hidden_shape = 1

        def _make_time_major(*args, **kw):
            if False:
                print('Hello World!')
            return make_time_major(self, train_batch.get(SampleBatch.SEQ_LENS), *args, **kw)
        actions = train_batch[SampleBatch.ACTIONS]
        dones = train_batch[SampleBatch.TERMINATEDS]
        rewards = train_batch[SampleBatch.REWARDS]
        behaviour_action_logp = train_batch[SampleBatch.ACTION_LOGP]
        behaviour_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS]
        if isinstance(output_hidden_shape, (list, tuple, np.ndarray)):
            unpacked_behaviour_logits = torch.split(behaviour_logits, list(output_hidden_shape), dim=1)
            unpacked_outputs = torch.split(model_out, list(output_hidden_shape), dim=1)
        else:
            unpacked_behaviour_logits = torch.chunk(behaviour_logits, output_hidden_shape, dim=1)
            unpacked_outputs = torch.chunk(model_out, output_hidden_shape, dim=1)
        values = model.value_function()
        values_time_major = _make_time_major(values)
        bootstrap_values_time_major = _make_time_major(train_batch[SampleBatch.VALUES_BOOTSTRAPPED])
        bootstrap_value = bootstrap_values_time_major[-1]
        if self.is_recurrent():
            max_seq_len = torch.max(train_batch[SampleBatch.SEQ_LENS])
            mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
            mask = torch.reshape(mask_orig, [-1])
        else:
            mask = torch.ones_like(rewards)
        loss_actions = actions if is_multidiscrete else torch.unsqueeze(actions, dim=1)
        loss = VTraceLoss(actions=_make_time_major(loss_actions), actions_logp=_make_time_major(action_dist.logp(actions)), actions_entropy=_make_time_major(action_dist.entropy()), dones=_make_time_major(dones), behaviour_action_logp=_make_time_major(behaviour_action_logp), behaviour_logits=_make_time_major(unpacked_behaviour_logits), target_logits=_make_time_major(unpacked_outputs), discount=self.config['gamma'], rewards=_make_time_major(rewards), values=values_time_major, bootstrap_value=bootstrap_value, dist_class=TorchCategorical if is_multidiscrete else dist_class, model=model, valid_mask=_make_time_major(mask), config=self.config, vf_loss_coeff=self.config['vf_loss_coeff'], entropy_coeff=self.entropy_coeff, clip_rho_threshold=self.config['vtrace_clip_rho_threshold'], clip_pg_rho_threshold=self.config['vtrace_clip_pg_rho_threshold'])
        model.tower_stats['pi_loss'] = loss.pi_loss
        model.tower_stats['vf_loss'] = loss.vf_loss
        model.tower_stats['entropy'] = loss.entropy
        model.tower_stats['mean_entropy'] = loss.mean_entropy
        model.tower_stats['total_loss'] = loss.total_loss
        values_batched = make_time_major(self, train_batch.get(SampleBatch.SEQ_LENS), values)
        model.tower_stats['vf_explained_var'] = explained_variance(torch.reshape(loss.value_targets, [-1]), torch.reshape(values_batched, [-1]))
        if self.config.get('_separate_vf_optimizer'):
            return (loss.loss_wo_vf, loss.vf_loss)
        else:
            return loss.total_loss

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        if False:
            while True:
                i = 10
        return convert_to_numpy({'cur_lr': self.cur_lr, 'total_loss': torch.mean(torch.stack(self.get_tower_stats('total_loss'))), 'policy_loss': torch.mean(torch.stack(self.get_tower_stats('pi_loss'))), 'entropy': torch.mean(torch.stack(self.get_tower_stats('mean_entropy'))), 'entropy_coeff': self.entropy_coeff, 'var_gnorm': global_norm(self.model.trainable_variables()), 'vf_loss': torch.mean(torch.stack(self.get_tower_stats('vf_loss'))), 'vf_explained_var': torch.mean(torch.stack(self.get_tower_stats('vf_explained_var')))})

    @override(TorchPolicyV2)
    def postprocess_trajectory(self, sample_batch: SampleBatch, other_agent_batches: Optional[SampleBatch]=None, episode: Optional['Episode']=None):
        if False:
            for i in range(10):
                print('nop')
        if self.config['vtrace']:
            sample_batch = compute_bootstrap_value(sample_batch, self)
        return sample_batch

    @override(TorchPolicyV2)
    def extra_grad_process(self, optimizer: 'torch.optim.Optimizer', loss: TensorType) -> Dict[str, TensorType]:
        if False:
            i = 10
            return i + 15
        return apply_grad_clipping(self, optimizer, loss)

    @override(TorchPolicyV2)
    def get_batch_divisibility_req(self) -> int:
        if False:
            print('Hello World!')
        return self.config['rollout_fragment_length']