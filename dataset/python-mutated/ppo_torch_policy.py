import logging
from typing import Dict, List, Type, Union
import ray
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_gae_for_sample_batch
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import EntropyCoeffSchedule, KLCoeffMixin, LearningRateSchedule, ValueNetworkMixin
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import apply_grad_clipping, explained_variance, sequence_mask, warn_if_infinite_kl_divergence
from ray.rllib.utils.typing import TensorType
(torch, nn) = try_import_torch()
logger = logging.getLogger(__name__)

class PPOTorchPolicy(ValueNetworkMixin, LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, TorchPolicyV2):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):
        if False:
            i = 10
            return i + 15
        config = dict(ray.rllib.algorithms.ppo.ppo.PPOConfig().to_dict(), **config)
        validate_config(config)
        TorchPolicyV2.__init__(self, observation_space, action_space, config, max_seq_len=config['model']['max_seq_len'])
        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config['lr'], config['lr_schedule'])
        EntropyCoeffSchedule.__init__(self, config['entropy_coeff'], config['entropy_coeff_schedule'])
        KLCoeffMixin.__init__(self, config)
        self._initialize_loss_from_dummy_batch()

    @override(TorchPolicyV2)
    def loss(self, model: ModelV2, dist_class: Type[ActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        if False:
            i = 10
            return i + 15
        'Compute loss for Proximal Policy Objective.\n\n        Args:\n            model: The Model to calculate the loss for.\n            dist_class: The action distr. class.\n            train_batch: The training data.\n\n        Returns:\n            The PPO loss tensor given the input batch.\n        '
        (logits, state) = model(train_batch)
        curr_action_dist = dist_class(logits, model)
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len, time_major=model.is_time_major())
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                if False:
                    return 10
                return torch.sum(t[mask]) / num_valid
        else:
            mask = None
            reduce_mean_valid = torch.mean
        prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
        logp_ratio = torch.exp(curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) - train_batch[SampleBatch.ACTION_LOGP])
        if self.config['kl_coeff'] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)
        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)
        surrogate_loss = torch.min(train_batch[Postprocessing.ADVANTAGES] * logp_ratio, train_batch[Postprocessing.ADVANTAGES] * torch.clamp(logp_ratio, 1 - self.config['clip_param'], 1 + self.config['clip_param']))
        if self.config['use_critic']:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config['vf_clip_param'])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)
        total_loss = reduce_mean_valid(-surrogate_loss + self.config['vf_loss_coeff'] * vf_loss_clipped - self.entropy_coeff * curr_entropy)
        if self.config['kl_coeff'] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss
        model.tower_stats['total_loss'] = total_loss
        model.tower_stats['mean_policy_loss'] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats['mean_vf_loss'] = mean_vf_loss
        model.tower_stats['vf_explained_var'] = explained_variance(train_batch[Postprocessing.VALUE_TARGETS], value_fn_out)
        model.tower_stats['mean_entropy'] = mean_entropy
        model.tower_stats['mean_kl_loss'] = mean_kl_loss
        return total_loss

    @override(TorchPolicyV2)
    def extra_grad_process(self, local_optimizer, loss):
        if False:
            return 10
        return apply_grad_clipping(self, local_optimizer, loss)

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        if False:
            return 10
        return convert_to_numpy({'cur_kl_coeff': self.kl_coeff, 'cur_lr': self.cur_lr, 'total_loss': torch.mean(torch.stack(self.get_tower_stats('total_loss'))), 'policy_loss': torch.mean(torch.stack(self.get_tower_stats('mean_policy_loss'))), 'vf_loss': torch.mean(torch.stack(self.get_tower_stats('mean_vf_loss'))), 'vf_explained_var': torch.mean(torch.stack(self.get_tower_stats('vf_explained_var'))), 'kl': torch.mean(torch.stack(self.get_tower_stats('mean_kl_loss'))), 'entropy': torch.mean(torch.stack(self.get_tower_stats('mean_entropy'))), 'entropy_coeff': self.entropy_coeff})

    @override(TorchPolicyV2)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        if False:
            i = 10
            return i + 15
        with torch.no_grad():
            return compute_gae_for_sample_batch(self, sample_batch, other_agent_batches, episode)