from typing import Dict, List, Optional, Type, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_gae_for_sample_batch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import EntropyCoeffSchedule, LearningRateSchedule, ValueNetworkMixin
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import apply_grad_clipping, sequence_mask
from ray.rllib.utils.typing import AgentID, TensorType
(torch, nn) = try_import_torch()

class A3CTorchPolicy(ValueNetworkMixin, LearningRateSchedule, EntropyCoeffSchedule, TorchPolicyV2):
    """PyTorch Policy class used with A3C."""

    def __init__(self, observation_space, action_space, config):
        if False:
            while True:
                i = 10
        TorchPolicyV2.__init__(self, observation_space, action_space, config, max_seq_len=config['model']['max_seq_len'])
        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config['lr'], config['lr_schedule'])
        EntropyCoeffSchedule.__init__(self, config['entropy_coeff'], config['entropy_coeff_schedule'])
        self._initialize_loss_from_dummy_batch()

    @override(TorchPolicyV2)
    def loss(self, model: ModelV2, dist_class: Type[TorchDistributionWrapper], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        if False:
            while True:
                i = 10
        'Constructs the loss function.\n\n        Args:\n            model: The Model to calculate the loss for.\n            dist_class: The action distr. class.\n            train_batch: The training data.\n\n        Returns:\n            The A3C loss tensor given the input batch.\n        '
        (logits, _) = model(train_batch)
        values = model.value_function()
        if self.is_recurrent():
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
            valid_mask = torch.reshape(mask_orig, [-1])
        else:
            valid_mask = torch.ones_like(values, dtype=torch.bool)
        dist = dist_class(logits, model)
        log_probs = dist.logp(train_batch[SampleBatch.ACTIONS]).reshape(-1)
        pi_err = -torch.sum(torch.masked_select(log_probs * train_batch[Postprocessing.ADVANTAGES], valid_mask))
        if self.config['use_critic']:
            value_err = 0.5 * torch.sum(torch.pow(torch.masked_select(values.reshape(-1) - train_batch[Postprocessing.VALUE_TARGETS], valid_mask), 2.0))
        else:
            value_err = 0.0
        entropy = torch.sum(torch.masked_select(dist.entropy(), valid_mask))
        total_loss = pi_err + value_err * self.config['vf_loss_coeff'] - entropy * self.entropy_coeff
        model.tower_stats['entropy'] = entropy
        model.tower_stats['pi_err'] = pi_err
        model.tower_stats['value_err'] = value_err
        return total_loss

    @override(TorchPolicyV2)
    def optimizer(self) -> Union[List['torch.optim.Optimizer'], 'torch.optim.Optimizer']:
        if False:
            i = 10
            return i + 15
        'Returns a torch optimizer (Adam) for A3C.'
        return torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        if False:
            i = 10
            return i + 15
        return convert_to_numpy({'cur_lr': self.cur_lr, 'entropy_coeff': self.entropy_coeff, 'policy_entropy': torch.mean(torch.stack(self.get_tower_stats('entropy'))), 'policy_loss': torch.mean(torch.stack(self.get_tower_stats('pi_err'))), 'vf_loss': torch.mean(torch.stack(self.get_tower_stats('value_err')))})

    @override(TorchPolicyV2)
    def postprocess_trajectory(self, sample_batch: SampleBatch, other_agent_batches: Optional[Dict[AgentID, SampleBatch]]=None, episode: Optional[Episode]=None):
        if False:
            return 10
        sample_batch = super().postprocess_trajectory(sample_batch)
        return compute_gae_for_sample_batch(self, sample_batch, other_agent_batches, episode)

    @override(TorchPolicyV2)
    def extra_grad_process(self, optimizer: 'torch.optim.Optimizer', loss: TensorType) -> Dict[str, TensorType]:
        if False:
            while True:
                i = 10
        return apply_grad_clipping(self, optimizer, loss)