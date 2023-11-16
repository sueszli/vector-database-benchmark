"""PyTorch policy class used for Simple Q-Learning"""
import logging
from typing import Any, Dict, List, Tuple, Type, Union
from ray.rllib.algorithms.simple_q.utils import make_q_models
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, TorchDistributionWrapper
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import TargetNetworkMixin, LearningRateSchedule
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import concat_multi_gpu_td_errors, huber_loss
from ray.rllib.utils.typing import TensorStructType, TensorType
(torch, nn) = try_import_torch()
F = None
if nn:
    F = nn.functional
logger = logging.getLogger(__name__)

class SimpleQTorchPolicy(LearningRateSchedule, TargetNetworkMixin, TorchPolicyV2):
    """PyTorch policy class used with SimpleQ."""

    def __init__(self, observation_space, action_space, config):
        if False:
            i = 10
            return i + 15
        TorchPolicyV2.__init__(self, observation_space, action_space, config, max_seq_len=config['model']['max_seq_len'])
        LearningRateSchedule.__init__(self, config['lr'], config['lr_schedule'])
        self._initialize_loss_from_dummy_batch()
        TargetNetworkMixin.__init__(self)

    @override(TorchPolicyV2)
    def make_model(self) -> ModelV2:
        if False:
            for i in range(10):
                print('nop')
        'Builds q_model and target_model for Simple Q learning.'
        (model, self.target_model) = make_q_models(self)
        return model

    @override(TorchPolicyV2)
    def compute_actions(self, *, input_dict, explore=True, timestep=None, episodes=None, is_training=False, **kwargs) -> Tuple[TensorStructType, List[TensorType], Dict[str, TensorStructType]]:
        if False:
            return 10
        if timestep is None:
            timestep = self.global_timestep
        q_vals = self._compute_q_values(self.model, input_dict[SampleBatch.OBS], is_training=is_training)
        distribution = TorchCategorical(q_vals, self.model)
        (actions, logp) = self.exploration.get_exploration_action(action_distribution=distribution, timestep=timestep, explore=explore)
        return (actions, [], {'q_values': q_vals, SampleBatch.ACTION_LOGP: logp, SampleBatch.ACTION_PROB: torch.exp(logp), SampleBatch.ACTION_DIST_INPUTS: q_vals})

    @override(TorchPolicyV2)
    def loss(self, model: ModelV2, dist_class: Type[TorchDistributionWrapper], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        if False:
            return 10
        'Compute loss for SimpleQ.\n\n        Args:\n            model: The Model to calculate the loss for.\n            dist_class: The action distr. class.\n            train_batch: The training data.\n\n        Returns:\n            The SimpleQ loss tensor given the input batch.\n        '
        target_model = self.target_models[model]
        q_t = self._compute_q_values(model, train_batch[SampleBatch.CUR_OBS], is_training=True)
        q_tp1 = self._compute_q_values(target_model, train_batch[SampleBatch.NEXT_OBS], is_training=True)
        one_hot_selection = F.one_hot(train_batch[SampleBatch.ACTIONS].long(), self.action_space.n)
        q_t_selected = torch.sum(q_t * one_hot_selection, 1)
        dones = train_batch[SampleBatch.TERMINATEDS].float()
        q_tp1_best_one_hot_selection = F.one_hot(torch.argmax(q_tp1, 1), self.action_space.n)
        q_tp1_best = torch.sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
        q_tp1_best_masked = (1.0 - dones) * q_tp1_best
        q_t_selected_target = train_batch[SampleBatch.REWARDS] + self.config['gamma'] * q_tp1_best_masked
        td_error = q_t_selected - q_t_selected_target.detach()
        loss = torch.mean(huber_loss(td_error))
        model.tower_stats['loss'] = loss
        model.tower_stats['td_error'] = td_error
        return loss

    @override(TorchPolicyV2)
    def extra_compute_grad_fetches(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        fetches = convert_to_numpy(concat_multi_gpu_td_errors(self))
        return dict({LEARNER_STATS_KEY: {}}, **fetches)

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        if False:
            print('Hello World!')
        return convert_to_numpy({'loss': torch.mean(torch.stack(self.get_tower_stats('loss'))), 'cur_lr': self.cur_lr})

    def _compute_q_values(self, model: ModelV2, obs_batch: TensorType, is_training=None) -> TensorType:
        if False:
            while True:
                i = 10
        _is_training = is_training if is_training is not None else False
        input_dict = SampleBatch(obs=obs_batch, _is_training=_is_training)
        (model_out, _) = model(input_dict, [], None)
        return model_out