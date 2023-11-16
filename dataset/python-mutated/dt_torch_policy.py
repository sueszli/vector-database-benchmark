from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
import gymnasium as gym
import numpy as np
import tree
from gymnasium.spaces import Box, Discrete
from rllib_dt.dt.dt_torch_model import DTTorchModel
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.mingpt import configure_gpt_optimizer
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, TorchDeterministic, TorchDistributionWrapper
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import LearningRateSchedule
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI, override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.torch_utils import apply_grad_clipping
from ray.rllib.utils.typing import TensorShape, TensorStructType, TensorType, TrainerConfigDict
if TYPE_CHECKING:
    from ray.rllib.evaluation import Episode
(torch, nn) = try_import_torch()
F = nn.functional

class DTTorchPolicy(LearningRateSchedule, TorchPolicyV2):

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: TrainerConfigDict):
        if False:
            for i in range(10):
                print('nop')
        LearningRateSchedule.__init__(self, config['lr'], config['lr_schedule'])
        TorchPolicyV2.__init__(self, observation_space, action_space, config, max_seq_len=config['model']['max_seq_len'])

    @override(TorchPolicyV2)
    def make_model_and_action_dist(self) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
        if False:
            while True:
                i = 10
        model_config = self.config['model']
        model_config.update(embed_dim=self.config['embed_dim'], max_ep_len=self.config['horizon'], num_layers=self.config['num_layers'], num_heads=self.config['num_heads'], embed_pdrop=self.config['embed_pdrop'], resid_pdrop=self.config['resid_pdrop'], attn_pdrop=self.config['attn_pdrop'], use_obs_output=self.config.get('loss_coef_obs', 0) > 0, use_return_output=self.config.get('loss_coef_returns_to_go', 0) > 0)
        num_outputs = int(np.product(self.observation_space.shape))
        model = ModelCatalog.get_model_v2(obs_space=self.observation_space, action_space=self.action_space, num_outputs=num_outputs, model_config=model_config, framework=self.config['framework'], model_interface=None, default_model=DTTorchModel, name='model')
        if isinstance(self.action_space, Discrete):
            action_dist = TorchCategorical
        elif isinstance(self.action_space, Box):
            action_dist = TorchDeterministic
        else:
            raise NotImplementedError
        return (model, action_dist)

    @override(TorchPolicyV2)
    def optimizer(self) -> Union[List['torch.optim.Optimizer'], 'torch.optim.Optimizer']:
        if False:
            for i in range(10):
                print('nop')
        optimizer = configure_gpt_optimizer(model=self.model, learning_rate=self.config['lr'], weight_decay=self.config['optimizer']['weight_decay'], betas=self.config['optimizer']['betas'])
        return optimizer

    @override(TorchPolicyV2)
    def postprocess_trajectory(self, sample_batch: SampleBatch, other_agent_batches: Optional[Dict[Any, SampleBatch]]=None, episode: Optional['Episode']=None) -> SampleBatch:
        if False:
            for i in range(10):
                print('nop')
        'Called by offline data reader after loading in one episode.\n\n        Adds a `terminateds` flag at the end of trajectory so that SegmentationBuffer\n        can split using this flag to avoid duplicate trajectories.\n        '
        ep_len = sample_batch.env_steps()
        sample_batch[SampleBatch.TERMINATEDS] = np.array([False] * (ep_len - 1) + [True])
        return sample_batch

    @PublicAPI
    def get_initial_input_dict(self, observation: TensorStructType) -> SampleBatch:
        if False:
            while True:
                i = 10
        'Get the initial input_dict to be passed into compute_single_action.\n\n        Args:\n            observation: first (unbatched) observation from env.reset()\n\n        Returns:\n            The input_dict for inference: {\n                OBS: [max_seq_len, obs_dim] array,\n                ACTIONS: [max_seq_len - 1, act_dim] array,\n                RETURNS_TO_GO: [max_seq_len - 1] array,\n                REWARDS: scalar,\n                TIMESTEPS: [max_seq_len - 1] array,\n            }\n            Note the sequence lengths are different, and is specified as per\n            view_requirements. Explanations in action_distribution_fn method.\n        '
        observation = convert_to_numpy(observation)
        obs_shape = observation.shape
        obs_dtype = observation.dtype
        act_shape = self.action_space.shape
        act_dtype = self.action_space.dtype
        observations = np.concatenate([np.zeros((self.max_seq_len - 1, *obs_shape), dtype=obs_dtype), observation[None]], axis=0)
        actions = np.zeros((self.max_seq_len - 1, *act_shape), dtype=act_dtype)
        rtg = np.zeros(self.max_seq_len - 1, dtype=np.float32)
        rewards = np.zeros((), dtype=np.float32)
        timesteps = np.full(self.max_seq_len - 1, fill_value=-1, dtype=np.int32)
        input_dict = SampleBatch({SampleBatch.OBS: observations, SampleBatch.ACTIONS: actions, SampleBatch.RETURNS_TO_GO: rtg, SampleBatch.REWARDS: rewards, SampleBatch.T: timesteps})
        return input_dict

    @PublicAPI
    def get_next_input_dict(self, input_dict: SampleBatch, action: TensorStructType, reward: TensorStructType, next_obs: TensorStructType, extra: Dict[str, TensorType]) -> SampleBatch:
        if False:
            i = 10
            return i + 15
        'Returns a new input_dict after stepping through the environment once.\n\n        Args:\n            input_dict: the input dict passed into compute_single_action.\n            action: the (unbatched) action taken this step.\n            reward: the (unbatched) reward from env.step\n            next_obs: the (unbatached) next observation from env.step\n            extra: the extra action out from compute_single_action.\n                In this case contains current returns to go *before* the current\n                reward is subtracted from target_return.\n\n        Returns:\n            A new input_dict to be passed into compute_single_action.\n            The input_dict for inference: {\n                OBS: [max_seq_len, obs_dim] array,\n                ACTIONS: [max_seq_len - 1, act_dim] array,\n                RETURNS_TO_GO: [max_seq_len - 1] array,\n                REWARDS: scalar,\n                TIMESTEPS: [max_seq_len - 1] array,\n            }\n            Note the sequence lengths are different, and is specified as per\n            view_requirements. Explanations in action_distribution_fn method.\n        '
        input_dict = tree.map_structure(convert_to_numpy, input_dict)
        (action, reward, next_obs, extra) = convert_to_numpy((action, reward, next_obs, extra))
        assert input_dict[SampleBatch.OBS].shape == (self.max_seq_len, *self.observation_space.shape)
        assert input_dict[SampleBatch.ACTIONS].shape == (self.max_seq_len - 1, *self.action_space.shape)
        assert input_dict[SampleBatch.RETURNS_TO_GO].shape == (self.max_seq_len - 1,)
        assert input_dict[SampleBatch.T].shape == (self.max_seq_len - 1,)
        input_dict[SampleBatch.OBS] = np.concatenate([input_dict[SampleBatch.OBS][1:], next_obs[None]], axis=0)
        input_dict[SampleBatch.ACTIONS] = np.concatenate([input_dict[SampleBatch.ACTIONS][1:], action[None]], axis=0)
        input_dict[SampleBatch.REWARDS] = np.asarray(reward)
        input_dict[SampleBatch.RETURNS_TO_GO] = np.concatenate([input_dict[SampleBatch.RETURNS_TO_GO][1:], np.asarray(extra[SampleBatch.RETURNS_TO_GO])[None]], axis=0)
        input_dict[SampleBatch.T] = np.concatenate([input_dict[SampleBatch.T][1:], input_dict[SampleBatch.T][-1:] + 1], axis=0)
        return input_dict

    @DeveloperAPI
    def get_initial_rtg_tensor(self, shape: TensorShape, dtype: Optional[Type]=torch.float32, device: Optional['torch.device']=None):
        if False:
            print('Hello World!')
        'Returns a initial/target returns-to-go tensor of the given shape.\n\n        Args:\n            shape: Shape of the rtg tensor.\n            dtype: Type of the data in the tensor. Defaults to torch.float32.\n            device: The device this tensor should be on. Defaults to self.device.\n        '
        if device is None:
            device = self.device
        if dtype is None:
            device = torch.float32
        assert self.config['target_return'] is not None, 'Must specify target_return.'
        initial_rtg = torch.full(shape, fill_value=self.config['target_return'], dtype=dtype, device=device)
        return initial_rtg

    @override(TorchPolicyV2)
    @DeveloperAPI
    def compute_actions(self, *args, **kwargs) -> Tuple[TensorStructType, List[TensorType], Dict[str, TensorType]]:
        if False:
            return 10
        raise ValueError('Please use compute_actions_from_input_dict instead.')

    @override(TorchPolicyV2)
    def compute_actions_from_input_dict(self, input_dict: Union[SampleBatch, Dict[str, TensorStructType]], explore: bool=None, timestep: Optional[int]=None, **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            input_dict: input_dict (that contains a batch dimension for each value).\n                Keys and shapes: {\n                    OBS: [batch_size, max_seq_len, obs_dim],\n                    ACTIONS: [batch_size, max_seq_len - 1, act_dim],\n                    RETURNS_TO_GO: [batch_size, max_seq_len - 1],\n                    REWARDS: [batch_size],\n                    TIMESTEPS: [batch_size, max_seq_len - 1],\n                }\n            explore: unused.\n            timestep: unused.\n        Returns:\n            A tuple consisting of a) actions, b) state_out, c) extra_fetches.\n        '
        with torch.no_grad():
            input_dict = input_dict.copy()
            input_dict = self._lazy_tensor_dict(input_dict)
            input_dict.set_training(True)
            (actions, state_out, extra_fetches) = self._compute_action_helper(input_dict)
            return (actions, state_out, extra_fetches)

    @with_lock
    @override(TorchPolicyV2)
    def _compute_action_helper(self, input_dict):
        if False:
            print('Hello World!')
        self.model.eval()
        batch_size = input_dict[SampleBatch.OBS].shape[0]
        timesteps = input_dict[SampleBatch.T]
        new_timestep = timesteps[:, -1:] + 1
        input_dict[SampleBatch.T] = torch.cat([timesteps, new_timestep], dim=1)
        input_dict[SampleBatch.ATTENTION_MASKS] = torch.where(input_dict[SampleBatch.T] >= 0, 1.0, 0.0)
        uncliped_timesteps = input_dict[SampleBatch.T]
        input_dict[SampleBatch.T] = torch.where(uncliped_timesteps < 0, torch.zeros_like(uncliped_timesteps), uncliped_timesteps)
        rtg = input_dict[SampleBatch.RETURNS_TO_GO]
        last_rtg = rtg[:, -1]
        last_reward = input_dict[SampleBatch.REWARDS]
        updated_rtg = last_rtg - last_reward
        initial_rtg = self.get_initial_rtg_tensor((batch_size, 1), dtype=rtg.dtype, device=rtg.device)
        new_rtg = torch.where(new_timestep == 0, initial_rtg, updated_rtg[:, None])
        input_dict[SampleBatch.RETURNS_TO_GO] = torch.cat([rtg, new_rtg], dim=1)[..., None]
        past_actions = input_dict[SampleBatch.ACTIONS]
        action_pad = torch.zeros((batch_size, 1, *past_actions.shape[2:]), dtype=past_actions.dtype, device=past_actions.device)
        input_dict[SampleBatch.ACTIONS] = torch.cat([past_actions, action_pad], dim=1)
        (model_out, _) = self.model(input_dict)
        preds = self.model.get_prediction(model_out, input_dict)
        dist_inputs = preds[SampleBatch.ACTIONS][:, -1]
        action_dist = self.dist_class(dist_inputs, self.model)
        actions = action_dist.deterministic_sample()
        extra_fetches = {SampleBatch.RETURNS_TO_GO: new_rtg.squeeze(-1), SampleBatch.ACTION_DIST_INPUTS: dist_inputs}
        self.global_timestep += len(input_dict[SampleBatch.CUR_OBS])
        return convert_to_numpy((actions, [], extra_fetches))

    @override(TorchPolicyV2)
    def loss(self, model: ModelV2, dist_class: Type[TorchDistributionWrapper], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        if False:
            return 10
        'Loss function.\n\n        Args:\n            model: The ModelV2 to run foward pass on.\n            dist_class: The distribution of this policy.\n            train_batch: Training SampleBatch.\n                Keys and shapes: {\n                    OBS: [batch_size, max_seq_len, obs_dim],\n                    ACTIONS: [batch_size, max_seq_len, act_dim],\n                    RETURNS_TO_GO: [batch_size, max_seq_len + 1, 1],\n                    TIMESTEPS: [batch_size, max_seq_len],\n                    ATTENTION_MASKS: [batch_size, max_seq_len],\n                }\n        Returns:\n            Loss scalar tensor.\n        '
        train_batch = self._lazy_tensor_dict(train_batch)
        (model_out, _) = self.model(train_batch)
        preds = self.model.get_prediction(model_out, train_batch)
        targets = self.model.get_targets(model_out, train_batch)
        masks = train_batch[SampleBatch.ATTENTION_MASKS]
        loss = self._masked_loss(preds, targets, masks)
        self.log('cur_lr', torch.tensor(self.cur_lr))
        return loss

    def _masked_loss(self, preds, targets, masks):
        if False:
            i = 10
            return i + 15
        losses = []
        for key in targets:
            assert key in preds, 'for target {key} there is no prediction from the output of the model'
            loss_coef = self.config.get(f'loss_coef_{key}', 1.0)
            if self._is_discrete(key):
                loss = loss_coef * self._masked_cross_entropy_loss(preds[key], targets[key], masks)
            else:
                loss = loss_coef * self._masked_mse_loss(preds[key], targets[key], masks)
            losses.append(loss)
            self.log(f'{key}_loss', loss)
        return sum(losses)

    def _is_discrete(self, key):
        if False:
            i = 10
            return i + 15
        return key == SampleBatch.ACTIONS and isinstance(self.action_space, Discrete)

    def _masked_cross_entropy_loss(self, preds: TensorType, targets: TensorType, masks: TensorType) -> TensorType:
        if False:
            print('Hello World!')
        "Computes cross-entropy loss between preds and targets, subject to a mask.\n\n        Args:\n            preds: logits of shape [B1, ..., Bn, M]\n            targets: index targets for preds of shape [B1, ..., Bn]\n            masks: 0 means don't compute loss, 1 means compute loss\n                shape [B1, ..., Bn]\n\n        Returns:\n            Scalar cross entropy loss.\n        "
        losses = F.cross_entropy(preds.reshape(-1, preds.shape[-1]), targets.reshape(-1).long(), reduction='none')
        losses = losses * masks.reshape(-1)
        return losses.mean()

    def _masked_mse_loss(self, preds: TensorType, targets: TensorType, masks: TensorType) -> TensorType:
        if False:
            i = 10
            return i + 15
        "Computes MSE loss between preds and targets, subject to a mask.\n\n        Args:\n            preds: logits of shape [B1, ..., Bn, M]\n            targets: index targets for preds of shape [B1, ..., Bn]\n            masks: 0 means don't compute loss, 1 means compute loss\n                shape [B1, ..., Bn]\n\n        Returns:\n            Scalar cross entropy loss.\n        "
        losses = F.mse_loss(preds, targets, reduction='none')
        losses = losses * masks.reshape(*masks.shape, *[1] * (len(preds.shape) - len(masks.shape)))
        return losses.mean()

    @override(TorchPolicyV2)
    def extra_grad_process(self, local_optimizer, loss):
        if False:
            print('Hello World!')
        return apply_grad_clipping(self, local_optimizer, loss)

    def log(self, key, value):
        if False:
            return 10
        self.model.tower_stats[key] = value

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        if False:
            return 10
        stats_dict = {k: torch.stack(self.get_tower_stats(k)).mean().item() for k in self.model.tower_stats}
        return stats_dict