from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.algorithms.ppo.tf.ppo_tf_rl_module import PPOTfRLModule
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_tf, try_import_torch
import gymnasium as gym
(tf1, tf, tfv) = try_import_tf()
(torch, nn) = try_import_torch()

class FrameStackingCartPoleRLMBase(PPORLModule):
    """An RLModules that takes the last n observations as input.

    The idea behind this model is to demonstrate how we can modify an existing RLModule
    with a custom view requirement. In this case, we hack a PPORModule so that it
    constructs its models for an observation space that is num_frames times larger than
    the original observation space. We then stack the last num_frames observations on
    top of each other and feed them into the encoder. This allows us to train a model
    that can make use of the temporal information in the observations.
    """
    num_frames = 16

    def __init__(self, config: RLModuleConfig):
        if False:
            i = 10
            return i + 15
        original_obs_space = config.observation_space
        stacked_obs_space_size = sum(config.observation_space.shape * self.num_frames)
        stacked_obs_space = gym.spaces.Box(low=config.observation_space.low[0], high=config.observation_space.high[0], shape=(stacked_obs_space_size,), dtype=config.observation_space.dtype)
        config.observation_space = stacked_obs_space
        super().__init__(config)
        self.config.observation_space = original_obs_space

    def update_default_view_requirements(self, defaults):
        if False:
            print('Hello World!')
        defaults['prev_n_obs'] = ViewRequirement(data_col='obs', shift='-{}:0'.format(self.num_frames - 1), space=self.config.observation_space)
        return defaults

    def _forward_inference(self, batch, *args, **kwargs):
        if False:
            return 10
        batch = self._preprocess_batch(batch)
        return super()._forward_inference(batch, *args, **kwargs)

    def _forward_train(self, batch, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        batch = self._preprocess_batch(batch)
        return super()._forward_train(batch, *args, **kwargs)

    def _forward_exploration(self, batch, *args, **kwargs):
        if False:
            while True:
                i = 10
        batch = self._preprocess_batch(batch)
        return super()._forward_exploration(batch, *args, **kwargs)

    def _preprocess_batch(self, batch):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('You can not use the base class directly, but a framework-specific subclass.')

class TorchFrameStackingCartPoleRLM(FrameStackingCartPoleRLMBase, PPOTorchRLModule):

    @staticmethod
    def _preprocess_batch(batch):
        if False:
            for i in range(10):
                print('nop')
        shape = batch['prev_n_obs'].shape
        obs = batch['prev_n_obs'].reshape((shape[0], shape[1] * shape[2]))
        batch[SampleBatch.OBS] = obs
        return batch

class TfFrameStackingCartPoleRLM(FrameStackingCartPoleRLMBase, PPOTfRLModule):

    @staticmethod
    def _preprocess_batch(batch):
        if False:
            return 10
        shape = batch['prev_n_obs'].shape
        obs = tf.reshape(batch['prev_n_obs'], (shape[0], shape[1] * shape[2]))
        batch[SampleBatch.OBS] = obs
        return batch