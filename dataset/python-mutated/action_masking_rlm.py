import gymnasium as gym
from ray.rllib.algorithms.ppo.tf.ppo_tf_rl_module import PPOTfRLModule
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch, try_import_tf
from ray.rllib.utils.torch_utils import FLOAT_MIN
(torch, nn) = try_import_torch()
(_, tf, _) = try_import_tf()

class ActionMaskRLMBase(RLModule):

    def __init__(self, config: RLModuleConfig):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(config.observation_space, gym.spaces.Dict):
            raise ValueError('This model requires the environment to provide a gym.spaces.Dict observation space.')
        config.observation_space = config.observation_space['observations']
        super().__init__(config)

class TorchActionMaskRLM(ActionMaskRLMBase, PPOTorchRLModule):

    def _forward_inference(self, batch, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return mask_forward_fn_torch(super()._forward_inference, batch, **kwargs)

    def _forward_train(self, batch, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return mask_forward_fn_torch(super()._forward_train, batch, **kwargs)

    def _forward_exploration(self, batch, *args, **kwargs):
        if False:
            return 10
        return mask_forward_fn_torch(super()._forward_exploration, batch, **kwargs)

class TFActionMaskRLM(ActionMaskRLMBase, PPOTfRLModule):

    def _forward_inference(self, batch, **kwargs):
        if False:
            i = 10
            return i + 15
        return mask_forward_fn_tf(super()._forward_inference, batch, **kwargs)

    def _forward_train(self, batch, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return mask_forward_fn_tf(super()._forward_train, batch, **kwargs)

    def _forward_exploration(self, batch, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return mask_forward_fn_tf(super()._forward_exploration, batch, **kwargs)

def mask_forward_fn_torch(forward_fn, batch, **kwargs):
    if False:
        i = 10
        return i + 15
    _check_batch(batch)
    action_mask = batch[SampleBatch.OBS]['action_mask']
    batch[SampleBatch.OBS] = batch[SampleBatch.OBS]['observations']
    outputs = forward_fn(batch, **kwargs)
    logits = outputs[SampleBatch.ACTION_DIST_INPUTS]
    inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
    masked_logits = logits + inf_mask
    outputs[SampleBatch.ACTION_DIST_INPUTS] = masked_logits
    return outputs

def mask_forward_fn_tf(forward_fn, batch, **kwargs):
    if False:
        while True:
            i = 10
    _check_batch(batch)
    action_mask = batch[SampleBatch.OBS]['action_mask']
    batch[SampleBatch.OBS] = batch[SampleBatch.OBS]['observations']
    outputs = forward_fn(batch, **kwargs)
    logits = outputs[SampleBatch.ACTION_DIST_INPUTS]
    inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
    masked_logits = logits + inf_mask
    outputs[SampleBatch.ACTION_DIST_INPUTS] = masked_logits
    return outputs

def _check_batch(batch):
    if False:
        i = 10
        return i + 15
    'Check whether the batch contains the required keys.'
    if 'action_mask' not in batch[SampleBatch.OBS]:
        raise ValueError("Action mask not found in observation. This model requires the environment to provide observations that include an action mask (i.e. an observation space of the Dict space type that looks as follows: \n{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),'observations': <observation_space>}")
    if 'observations' not in batch[SampleBatch.OBS]:
        raise ValueError("Observations not found in observation.This model requires the environment to provide observations that include a  (i.e. an observation space of the Dict space type that looks as follows: \n{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),'observations': <observation_space>}")