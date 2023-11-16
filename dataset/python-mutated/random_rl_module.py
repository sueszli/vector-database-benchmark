from typing import Mapping, Any
from ray.rllib.core.rl_module import RLModule
from ray.rllib.policy.sample_batch import SampleBatch
import tree
import pathlib
import gymnasium as gym

class RandomRLModule(RLModule):

    def _random_forward(self, batch, **kwargs):
        if False:
            return 10
        obs_batch_size = len(tree.flatten(batch[SampleBatch.OBS])[0])
        actions = [self.config.action_space.sample() for _ in range(obs_batch_size)]
        return {SampleBatch.ACTIONS: actions}

    def _forward_inference(self, batch, **kwargs):
        if False:
            i = 10
            return i + 15
        return self._random_forward(batch, **kwargs)

    def _forward_exploration(self, batch, **kwargs):
        if False:
            return 10
        return self._random_forward(batch, **kwargs)

    def _forward_train(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('This RLM should only run in evaluation.')

    def output_specs_inference(self):
        if False:
            while True:
                i = 10
        return [SampleBatch.ACTIONS]

    def output_specs_exploration(self):
        if False:
            for i in range(10):
                print('nop')
        return [SampleBatch.ACTIONS]

    def get_state(self) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        return {}

    def set_state(self, state_dict: Mapping[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def _module_state_file_name(self) -> pathlib.Path:
        if False:
            print('Hello World!')
        return pathlib.Path('random_rl_module_dummy_state')

    def save_state(self, path) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def compile(self, *args, **kwargs):
        if False:
            return 10
        'Dummy method for compatibility with TorchRLModule.\n\n        This is hit when RolloutWorker tries to compile TorchRLModule.'
        pass

    @classmethod
    def from_model_config(cls, observation_space: gym.Space, action_space: gym.Space, *, model_config_dict: Mapping[str, Any]) -> 'RLModule':
        if False:
            return 10
        return cls(action_space)