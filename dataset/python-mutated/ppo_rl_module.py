"""
This file holds framework-agnostic components for PPO's RLModules.
"""
import abc
from typing import Type
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.models.distributions import Distribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.annotations import override

@ExperimentalAPI
class PPORLModule(RLModule, abc.ABC):

    def setup(self):
        if False:
            i = 10
            return i + 15
        catalog = self.config.get_catalog()
        self.encoder = catalog.build_actor_critic_encoder(framework=self.framework)
        self.pi = catalog.build_pi_head(framework=self.framework)
        self.vf = catalog.build_vf_head(framework=self.framework)
        self.action_dist_cls = catalog.get_action_dist_cls(framework=self.framework)

    def get_train_action_dist_cls(self) -> Type[Distribution]:
        if False:
            while True:
                i = 10
        return self.action_dist_cls

    def get_exploration_action_dist_cls(self) -> Type[Distribution]:
        if False:
            while True:
                i = 10
        return self.action_dist_cls

    def get_inference_action_dist_cls(self) -> Type[Distribution]:
        if False:
            return 10
        return self.action_dist_cls

    @override(RLModule)
    def get_initial_state(self) -> dict:
        if False:
            return 10
        if hasattr(self.encoder, 'get_initial_state'):
            return self.encoder.get_initial_state()
        else:
            return {}

    @override(RLModule)
    def input_specs_inference(self) -> SpecDict:
        if False:
            while True:
                i = 10
        return self.input_specs_exploration()

    @override(RLModule)
    def output_specs_inference(self) -> SpecDict:
        if False:
            for i in range(10):
                print('nop')
        return [SampleBatch.ACTION_DIST_INPUTS]

    @override(RLModule)
    def input_specs_exploration(self):
        if False:
            while True:
                i = 10
        return [SampleBatch.OBS]

    @override(RLModule)
    def output_specs_exploration(self) -> SpecDict:
        if False:
            return 10
        return [SampleBatch.VF_PREDS, SampleBatch.ACTION_DIST_INPUTS]

    @override(RLModule)
    def input_specs_train(self) -> SpecDict:
        if False:
            return 10
        return self.input_specs_exploration()

    @override(RLModule)
    def output_specs_train(self) -> SpecDict:
        if False:
            return 10
        return [SampleBatch.VF_PREDS, SampleBatch.ACTION_DIST_INPUTS]