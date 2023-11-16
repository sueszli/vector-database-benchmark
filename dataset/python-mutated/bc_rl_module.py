import abc
from typing import Any, List, Mapping, Type, Union
from ray.rllib.core.models.base import ENCODER_OUT, STATE_OUT
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.models.distributions import Distribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import ExperimentalAPI, override
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.typing import TensorType

@ExperimentalAPI
class BCRLModule(RLModule, abc.ABC):

    @override(RLModule)
    def setup(self):
        if False:
            while True:
                i = 10
        catalog = self.config.get_catalog()
        self.encoder = catalog.build_encoder(framework=self.framework)
        self.pi = catalog.build_pi_head(framework=self.framework)
        self.action_dist_cls = catalog.get_action_dist_cls(framework=self.framework)

    @override(RLModule)
    def get_train_action_dist_cls(self) -> Type[Distribution]:
        if False:
            print('Hello World!')
        return self.action_dist_cls

    @override(RLModule)
    def get_exploration_action_dist_cls(self) -> Type[Distribution]:
        if False:
            for i in range(10):
                print('nop')
        return self.action_dist_cls

    @override(RLModule)
    def get_inference_action_dist_cls(self) -> Type[Distribution]:
        if False:
            while True:
                i = 10
        return self.action_dist_cls

    @override(RLModule)
    def get_initial_state(self) -> Union[dict, List[TensorType]]:
        if False:
            print('Hello World!')
        if hasattr(self.encoder, 'get_initial_state'):
            return self.encoder.get_initial_state()
        else:
            return {}

    @override(RLModule)
    def output_specs_inference(self) -> SpecType:
        if False:
            while True:
                i = 10
        return self.output_specs_exploration()

    @override(RLModule)
    def output_specs_exploration(self) -> SpecType:
        if False:
            while True:
                i = 10
        return [SampleBatch.ACTION_DIST_INPUTS]

    @override(RLModule)
    def output_specs_train(self) -> SpecType:
        if False:
            i = 10
            return i + 15
        return self.output_specs_exploration()

    @override(RLModule)
    def _forward_inference(self, batch: NestedDict, **kwargs) -> Mapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'BC forward pass during inference.\n\n        See the `BCTorchRLModule._forward_exploration` method for\n        implementation details.\n        '
        return self._forward_exploration(batch)

    @override(RLModule)
    def _forward_exploration(self, batch: NestedDict, **kwargs) -> Mapping[str, Any]:
        if False:
            print('Hello World!')
        'BC forward pass during exploration.\n\n        Besides the action distribution this method also returns a possible\n        state in case a stateful encoder is used.\n\n        Note that for BC `_forward_train`, `_forward_exploration`, and\n        `_forward_inference` return the same items and therefore only\n        `_forward_exploration` is implemented and is used by the two other\n        forward methods.\n        '
        output = {}
        encoder_outs = self.encoder(batch)
        if STATE_OUT in encoder_outs:
            output[STATE_OUT] = encoder_outs[STATE_OUT]
        action_logits = self.pi(encoder_outs[ENCODER_OUT])
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits
        return output

    @override(RLModule)
    def _forward_train(self, batch: NestedDict, **kwargs) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        'BC forward pass during training.\n\n        See the `BCTorchRLModule._forward_exploration` method for\n        implementation details.\n        '
        return self._forward_exploration(batch)