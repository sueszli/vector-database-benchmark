import tensorflow as tf
from typing import Any, Mapping
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.models.tf.tf_distributions import TfCategorical
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModule, MultiAgentRLModuleConfig
from ray.rllib.core.rl_module.tf.tf_rl_module import TfRLModule
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.nested_dict import NestedDict

class DiscreteBCTFModule(TfRLModule):

    def __init__(self, config: RLModuleConfig) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)

    def setup(self):
        if False:
            return 10
        input_dim = self.config.observation_space.shape[0]
        hidden_dim = self.config.model_config_dict['fcnet_hiddens'][0]
        output_dim = self.config.action_space.n
        layers = []
        layers.append(tf.keras.Input(shape=(input_dim,)))
        layers.append(tf.keras.layers.ReLU())
        layers.append(tf.keras.layers.Dense(hidden_dim))
        layers.append(tf.keras.layers.ReLU())
        layers.append(tf.keras.layers.Dense(output_dim))
        self.policy = tf.keras.Sequential(layers)
        self._input_dim = input_dim

    def get_train_action_dist_cls(self):
        if False:
            i = 10
            return i + 15
        return TfCategorical

    def get_exploration_action_dist_cls(self):
        if False:
            while True:
                i = 10
        return TfCategorical

    def get_inference_action_dist_cls(self):
        if False:
            while True:
                i = 10
        return TfCategorical

    @override(RLModule)
    def output_specs_exploration(self) -> SpecType:
        if False:
            for i in range(10):
                print('nop')
        return [SampleBatch.ACTION_DIST_INPUTS]

    @override(RLModule)
    def output_specs_inference(self) -> SpecType:
        if False:
            for i in range(10):
                print('nop')
        return [SampleBatch.ACTION_DIST_INPUTS]

    @override(RLModule)
    def output_specs_train(self) -> SpecType:
        if False:
            return 10
        return [SampleBatch.ACTION_DIST_INPUTS]

    def _forward_shared(self, batch: NestedDict) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        action_logits = self.policy(batch['obs'])
        return {SampleBatch.ACTION_DIST_INPUTS: action_logits}

    @override(RLModule)
    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        if False:
            return 10
        return self._forward_shared(batch)

    @override(RLModule)
    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        if False:
            return 10
        return self._forward_shared(batch)

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        return self._forward_shared(batch)

    @override(RLModule)
    def get_state(self) -> Mapping[str, Any]:
        if False:
            return 10
        return {'policy': self.policy.get_weights()}

    @override(RLModule)
    def set_state(self, state: Mapping[str, Any]) -> None:
        if False:
            while True:
                i = 10
        self.policy.set_weights(state['policy'])

class BCTfRLModuleWithSharedGlobalEncoder(TfRLModule):

    def __init__(self, encoder, local_dim, hidden_dim, action_dim):
        if False:
            return 10
        super().__init__()
        self.encoder = encoder
        self.policy_head = tf.keras.Sequential([tf.keras.layers.Dense(hidden_dim + local_dim, input_shape=(hidden_dim + local_dim,), activation='relu'), tf.keras.layers.Dense(hidden_dim, activation='relu'), tf.keras.layers.Dense(action_dim)])

    @override(RLModule)
    def _default_input_specs(self):
        if False:
            print('Hello World!')
        return [('obs', 'global'), ('obs', 'local')]

    @override(RLModule)
    def _forward_inference(self, batch):
        if False:
            i = 10
            return i + 15
        return self._common_forward(batch)

    @override(RLModule)
    def _forward_exploration(self, batch):
        if False:
            i = 10
            return i + 15
        return self._common_forward(batch)

    @override(RLModule)
    def _forward_train(self, batch):
        if False:
            for i in range(10):
                print('nop')
        return self._common_forward(batch)

    def _common_forward(self, batch):
        if False:
            while True:
                i = 10
        obs = batch['obs']
        global_enc = self.encoder(obs['global'])
        policy_in = tf.concat([global_enc, obs['local']], axis=-1)
        action_logits = self.policy_head(policy_in)
        return {SampleBatch.ACTION_DIST_INPUTS: action_logits}

class BCTfMultiAgentModuleWithSharedEncoder(MultiAgentRLModule):

    def __init__(self, config: MultiAgentRLModuleConfig) -> None:
        if False:
            print('Hello World!')
        super().__init__(config)

    def setup(self):
        if False:
            return 10
        module_specs = self.config.modules
        module_spec = next(iter(module_specs.values()))
        global_dim = module_spec.observation_space['global'].shape[0]
        hidden_dim = module_spec.model_config_dict['fcnet_hiddens'][0]
        shared_encoder = tf.keras.Sequential([tf.keras.Input(shape=(global_dim,)), tf.keras.layers.ReLU(), tf.keras.layers.Dense(hidden_dim)])
        for (module_id, module_spec) in module_specs.items():
            self._rl_modules[module_id] = module_spec.module_class(encoder=shared_encoder, local_dim=module_spec.observation_space['local'].shape[0], hidden_dim=hidden_dim, action_dim=module_spec.action_space.n)

    def serialize(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def deserialize(self, data):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError