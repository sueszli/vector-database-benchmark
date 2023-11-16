import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import _check_if_diag_gaussian
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import FreeLogStdMLPHeadConfig, MLPHeadConfig
from ray.rllib.core.models.base import Model
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic

class BCCatalog(Catalog):
    """The Catalog class used to build models for BC.

    BCCatalog provides the following models:
        - Encoder: The encoder used to encode the observations.
        - Pi Head: The head used for the policy logits.

    The default encoder is chosen by RLlib dependent on the observation space.
    See `ray.rllib.core.models.encoders::Encoder` for details. To define the
    network architecture use the `model_config_dict[fcnet_hiddens]` and
    `model_config_dict[fcnet_activation]`.

    To implement custom logic, override `BCCatalog.build_encoder()` or modify the
    `EncoderConfig` at `BCCatalog.encoder_config`.

    Any custom head can be built by overriding the `build_pi_head()` method.
    Alternatively, the `PiHeadConfig` can be overridden to build a custom
    policy head during runtime. To change solely the network architecture,
    `model_config_dict["post_fcnet_hiddens"]` and
    `model_config_dict["post_fcnet_activation"]` can be used.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, model_config_dict: dict):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the BCCatalog.\n\n        Args:\n            observation_space: The observation space if the Encoder.\n            action_space: The action space for the Pi Head.\n            model_cnfig_dict: The model config to use..\n        '
        super().__init__(observation_space=observation_space, action_space=action_space, model_config_dict=model_config_dict)
        self.pi_head_hiddens = self._model_config_dict['post_fcnet_hiddens']
        self.pi_head_activation = self._model_config_dict['post_fcnet_activation']
        self.pi_head_config = None

    @OverrideToImplementCustomLogic
    def build_pi_head(self, framework: str) -> Model:
        if False:
            for i in range(10):
                print('nop')
        'Builds the policy head.\n\n        The default behavior is to build the head from the pi_head_config.\n        This can be overridden to build a custom policy head as a means of configuring\n        the behavior of a BCRLModule implementation.\n\n        Args:\n            framework: The framework to use. Either "torch" or "tf2".\n\n        Returns:\n            The policy head.\n        '
        action_distribution_cls = self.get_action_dist_cls(framework=framework)
        if self._model_config_dict['free_log_std']:
            _check_if_diag_gaussian(action_distribution_cls=action_distribution_cls, framework=framework)
        required_output_dim = action_distribution_cls.required_input_dim(space=self.action_space, model_config=self._model_config_dict)
        pi_head_config_cls = FreeLogStdMLPHeadConfig if self._model_config_dict['free_log_std'] else MLPHeadConfig
        self.pi_head_config = pi_head_config_cls(input_dims=self._latent_dims, hidden_layer_dims=self.pi_head_hiddens, hidden_layer_activation=self.pi_head_activation, output_layer_dim=required_output_dim, output_layer_activation='linear')
        return self.pi_head_config.build(framework=framework)