from gymnasium.spaces import Box, Discrete, Space
import numpy as np
from typing import List, Optional, Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.tf_utils import get_placeholder
from ray.rllib.utils.typing import FromConfigSpec, ModelConfigDict, TensorType
(tf1, tf, tfv) = try_import_tf()

class _MovingMeanStd:
    """Track moving mean, std and count."""

    def __init__(self, epsilon: float=0.0001, shape: Optional[List[int]]=None):
        if False:
            i = 10
            return i + 15
        'Initialize object.\n\n        Args:\n            epsilon: Initial count.\n            shape: Shape of the trackables mean and std.\n        '
        if not shape:
            shape = []
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        'Normalize input batch using moving mean and std.\n\n        Args:\n            inputs: Input batch to normalize.\n\n        Returns:\n            Logarithmic scaled normalized output.\n        '
        batch_mean = np.mean(inputs, axis=0)
        batch_var = np.var(inputs, axis=0)
        batch_count = inputs.shape[0]
        self.update_params(batch_mean, batch_var, batch_count)
        return np.log(inputs / self.std + 1)

    def update_params(self, batch_mean: float, batch_var: float, batch_count: float) -> None:
        if False:
            while True:
                i = 10
        'Update moving mean, std and count.\n\n        Args:\n            batch_mean: Input batch mean.\n            batch_var: Input batch variance.\n            batch_count: Number of cases in the batch.\n        '
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta + batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.power(delta, 2) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    @property
    def std(self) -> float:
        if False:
            print('Hello World!')
        'Get moving standard deviation.\n\n        Returns:\n            Returns moving standard deviation.\n        '
        return np.sqrt(self.var)

@PublicAPI
def update_beta(beta_schedule: str, beta: float, rho: float, step: int) -> float:
    if False:
        i = 10
        return i + 15
    'Update beta based on schedule and training step.\n\n    Args:\n        beta_schedule: Schedule for beta update.\n        beta: Initial beta.\n        rho: Schedule decay parameter.\n        step: Current training iteration.\n\n    Returns:\n        Updated beta as per input schedule.\n    '
    if beta_schedule == 'linear_decay':
        return beta * (1.0 - rho) ** step
    return beta

@PublicAPI
def compute_states_entropy(obs_embeds: np.ndarray, embed_dim: int, k_nn: int) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Compute states entropy using K nearest neighbour method.\n\n    Args:\n        obs_embeds: Observation latent representation using\n            encoder model.\n        embed_dim: Embedding vector dimension.\n        k_nn: Number of nearest neighbour for K-NN estimation.\n\n    Returns:\n        Computed states entropy.\n    '
    obs_embeds_ = np.reshape(obs_embeds, [-1, embed_dim])
    dist = np.linalg.norm(obs_embeds_[:, None, :] - obs_embeds_[None, :, :], axis=-1)
    return dist.argsort(axis=-1)[:, :k_nn][:, -1].astype(np.float32)

@PublicAPI
class RE3(Exploration):
    """Random Encoder for Efficient Exploration.

    Implementation of:
    [1] State entropy maximization with random encoders for efficient
    exploration. Seo, Chen, Shin, Lee, Abbeel, & Lee, (2021).
    arXiv preprint arXiv:2102.09430.

    Estimates state entropy using a particle-based k-nearest neighbors (k-NN)
    estimator in the latent space. The state's latent representation is
    calculated using an encoder with randomly initialized parameters.

    The entropy of a state is considered as intrinsic reward and added to the
    environment's extrinsic reward for policy optimization.
    Entropy is calculated per batch, it does not take the distribution of
    the entire replay buffer into consideration.
    """

    def __init__(self, action_space: Space, *, framework: str, model: ModelV2, embeds_dim: int=128, encoder_net_config: Optional[ModelConfigDict]=None, beta: float=0.2, beta_schedule: str='constant', rho: float=0.1, k_nn: int=50, random_timesteps: int=10000, sub_exploration: Optional[FromConfigSpec]=None, **kwargs):
        if False:
            return 10
        'Initialize RE3.\n\n        Args:\n            action_space: The action space in which to explore.\n            framework: Supports "tf", this implementation does not\n                support torch.\n            model: The policy\'s model.\n            embeds_dim: The dimensionality of the observation embedding\n                vectors in latent space.\n            encoder_net_config: Optional model\n                configuration for the encoder network, producing embedding\n                vectors from observations. This can be used to configure\n                fcnet- or conv_net setups to properly process any\n                observation space.\n            beta: Hyperparameter to choose between exploration and\n                exploitation.\n            beta_schedule: Schedule to use for beta decay, one of\n                "constant" or "linear_decay".\n            rho: Beta decay factor, used for on-policy algorithm.\n            k_nn: Number of neighbours to set for K-NN entropy\n                estimation.\n            random_timesteps: The number of timesteps to act completely\n                randomly (see [1]).\n            sub_exploration: The config dict for the underlying Exploration\n                to use (e.g. epsilon-greedy for DQN). If None, uses the\n                FromSpecDict provided in the Policy\'s default config.\n\n        Raises:\n            ValueError: If the input framework is Torch.\n        '
        if framework == 'torch':
            raise ValueError('This RE3 implementation does not support Torch.')
        super().__init__(action_space, model=model, framework=framework, **kwargs)
        self.beta = beta
        self.rho = rho
        self.k_nn = k_nn
        self.embeds_dim = embeds_dim
        if encoder_net_config is None:
            encoder_net_config = self.policy_config['model'].copy()
        self.encoder_net_config = encoder_net_config
        if sub_exploration is None:
            if isinstance(self.action_space, Discrete):
                sub_exploration = {'type': 'EpsilonGreedy', 'epsilon_schedule': {'type': 'PiecewiseSchedule', 'endpoints': [(0, 1.0), (random_timesteps + 1, 1.0), (random_timesteps + 2, 0.01)], 'outside_value': 0.01}}
            elif isinstance(self.action_space, Box):
                sub_exploration = {'type': 'OrnsteinUhlenbeckNoise', 'random_timesteps': random_timesteps}
            else:
                raise NotImplementedError
        self.sub_exploration = sub_exploration
        self._encoder_net = ModelCatalog.get_model_v2(self.model.obs_space, self.action_space, self.embeds_dim, model_config=self.encoder_net_config, framework=self.framework, name='encoder_net')
        if self.framework == 'tf':
            self._obs_ph = get_placeholder(space=self.model.obs_space, name='_encoder_obs')
            self._obs_embeds = tf.stop_gradient(self._encoder_net({SampleBatch.OBS: self._obs_ph})[0])
        self.exploration_submodule = from_config(cls=Exploration, config=self.sub_exploration, action_space=self.action_space, framework=self.framework, policy_config=self.policy_config, model=self.model, num_workers=self.num_workers, worker_index=self.worker_index)

    @override(Exploration)
    def get_exploration_action(self, *, action_distribution: ActionDistribution, timestep: Union[int, TensorType], explore: bool=True):
        if False:
            return 10
        return self.exploration_submodule.get_exploration_action(action_distribution=action_distribution, timestep=timestep, explore=explore)

    @override(Exploration)
    def postprocess_trajectory(self, policy, sample_batch, tf_sess=None):
        if False:
            print('Hello World!')
        "Calculate states' latent representations/embeddings.\n\n        Embeddings are added to the SampleBatch object such that it doesn't\n        need to be calculated during each training step.\n        "
        if self.framework != 'torch':
            sample_batch = self._postprocess_tf(policy, sample_batch, tf_sess)
        else:
            raise ValueError('Not implemented for Torch.')
        return sample_batch

    def _postprocess_tf(self, policy, sample_batch, tf_sess):
        if False:
            while True:
                i = 10
        "Calculate states' embeddings and add it to SampleBatch."
        if self.framework == 'tf':
            obs_embeds = tf_sess.run(self._obs_embeds, feed_dict={self._obs_ph: sample_batch[SampleBatch.OBS]})
        else:
            obs_embeds = tf.stop_gradient(self._encoder_net({SampleBatch.OBS: sample_batch[SampleBatch.OBS]})[0]).numpy()
        sample_batch[SampleBatch.OBS_EMBEDS] = obs_embeds
        return sample_batch