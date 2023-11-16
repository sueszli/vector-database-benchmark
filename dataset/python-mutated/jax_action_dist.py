import time
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.framework import try_import_jax, try_import_tfp
from ray.rllib.utils.typing import TensorType, List
from ray.util import log_once
(jax, flax) = try_import_jax()
tfp = try_import_tfp()

class JAXDistribution(ActionDistribution):
    """Wrapper class for JAX distributions."""

    @override(ActionDistribution)
    def __init__(self, inputs: List[TensorType], model: ModelV2):
        if False:
            print('Hello World!')
        super().__init__(inputs, model)
        if log_once('jax_distribution_deprecation_warning'):
            deprecation_warning(old='ray.rllib.models.jax.jax_action_dist.JAXDistribution')
        self.last_sample = None
        self.prng_key = jax.random.PRNGKey(seed=int(time.time()))

    @override(ActionDistribution)
    def logp(self, actions: TensorType) -> TensorType:
        if False:
            while True:
                i = 10
        return self.dist.log_prob(actions)

    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        if False:
            i = 10
            return i + 15
        return self.dist.entropy()

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution) -> TensorType:
        if False:
            print('Hello World!')
        return self.dist.kl_divergence(other.dist)

    @override(ActionDistribution)
    def sample(self) -> TensorType:
        if False:
            while True:
                i = 10
        (_, self.prng_key) = jax.random.split(self.prng_key)
        self.last_sample = jax.random.categorical(self.prng_key, self.inputs)
        return self.last_sample

    @override(ActionDistribution)
    def sampled_action_logp(self) -> TensorType:
        if False:
            i = 10
            return i + 15
        assert self.last_sample is not None
        return self.logp(self.last_sample)

class JAXCategorical(JAXDistribution):
    """Wrapper class for a JAX Categorical distribution."""

    @override(ActionDistribution)
    def __init__(self, inputs, model=None, temperature=1.0):
        if False:
            return 10
        if temperature != 1.0:
            assert temperature > 0.0, 'Categorical `temperature` must be > 0.0!'
            inputs /= temperature
        super().__init__(inputs, model)
        self.dist = tfp.experimental.substrates.jax.distributions.Categorical(logits=self.inputs)

    @override(ActionDistribution)
    def deterministic_sample(self):
        if False:
            i = 10
            return i + 15
        self.last_sample = self.inputs.argmax(axis=1)
        return self.last_sample

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        if False:
            return 10
        return action_space.n