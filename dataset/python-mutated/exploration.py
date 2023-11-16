from gymnasium.spaces import Space
from typing import Dict, List, Optional, Union, TYPE_CHECKING
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.framework import try_import_torch, TensorType
from ray.rllib.utils.typing import LocalOptimizer, AlgorithmConfigDict
if TYPE_CHECKING:
    from ray.rllib.policy.policy import Policy
    from ray.rllib.utils import try_import_tf
    (_, tf, _) = try_import_tf()
(_, nn) = try_import_torch()

@PublicAPI
class Exploration:
    """Implements an exploration strategy for Policies.

    An Exploration takes model outputs, a distribution, and a timestep from
    the agent and computes an action to apply to the environment using an
    implemented exploration schema.
    """

    def __init__(self, action_space: Space, *, framework: str, policy_config: AlgorithmConfigDict, model: ModelV2, num_workers: int, worker_index: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            action_space: The action space in which to explore.\n            framework: One of "tf" or "torch".\n            policy_config: The Policy\'s config dict.\n            model: The Policy\'s model.\n            num_workers: The overall number of workers used.\n            worker_index: The index of the worker using this class.\n        '
        self.action_space = action_space
        self.policy_config = policy_config
        self.model = model
        self.num_workers = num_workers
        self.worker_index = worker_index
        self.framework = framework
        self.device = None
        if isinstance(self.model, nn.Module):
            params = list(self.model.parameters())
            if params:
                self.device = params[0].device

    @DeveloperAPI
    def before_compute_actions(self, *, timestep: Optional[Union[TensorType, int]]=None, explore: Optional[Union[TensorType, bool]]=None, tf_sess: Optional['tf.Session']=None, **kwargs):
        if False:
            while True:
                i = 10
        'Hook for preparations before policy.compute_actions() is called.\n\n        Args:\n            timestep: An optional timestep tensor.\n            explore: An optional explore boolean flag.\n            tf_sess: The tf-session object to use.\n            **kwargs: Forward compatibility kwargs.\n        '
        pass

    @DeveloperAPI
    def get_exploration_action(self, *, action_distribution: ActionDistribution, timestep: Union[TensorType, int], explore: bool=True):
        if False:
            return 10
        'Returns a (possibly) exploratory action and its log-likelihood.\n\n        Given the Model\'s logits outputs and action distribution, returns an\n        exploratory action.\n\n        Args:\n            action_distribution: The instantiated\n                ActionDistribution object to work with when creating\n                exploration actions.\n            timestep: The current sampling time step. It can be a tensor\n                for TF graph mode, otherwise an integer.\n            explore: True: "Normal" exploration behavior.\n                False: Suppress all exploratory behavior and return\n                a deterministic action.\n\n        Returns:\n            A tuple consisting of 1) the chosen exploration action or a\n            tf-op to fetch the exploration action from the graph and\n            2) the log-likelihood of the exploration action.\n        '
        pass

    @DeveloperAPI
    def on_episode_start(self, policy: 'Policy', *, environment: BaseEnv=None, episode: int=None, tf_sess: Optional['tf.Session']=None):
        if False:
            while True:
                i = 10
        'Handles necessary exploration logic at the beginning of an episode.\n\n        Args:\n            policy: The Policy object that holds this Exploration.\n            environment: The environment object we are acting in.\n            episode: The number of the episode that is starting.\n            tf_sess: In case of tf, the session object.\n        '
        pass

    @DeveloperAPI
    def on_episode_end(self, policy: 'Policy', *, environment: BaseEnv=None, episode: int=None, tf_sess: Optional['tf.Session']=None):
        if False:
            while True:
                i = 10
        'Handles necessary exploration logic at the end of an episode.\n\n        Args:\n            policy: The Policy object that holds this Exploration.\n            environment: The environment object we are acting in.\n            episode: The number of the episode that is starting.\n            tf_sess: In case of tf, the session object.\n        '
        pass

    @DeveloperAPI
    def postprocess_trajectory(self, policy: 'Policy', sample_batch: SampleBatch, tf_sess: Optional['tf.Session']=None):
        if False:
            i = 10
            return i + 15
        'Handles post-processing of done episode trajectories.\n\n        Changes the given batch in place. This callback is invoked by the\n        sampler after policy.postprocess_trajectory() is called.\n\n        Args:\n            policy: The owning policy object.\n            sample_batch: The SampleBatch object to post-process.\n            tf_sess: An optional tf.Session object.\n        '
        return sample_batch

    @DeveloperAPI
    def get_exploration_optimizer(self, optimizers: List[LocalOptimizer]) -> List[LocalOptimizer]:
        if False:
            while True:
                i = 10
        "May add optimizer(s) to the Policy's own `optimizers`.\n\n        The number of optimizers (Policy's plus Exploration's optimizers) must\n        match the number of loss terms produced by the Policy's loss function\n        and the Exploration component's loss terms.\n\n        Args:\n            optimizers: The list of the Policy's local optimizers.\n\n        Returns:\n            The updated list of local optimizers to use on the different\n            loss terms.\n        "
        return optimizers

    @DeveloperAPI
    def get_state(self, sess: Optional['tf.Session']=None) -> Dict[str, TensorType]:
        if False:
            for i in range(10):
                print('nop')
        "Returns the current exploration state.\n\n        Args:\n            sess: An optional tf Session object to use.\n\n        Returns:\n            The Exploration object's current state.\n        "
        return {}

    @DeveloperAPI
    def set_state(self, state: object, sess: Optional['tf.Session']=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Sets the Exploration object's state to the given values.\n\n        Note that some exploration components are stateless, even though they\n        decay some values over time (e.g. EpsilonGreedy). However the decay is\n        only dependent on the current global timestep of the policy and we\n        therefore don't need to keep track of it.\n\n        Args:\n            state: The state to set this Exploration to.\n            sess: An optional tf Session object to use.\n        "
        pass