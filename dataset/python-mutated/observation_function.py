from typing import Dict
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.utils.framework import TensorType
from ray.rllib.utils.typing import AgentID, PolicyID

class ObservationFunction:
    """Interceptor function for rewriting observations from the environment.

    These callbacks can be used for preprocessing of observations, especially
    in multi-agent scenarios.

    Observation functions can be specified in the multi-agent config by
    specifying ``{"observation_fn": your_obs_func}``. Note that
    ``your_obs_func`` can be a plain Python function.

    This API is **experimental**.
    """

    def __call__(self, agent_obs: Dict[AgentID, TensorType], worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode, **kw) -> Dict[AgentID, TensorType]:
        if False:
            print('Hello World!')
        'Callback run on each environment step to observe the environment.\n\n        This method takes in the original agent observation dict returned by\n        a MultiAgentEnv, and returns a possibly modified one. It can be\n        thought of as a "wrapper" around the environment.\n\n        TODO(ekl): allow end-to-end differentiation through the observation\n            function and policy losses.\n\n        TODO(ekl): enable batch processing.\n\n        Args:\n            agent_obs: Dictionary of default observations from the\n                environment. The default implementation of observe() simply\n                returns this dict.\n            worker: Reference to the current rollout worker.\n            base_env: BaseEnv running the episode. The underlying\n                sub environment objects (BaseEnvs are vectorized) can be\n                retrieved by calling `base_env.get_sub_environments()`.\n            policies: Mapping of policy id to policy objects. In single\n                agent mode there will only be a single "default" policy.\n            episode: Episode state object.\n            kwargs: Forward compatibility placeholder.\n\n        Returns:\n            new_agent_obs: copy of agent obs with updates. You can\n                rewrite or drop data from the dict if needed (e.g., the env\n                can have a dummy "global" observation, and the observer can\n                merge the global state into individual observations.\n\n        .. testcode::\n            :skipif: True\n\n            # Observer that merges global state into individual obs. It is\n            # rewriting the discrete obs into a tuple with global state.\n            example_obs_fn1({"a": 1, "b": 2, "global_state": 101}, ...)\n\n        .. testoutput::\n\n            {"a": [1, 101], "b": [2, 101]}\n\n        .. testcode::\n            :skipif: True\n\n            # Observer for e.g., custom centralized critic model. It is\n            # rewriting the discrete obs into a dict with more data.\n            example_obs_fn2({"a": 1, "b": 2}, ...)\n\n        .. testoutput::\n\n            {"a": {"self": 1, "other": 2}, "b": {"self": 2, "other": 1}}\n        '
        return agent_obs