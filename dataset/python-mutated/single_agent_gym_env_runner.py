from typing import List, Optional, Tuple
import gymnasium as gym
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.annotations import override

class SingleAgentGymEnvRunner(EnvRunner):
    """A simple single-agent EnvRunner subclass for testing purposes.

    Uses a gym.vector.Env environment and random actions.
    """

    def __init__(self, *, config: AlgorithmConfig, **kwargs):
        if False:
            while True:
                i = 10
        'Initializes a SingleAgentGymEnvRunner instance.\n\n        Args:\n            config: The config to use to setup this EnvRunner.\n        '
        super().__init__(config=config, **kwargs)
        self.env = gym.vector.make(self.config.env, num_envs=self.config.num_envs_per_worker, asynchronous=self.config.remote_worker_envs, **dict(self.config.env_config, **{'render_mode': 'rgb_array'}))
        self.num_envs = self.env.num_envs
        self._needs_initial_reset = True
        self._episodes = [None for _ in range(self.num_envs)]

    @override(EnvRunner)
    def sample(self, *, num_timesteps: Optional[int]=None, num_episodes: Optional[int]=None, force_reset: bool=False, **kwargs) -> Tuple[List[SingleAgentEpisode], List[SingleAgentEpisode]]:
        if False:
            print('Hello World!')
        'Returns a tuple (list of completed episodes, list of ongoing episodes).\n\n        Args:\n            num_timesteps: If provided, will step exactly this number of timesteps\n                through the environment. Note that only one or none of `num_timesteps`\n                and `num_episodes` may be provided, but never both. If both\n                `num_timesteps` and `num_episodes` are None, will determine how to\n                sample via `self.config`.\n            num_episodes: If provided, will step through the env(s) until exactly this\n                many episodes have been completed. Note that only one or none of\n                `num_timesteps` and `num_episodes` may be provided, but never both.\n                If both `num_timesteps` and `num_episodes` are None, will determine how\n                to sample via `self.config`.\n            force_reset: If True, will force-reset the env at the very beginning and\n                thus begin sampling from freshly started episodes.\n            **kwargs: Forward compatibility kwargs.\n\n        Returns:\n            A tuple consisting of: A list of SingleAgentEpisode instances that are\n            already done (either terminated or truncated, hence their `is_done` property\n            is True), a list of SingleAgentEpisode instances that are still ongoing\n            (their `is_done` property is False).\n        '
        assert not (num_timesteps is not None and num_episodes is not None)
        if num_timesteps is None and num_episodes is None:
            if self.config.batch_mode == 'truncate_episodes':
                num_timesteps = self.config.rollout_fragment_length * self.num_envs
            else:
                num_episodes = self.num_envs
        if num_timesteps is not None:
            return self._sample_timesteps(num_timesteps=num_timesteps, force_reset=force_reset)
        else:
            return self._sample_episodes(num_episodes=num_episodes)

    def _sample_timesteps(self, num_timesteps: int, force_reset: bool=False) -> Tuple[List[SingleAgentEpisode], List[SingleAgentEpisode]]:
        if False:
            for i in range(10):
                print('nop')
        'Runs n timesteps on the environment(s) and returns experiences.\n\n        Timesteps are counted in total (across all vectorized sub-environments). For\n        example, if self.num_envs=2 and num_timesteps=10, each sub-environment\n        will be sampled for 5 steps.\n        '
        done_episodes_to_return = []
        if force_reset or self._needs_initial_reset:
            (obs, _) = self.env.reset()
            self._episodes = [SingleAgentEpisode(observations=[o]) for o in self._split_by_env(obs)]
            self._needs_initial_reset = False
        ts = 0
        while ts < num_timesteps:
            actions = self.env.action_space.sample()
            (obs, rewards, terminateds, truncateds, infos) = self.env.step(actions)
            ts += self.num_envs
            for (i, (o, a, r, term, trunc)) in enumerate(zip(self._split_by_env(obs), self._split_by_env(actions), self._split_by_env(rewards), self._split_by_env(terminateds), self._split_by_env(truncateds))):
                if term or trunc:
                    self._episodes[i].add_timestep(infos['final_observation'][i], a, r, is_terminated=term, is_truncated=trunc)
                    done_episodes_to_return.append(self._episodes[i])
                    self._episodes[i] = SingleAgentEpisode(observations=[o])
                else:
                    self._episodes[i].add_timestep(o, a, r)
        ongoing_episodes = self._episodes
        self._episodes = [SingleAgentEpisode(id_=eps.id_, observations=[eps.observations[-1]]) for eps in self._episodes]
        return (done_episodes_to_return, ongoing_episodes)

    def _sample_episodes(self, num_episodes: int):
        if False:
            while True:
                i = 10
        'Runs n episodes (reset first) on the environment(s) and returns experiences.\n\n        Episodes are counted in total (across all vectorized sub-environments). For\n        example, if self.num_envs=2 and num_episodes=10, each sub-environment\n        will run 5 episodes.\n        '
        done_episodes_to_return = []
        (obs, _) = self.env.reset()
        episodes = [SingleAgentEpisode(observations=[o]) for o in self._split_by_env(obs)]
        eps = 0
        while eps < num_episodes:
            actions = self.env.action_space.sample()
            (obs, rewards, terminateds, truncateds, infos) = self.env.step(actions)
            for (i, (o, a, r, term, trunc)) in enumerate(zip(self._split_by_env(obs), self._split_by_env(actions), self._split_by_env(rewards), self._split_by_env(terminateds), self._split_by_env(truncateds))):
                if term or trunc:
                    eps += 1
                    episodes[i].add_timestep(infos['final_observation'][i], a, r, is_terminated=term, is_truncated=trunc)
                    done_episodes_to_return.append(episodes[i])
                    if eps == num_episodes:
                        break
                    episodes[i] = SingleAgentEpisode(observations=[o])
                else:
                    episodes[i].add_timestep(o, a, r)
        self._needs_initial_reset = True
        return (done_episodes_to_return, [])

    @override(EnvRunner)
    def assert_healthy(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.env

    @override(EnvRunner)
    def stop(self):
        if False:
            return 10
        self.env.close()

    def _split_by_env(self, inputs):
        if False:
            return 10
        return [inputs[i] for i in range(self.num_envs)]