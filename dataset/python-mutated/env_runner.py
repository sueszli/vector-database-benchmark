"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""
from collections import defaultdict
from functools import partial
from typing import List, Tuple
import gymnasium as gym
import numpy as np
import tree
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.wrappers.atari_wrappers import NoopResetEnv, MaxAndSkipEnv
from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv
from ray.rllib.env.utils import _gym_env_creator
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.numpy import one_hot
from ray.tune.registry import ENV_CREATOR, _global_registry
(_, tf, _) = try_import_tf()

class DreamerV3EnvRunner(EnvRunner):
    """An environment runner to collect data from vectorized gymnasium environments."""

    def __init__(self, config: AlgorithmConfig, **kwargs):
        if False:
            print('Hello World!')
        'Initializes a DreamerV3EnvRunner instance.\n\n        Args:\n            config: The config to use to setup this EnvRunner.\n        '
        super().__init__(config=config)
        if self.config.env.startswith('ALE/'):
            from supersuit.generic_wrappers import resize_v1
            wrappers = [partial(gym.wrappers.TimeLimit, max_episode_steps=108000), partial(resize_v1, x_size=64, y_size=64), NormalizedImageEnv, NoopResetEnv, MaxAndSkipEnv]
            self.env = gym.vector.make('GymV26Environment-v0', env_id=self.config.env, wrappers=wrappers, num_envs=self.config.num_envs_per_worker, asynchronous=self.config.remote_worker_envs, make_kwargs=dict(self.config.env_config, **{'render_mode': 'rgb_array'}))
        elif self.config.env.startswith('DMC/'):
            parts = self.config.env.split('/')
            assert len(parts) == 3, f"ERROR: DMC env must be formatted as 'DMC/[task]/[domain]', e.g. 'DMC/cartpole/swingup'! You provided '{self.config.env}'."
            gym.register('dmc_env-v0', lambda from_pixels=True: DMCEnv(parts[1], parts[2], from_pixels=from_pixels, channels_first=False))
            self.env = gym.vector.make('dmc_env-v0', wrappers=[ActionClip], num_envs=self.config.num_envs_per_worker, asynchronous=self.config.remote_worker_envs, **dict(self.config.env_config))
        else:
            gym.register('dreamerv3-custom-env-v0', partial(_global_registry.get(ENV_CREATOR, self.config.env), self.config.env_config) if _global_registry.contains(ENV_CREATOR, self.config.env) else partial(_gym_env_creator, env_context=self.config.env_config, env_descriptor=self.config.env))
            self.env = gym.vector.make('dreamerv3-custom-env-v0', num_envs=self.config.num_envs_per_worker, asynchronous=False)
        self.num_envs = self.env.num_envs
        assert self.num_envs == self.config.num_envs_per_worker
        if self.config.share_module_between_env_runner_and_learner:
            self.module = None
        else:
            (policy_dict, _) = self.config.get_multi_agent_setup(env=self.env)
            module_spec = self.config.get_marl_module_spec(policy_dict=policy_dict)
            self.module = module_spec.build()[DEFAULT_POLICY_ID]
        self._needs_initial_reset = True
        self._episodes = [None for _ in range(self.num_envs)]
        self._done_episodes_for_metrics = []
        self._ongoing_episodes_for_metrics = defaultdict(list)
        self._ts_since_last_metrics = 0

    @override(EnvRunner)
    def sample(self, *, num_timesteps: int=None, num_episodes: int=None, explore: bool=True, random_actions: bool=False, with_render_data: bool=False) -> Tuple[List[SingleAgentEpisode], List[SingleAgentEpisode]]:
        if False:
            i = 10
            return i + 15
        'Runs and returns a sample (n timesteps or m episodes) on the environment(s).\n\n        Timesteps or episodes are counted in total (across all vectorized\n        sub-environments). For example, if self.num_envs=2 and num_timesteps=10, each\n        sub-environment will be sampled for 5 steps. If self.num_envs=3 and\n        num_episodes=30, each sub-environment will be sampled for 10 episodes.\n\n        Args:\n            num_timesteps: The number of timesteps to sample from the environment(s).\n                Note that only exactly one of `num_timesteps` or `num_episodes` must be\n                provided.\n            num_episodes: The number of full episodes to sample from the environment(s).\n                Note that only exactly one of `num_timesteps` or `num_episodes` must be\n                provided.\n            explore: Indicates whether to utilize exploration when picking actions.\n            random_actions: Whether to only use random actions. If True, the value of\n                `explore` is ignored.\n            force_reset: Whether to reset the environment(s) before starting to sample.\n                If False, will still reset the environment(s) if they were left in\n                a terminated or truncated state during previous sample calls.\n            with_render_data: If True, will record rendering images per timestep\n                in the returned Episodes. This data can be used to create video\n                reports.\n                TODO (sven): Note that this is only supported for runnign with\n                 `num_episodes` yet.\n\n        Returns:\n            A tuple consisting of a) list of Episode instances that are done and\n            b) list of Episode instances that are still ongoing.\n        '
        if num_timesteps is None and num_episodes is None:
            if self.config.batch_mode == 'truncate_episodes':
                num_timesteps = self.config.rollout_fragment_length * self.num_envs
            else:
                num_episodes = self.num_envs
        if num_timesteps is not None:
            return self._sample_timesteps(num_timesteps=num_timesteps, explore=explore, random_actions=random_actions, force_reset=False)
        else:
            return (self._sample_episodes(num_episodes=num_episodes, explore=explore, random_actions=random_actions, with_render_data=with_render_data), [])

    def _sample_timesteps(self, num_timesteps: int, explore: bool=True, random_actions: bool=False, force_reset: bool=False) -> Tuple[List[SingleAgentEpisode], List[SingleAgentEpisode]]:
        if False:
            return 10
        'Helper method to run n timesteps.\n\n        See docstring of self.sample() for more details.\n        '
        done_episodes_to_return = []
        initial_states = tree.map_structure(lambda s: np.repeat(s, self.num_envs, axis=0), self.module.get_initial_state())
        if force_reset or self._needs_initial_reset:
            (obs, _) = self.env.reset()
            self._episodes = [SingleAgentEpisode() for _ in range(self.num_envs)]
            states = initial_states
            is_first = np.ones((self.num_envs,))
            self._needs_initial_reset = False
            for i in range(self.num_envs):
                self._episodes[i].add_initial_observation(initial_observation=obs[i], initial_state={k: s[i] for (k, s) in states.items()})
        else:
            obs = np.stack([eps.observations[-1] for eps in self._episodes])
            states = {k: np.stack([initial_states[k][i] if eps.states is None else eps.states[k] for (i, eps) in enumerate(self._episodes)]) for k in initial_states.keys()}
            is_first = np.zeros((self.num_envs,))
            for (i, eps) in enumerate(self._episodes):
                if eps.states is None:
                    is_first[i] = 1.0
        ts = 0
        while ts < num_timesteps:
            if random_actions:
                actions = self.env.action_space.sample()
            else:
                batch = {STATE_IN: tree.map_structure(lambda s: tf.convert_to_tensor(s), states), SampleBatch.OBS: tf.convert_to_tensor(obs), 'is_first': tf.convert_to_tensor(is_first)}
                if explore:
                    outs = self.module.forward_exploration(batch)
                else:
                    outs = self.module.forward_inference(batch)
                actions = outs[SampleBatch.ACTIONS].numpy()
                if isinstance(self.env.single_action_space, gym.spaces.Discrete):
                    actions = np.argmax(actions, axis=-1)
                states = tree.map_structure(lambda s: s.numpy(), outs[STATE_OUT])
            (obs, rewards, terminateds, truncateds, infos) = self.env.step(actions)
            ts += self.num_envs
            for i in range(self.num_envs):
                s = {k: s[i] for (k, s) in states.items()}
                if terminateds[i] or truncateds[i]:
                    self._episodes[i].add_timestep(infos['final_observation'][i], actions[i], rewards[i], state=s, is_terminated=terminateds[i], is_truncated=truncateds[i])
                    for (k, v) in self.module.get_initial_state().items():
                        states[k][i] = v.numpy()
                    is_first[i] = True
                    done_episodes_to_return.append(self._episodes[i])
                    self._episodes[i] = SingleAgentEpisode(observations=[obs[i]], states=s)
                else:
                    self._episodes[i].add_timestep(obs[i], actions[i], rewards[i], state=s)
                    is_first[i] = False
        self._done_episodes_for_metrics.extend(done_episodes_to_return)
        ongoing_episodes = self._episodes
        self._episodes = [eps.create_successor() for eps in self._episodes]
        for eps in ongoing_episodes:
            self._ongoing_episodes_for_metrics[eps.id_].append(eps)
        self._ts_since_last_metrics += ts
        return (done_episodes_to_return, ongoing_episodes)

    def _sample_episodes(self, num_episodes: int, explore: bool=True, random_actions: bool=False, with_render_data: bool=False) -> List[SingleAgentEpisode]:
        if False:
            print('Hello World!')
        'Helper method to run n episodes.\n\n        See docstring of `self.sample()` for more details.\n        '
        done_episodes_to_return = []
        (obs, _) = self.env.reset()
        episodes = [SingleAgentEpisode() for _ in range(self.num_envs)]
        states = tree.map_structure(lambda s: np.repeat(s, self.num_envs, axis=0), self.module.get_initial_state())
        is_first = np.ones((self.num_envs,))
        render_images = [None] * self.num_envs
        if with_render_data:
            render_images = [e.render() for e in self.env.envs]
        for i in range(self.num_envs):
            episodes[i].add_initial_observation(initial_observation=obs[i], initial_state={k: s[i] for (k, s) in states.items()}, initial_render_image=render_images[i])
        eps = 0
        while eps < num_episodes:
            if random_actions:
                actions = self.env.action_space.sample()
            else:
                batch = {STATE_IN: tree.map_structure(lambda s: tf.convert_to_tensor(s), states), SampleBatch.OBS: tf.convert_to_tensor(obs), 'is_first': tf.convert_to_tensor(is_first)}
                if explore:
                    outs = self.module.forward_exploration(batch)
                else:
                    outs = self.module.forward_inference(batch)
                actions = outs[SampleBatch.ACTIONS].numpy()
                if isinstance(self.env.single_action_space, gym.spaces.Discrete):
                    actions = np.argmax(actions, axis=-1)
                states = tree.map_structure(lambda s: s.numpy(), outs[STATE_OUT])
            (obs, rewards, terminateds, truncateds, infos) = self.env.step(actions)
            if with_render_data:
                render_images = [e.render() for e in self.env.envs]
            for i in range(self.num_envs):
                s = {k: s[i] for (k, s) in states.items()}
                if terminateds[i] or truncateds[i]:
                    eps += 1
                    episodes[i].add_timestep(infos['final_observation'][i], actions[i], rewards[i], state=s, is_terminated=terminateds[i], is_truncated=truncateds[i])
                    done_episodes_to_return.append(episodes[i])
                    if eps == num_episodes:
                        break
                    for (k, v) in self.module.get_initial_state().items():
                        states[k][i] = v.numpy()
                    is_first[i] = True
                    episodes[i] = SingleAgentEpisode(observations=[obs[i]], states=s, render_images=[render_images[i]])
                else:
                    episodes[i].add_timestep(obs[i], actions[i], rewards[i], state=s, render_image=render_images[i])
                    is_first[i] = False
        self._done_episodes_for_metrics.extend(done_episodes_to_return)
        self._ts_since_last_metrics += sum((len(eps) for eps in done_episodes_to_return))
        self._needs_initial_reset = True
        return done_episodes_to_return

    def get_metrics(self) -> List[RolloutMetrics]:
        if False:
            print('Hello World!')
        metrics = []
        for eps in self._done_episodes_for_metrics:
            episode_length = len(eps)
            episode_reward = eps.get_return()
            if eps.id_ in self._ongoing_episodes_for_metrics:
                for eps2 in self._ongoing_episodes_for_metrics[eps.id_]:
                    episode_length += len(eps2)
                    episode_reward += eps2.get_return()
                del self._ongoing_episodes_for_metrics[eps.id_]
            metrics.append(RolloutMetrics(episode_length=episode_length, episode_reward=episode_reward))
        self._done_episodes_for_metrics.clear()
        self._ts_since_last_metrics = 0
        return metrics

    def set_weights(self, weights, global_vars=None):
        if False:
            for i in range(10):
                print('nop')
        'Writes the weights of our (single-agent) RLModule.'
        if self.module is None:
            assert self.config.share_module_between_env_runner_and_learner
        else:
            self.module.set_state(weights[DEFAULT_POLICY_ID])

    @override(EnvRunner)
    def assert_healthy(self):
        if False:
            return 10
        assert self.env and self.module

    @override(EnvRunner)
    def stop(self):
        if False:
            while True:
                i = 10
        self.env.close()

class NormalizedImageEnv(gym.ObservationWrapper):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=self.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        if False:
            return 10
        return observation.astype(np.float32) / 128.0 - 1.0

class OneHot(gym.ObservationWrapper):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(self.observation_space.n,), dtype=np.float32)

    def reset(self, **kwargs):
        if False:
            i = 10
            return i + 15
        ret = self.env.reset(**kwargs)
        return (self._get_obs(ret[0]), ret[1])

    def step(self, action):
        if False:
            for i in range(10):
                print('nop')
        ret = self.env.step(action)
        return (self._get_obs(ret[0]), ret[1], ret[2], ret[3], ret[4])

    def _get_obs(self, obs):
        if False:
            return 10
        return one_hot(obs, depth=self.observation_space.shape[0])

class ActionClip(gym.ActionWrapper):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self._low = -1.0
        self._high = 1.0
        self.action_space = gym.spaces.Box(self._low, self._high, self.action_space.shape, self.action_space.dtype)

    def action(self, action):
        if False:
            for i in range(10):
                print('nop')
        return np.clip(action, self._low, self._high)