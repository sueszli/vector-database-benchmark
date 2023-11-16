import copy
import warnings
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy

class HerReplayBuffer(DictReplayBuffer):
    """
    Hindsight Experience Replay (HER) buffer.
    Paper: https://arxiv.org/abs/1707.01495

    Replay buffer for sampling HER (Hindsight Experience Replay) transitions.

    .. note::

      Compared to other implementations, the ``future`` goal sampling strategy is inclusive:
      the current transition can be used when re-sampling.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param env: The training environment
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    :param n_sampled_goal: Number of virtual transitions to create per real transition,
        by sampling new goals.
    :param goal_selection_strategy: Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future']
    :param copy_info_dict: Whether to copy the info dictionary and pass it to
        ``compute_reward()`` method.
        Please note that the copy may cause a slowdown.
        False by default.
    """
    env: Optional[VecEnv]

    def __init__(self, buffer_size: int, observation_space: spaces.Dict, action_space: spaces.Space, env: VecEnv, device: Union[th.device, str]='auto', n_envs: int=1, optimize_memory_usage: bool=False, handle_timeout_termination: bool=True, n_sampled_goal: int=4, goal_selection_strategy: Union[GoalSelectionStrategy, str]='future', copy_info_dict: bool=False):
        if False:
            i = 10
            return i + 15
        super().__init__(buffer_size, observation_space, action_space, device=device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)
        self.env = env
        self.copy_info_dict = copy_info_dict
        if isinstance(goal_selection_strategy, str):
            self.goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy.lower()]
        else:
            self.goal_selection_strategy = goal_selection_strategy
        assert isinstance(self.goal_selection_strategy, GoalSelectionStrategy), f'Invalid goal selection strategy, please use one of {list(GoalSelectionStrategy)}'
        self.n_sampled_goal = n_sampled_goal
        self.her_ratio = 1 - 1.0 / (self.n_sampled_goal + 1)
        self.infos = np.array([[{} for _ in range(self.n_envs)] for _ in range(self.buffer_size)])
        self.ep_start = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.ep_length = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self._current_ep_start = np.zeros(self.n_envs, dtype=np.int64)

    def __getstate__(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Gets state for pickling.\n\n        Excludes self.env, as in general Env's may not be pickleable.\n        "
        state = self.__dict__.copy()
        del state['env']
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        if False:
            return 10
        '\n        Restores pickled state.\n\n        User must call ``set_env()`` after unpickling before using.\n\n        :param state:\n        '
        self.__dict__.update(state)
        assert 'env' not in state
        self.env = None

    def set_env(self, env: VecEnv) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Sets the environment.\n\n        :param env:\n        '
        if self.env is not None:
            raise ValueError('Trying to set env of already initialized environment.')
        self.env = env

    def add(self, obs: Dict[str, np.ndarray], next_obs: Dict[str, np.ndarray], action: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        for env_idx in range(self.n_envs):
            episode_start = self.ep_start[self.pos, env_idx]
            episode_length = self.ep_length[self.pos, env_idx]
            if episode_length > 0:
                episode_end = episode_start + episode_length
                episode_indices = np.arange(self.pos, episode_end) % self.buffer_size
                self.ep_length[episode_indices, env_idx] = 0
        self.ep_start[self.pos] = self._current_ep_start.copy()
        if self.copy_info_dict:
            self.infos[self.pos] = infos
        super().add(obs, next_obs, action, reward, done, infos)
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                self._compute_episode_length(env_idx)

    def _compute_episode_length(self, env_idx: int) -> None:
        if False:
            return 10
        '\n        Compute and store the episode length for environment with index env_idx\n\n        :param env_idx: index of the environment for which the episode length should be computed\n        '
        episode_start = self._current_ep_start[env_idx]
        episode_end = self.pos
        if episode_end < episode_start:
            episode_end += self.buffer_size
        episode_indices = np.arange(episode_start, episode_end) % self.buffer_size
        self.ep_length[episode_indices, env_idx] = episode_end - episode_start
        self._current_ep_start[env_idx] = self.pos

    def sample(self, batch_size: int, env: Optional[VecNormalize]=None) -> DictReplayBufferSamples:
        if False:
            i = 10
            return i + 15
        '\n        Sample elements from the replay buffer.\n\n        :param batch_size: Number of element to sample\n        :param env: Associated VecEnv to normalize the observations/rewards when sampling\n        :return: Samples\n        '
        is_valid = self.ep_length > 0
        if not np.any(is_valid):
            raise RuntimeError('Unable to sample before the end of the first episode. We recommend choosing a value for learning_starts that is greater than the maximum number of timesteps in the environment.')
        valid_indices = np.flatnonzero(is_valid)
        sampled_indices = np.random.choice(valid_indices, size=batch_size, replace=True)
        (batch_indices, env_indices) = np.unravel_index(sampled_indices, is_valid.shape)
        nb_virtual = int(self.her_ratio * batch_size)
        (virtual_batch_indices, real_batch_indices) = np.split(batch_indices, [nb_virtual])
        (virtual_env_indices, real_env_indices) = np.split(env_indices, [nb_virtual])
        real_data = self._get_real_samples(real_batch_indices, real_env_indices, env)
        virtual_data = self._get_virtual_samples(virtual_batch_indices, virtual_env_indices, env)
        observations = {key: th.cat((real_data.observations[key], virtual_data.observations[key])) for key in virtual_data.observations.keys()}
        actions = th.cat((real_data.actions, virtual_data.actions))
        next_observations = {key: th.cat((real_data.next_observations[key], virtual_data.next_observations[key])) for key in virtual_data.next_observations.keys()}
        dones = th.cat((real_data.dones, virtual_data.dones))
        rewards = th.cat((real_data.rewards, virtual_data.rewards))
        return DictReplayBufferSamples(observations=observations, actions=actions, next_observations=next_observations, dones=dones, rewards=rewards)

    def _get_real_samples(self, batch_indices: np.ndarray, env_indices: np.ndarray, env: Optional[VecNormalize]=None) -> DictReplayBufferSamples:
        if False:
            while True:
                i = 10
        '\n        Get the samples corresponding to the batch and environment indices.\n\n        :param batch_indices: Indices of the transitions\n        :param env_indices: Indices of the envrionments\n        :param env: associated gym VecEnv to normalize the\n            observations/rewards when sampling, defaults to None\n        :return: Samples\n        '
        obs_ = self._normalize_obs({key: obs[batch_indices, env_indices, :] for (key, obs) in self.observations.items()}, env)
        next_obs_ = self._normalize_obs({key: obs[batch_indices, env_indices, :] for (key, obs) in self.next_observations.items()}, env)
        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)
        observations = {key: self.to_torch(obs) for (key, obs) in obs_.items()}
        next_observations = {key: self.to_torch(obs) for (key, obs) in next_obs_.items()}
        return DictReplayBufferSamples(observations=observations, actions=self.to_torch(self.actions[batch_indices, env_indices]), next_observations=next_observations, dones=self.to_torch(self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])).reshape(-1, 1), rewards=self.to_torch(self._normalize_reward(self.rewards[batch_indices, env_indices].reshape(-1, 1), env)))

    def _get_virtual_samples(self, batch_indices: np.ndarray, env_indices: np.ndarray, env: Optional[VecNormalize]=None) -> DictReplayBufferSamples:
        if False:
            i = 10
            return i + 15
        '\n        Get the samples, sample new desired goals and compute new rewards.\n\n        :param batch_indices: Indices of the transitions\n        :param env_indices: Indices of the envrionments\n        :param env: associated gym VecEnv to normalize the\n            observations/rewards when sampling, defaults to None\n        :return: Samples, with new desired goals and new rewards\n        '
        obs = {key: obs[batch_indices, env_indices, :] for (key, obs) in self.observations.items()}
        next_obs = {key: obs[batch_indices, env_indices, :] for (key, obs) in self.next_observations.items()}
        if self.copy_info_dict:
            infos = copy.deepcopy(self.infos[batch_indices, env_indices])
        else:
            infos = [{} for _ in range(len(batch_indices))]
        new_goals = self._sample_goals(batch_indices, env_indices)
        obs['desired_goal'] = new_goals
        next_obs['desired_goal'] = new_goals
        assert self.env is not None, 'You must initialize HerReplayBuffer with a VecEnv so it can compute rewards for virtual transitions'
        rewards = self.env.env_method('compute_reward', next_obs['achieved_goal'], obs['desired_goal'], infos, indices=[0])
        rewards = rewards[0].astype(np.float32)
        obs = self._normalize_obs(obs, env)
        next_obs = self._normalize_obs(next_obs, env)
        observations = {key: self.to_torch(obs) for (key, obs) in obs.items()}
        next_observations = {key: self.to_torch(obs) for (key, obs) in next_obs.items()}
        return DictReplayBufferSamples(observations=observations, actions=self.to_torch(self.actions[batch_indices, env_indices]), next_observations=next_observations, dones=self.to_torch(self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])).reshape(-1, 1), rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)))

    def _sample_goals(self, batch_indices: np.ndarray, env_indices: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Sample goals based on goal_selection_strategy.\n\n        :param batch_indices: Indices of the transitions\n        :param env_indices: Indices of the envrionments\n        :return: Sampled goals\n        '
        batch_ep_start = self.ep_start[batch_indices, env_indices]
        batch_ep_length = self.ep_length[batch_indices, env_indices]
        if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            transition_indices_in_episode = batch_ep_length - 1
        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            current_indices_in_episode = (batch_indices - batch_ep_start) % self.buffer_size
            transition_indices_in_episode = np.random.randint(current_indices_in_episode, batch_ep_length)
        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            transition_indices_in_episode = np.random.randint(0, batch_ep_length)
        else:
            raise ValueError(f'Strategy {self.goal_selection_strategy} for sampling goals not supported!')
        transition_indices = (transition_indices_in_episode + batch_ep_start) % self.buffer_size
        return self.next_observations['achieved_goal'][transition_indices, env_indices]

    def truncate_last_trajectory(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        If called, we assume that the last trajectory in the replay buffer was finished\n        (and truncate it).\n        If not called, we assume that we continue the same trajectory (same episode).\n        '
        if (self._current_ep_start != self.pos).any():
            warnings.warn('The last trajectory in the replay buffer will be truncated.\nIf you are in the same episode as when the replay buffer was saved,\nyou should use `truncate_last_trajectory=False` to avoid that issue.')
            for env_idx in np.where(self._current_ep_start != self.pos)[0]:
                self.dones[self.pos - 1, env_idx] = True
                self._compute_episode_length(env_idx)
                if self.handle_timeout_termination:
                    self.timeouts[self.pos - 1, env_idx] = True