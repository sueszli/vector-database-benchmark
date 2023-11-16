"""Examples for recommender system simulating envs ready to be used by RLlib Algorithms.

This env follows RecSim obs and action APIs.
"""
import gymnasium as gym
import numpy as np
from typing import Optional
from ray.rllib.utils.numpy import softmax

class ParametricRecSys(gym.Env):
    """A recommendation environment which generates items with visible features
    randomly (parametric actions).
    The environment can be configured to be multi-user, i.e. different models
    will be learned independently for each user, by setting num_users_in_db
    parameter.
    To enable slate recommendation, the `slate_size` config parameter can be
    set as > 1.
    """

    def __init__(self, embedding_size: int=20, num_docs_to_select_from: int=10, slate_size: int=1, num_docs_in_db: Optional[int]=None, num_users_in_db: Optional[int]=None, user_time_budget: float=60.0):
        if False:
            print('Hello World!')
        'Initializes a ParametricRecSys instance.\n\n        Args:\n            embedding_size: Embedding size for both users and docs.\n                Each value in the user/doc embeddings can have values between\n                -1.0 and 1.0.\n            num_docs_to_select_from: The number of documents to present to the\n                agent each timestep. The agent will then have to pick a slate\n                out of these.\n            slate_size: The size of the slate to recommend to the user at each\n                timestep.\n            num_docs_in_db: The total number of documents in the DB. Set this\n                to None, in case you would like to resample docs from an\n                infinite pool.\n            num_users_in_db: The total number of users in the DB. Set this to\n                None, in case you would like to resample users from an infinite\n                pool.\n            user_time_budget: The total time budget a user has throughout an\n                episode. Once this time budget is used up (through engagements\n                with clicked/selected documents), the episode ends.\n        '
        self.embedding_size = embedding_size
        self.num_docs_to_select_from = num_docs_to_select_from
        self.slate_size = slate_size
        self.num_docs_in_db = num_docs_in_db
        self.docs_db = None
        self.num_users_in_db = num_users_in_db
        self.users_db = None
        self.current_user = None
        self.user_time_budget = user_time_budget
        self.current_user_budget = user_time_budget
        self.observation_space = gym.spaces.Dict({'doc': gym.spaces.Dict({str(i): gym.spaces.Box(-1.0, 1.0, shape=(self.embedding_size,), dtype=np.float32) for i in range(self.num_docs_to_select_from)}), 'user': gym.spaces.Box(-1.0, 1.0, shape=(self.embedding_size,), dtype=np.float32), 'response': gym.spaces.Tuple([gym.spaces.Dict({'click': gym.spaces.Discrete(2), 'engagement': gym.spaces.Box(0.0, 100.0, shape=(), dtype=np.float32)}) for _ in range(self.slate_size)])})
        self.action_space = gym.spaces.MultiDiscrete([self.num_docs_to_select_from for _ in range(self.slate_size)])

    def _get_embedding(self):
        if False:
            return 10
        return np.random.uniform(-1, 1, size=(self.embedding_size,)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if False:
            for i in range(10):
                print('nop')
        self.current_user_budget = self.user_time_budget
        if self.num_users_in_db is not None:
            if self.users_db is None:
                self.users_db = [self._get_embedding() for _ in range(self.num_users_in_db)]
            self.current_user = self.users_db[np.random.choice(self.num_users_in_db)]
        else:
            self.current_user = self._get_embedding()
        return (self._get_obs(), {})

    def step(self, action):
        if False:
            print('Hello World!')
        scores = softmax([np.dot(self.current_user, doc) for doc in self.currently_suggested_docs])
        best_reward = np.max(scores)
        user_doc_overlaps = np.array([scores[a] for a in action] + [0.0])
        probabilities = softmax(user_doc_overlaps)
        which_clicked = np.random.choice(np.arange(self.slate_size + 1), p=probabilities)
        reward = 0.0
        if which_clicked < self.slate_size:
            regret = best_reward - user_doc_overlaps[which_clicked]
            reward = (1 - regret) * 100
            self.current_user_budget -= 1.0
        done = truncated = self.current_user_budget <= 0.0
        response = tuple(({'click': int(idx == which_clicked), 'engagement': reward if idx == which_clicked else 0.0} for idx in range(len(user_doc_overlaps) - 1)))
        return (self._get_obs(response=response), reward, done, truncated, {})

    def _get_obs(self, response=None):
        if False:
            i = 10
            return i + 15
        if self.num_docs_in_db is not None:
            if self.docs_db is None:
                self.docs_db = [self._get_embedding() for _ in range(self.num_docs_in_db)]
            self.currently_suggested_docs = [self.docs_db[doc_idx].astype(np.float32) for doc_idx in np.random.choice(self.num_docs_in_db, size=(self.num_docs_to_select_from,), replace=False)]
        else:
            self.currently_suggested_docs = [self._get_embedding() for _ in range(self.num_docs_to_select_from)]
        doc = {str(i): d for (i, d) in enumerate(self.currently_suggested_docs)}
        if not response:
            response = self.observation_space['response'].sample()
        return {'user': self.current_user.astype(np.float32), 'doc': doc, 'response': response}
if __name__ == '__main__':
    'Test RecommSys env with random actions for baseline performance.'
    env = ParametricRecSys(num_docs_in_db=100, num_users_in_db=1)
    (obs, info) = env.reset()
    num_episodes = 0
    episode_rewards = []
    episode_reward = 0.0
    while num_episodes < 100:
        action = env.action_space.sample()
        (obs, reward, done, truncated, _) = env.step(action)
        episode_reward += reward
        if done:
            print(f'episode reward = {episode_reward}')
            env.reset()
            num_episodes += 1
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
    print(f'Avg reward={np.mean(episode_rewards)}')