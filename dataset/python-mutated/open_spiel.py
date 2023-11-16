from typing import Optional
import numpy as np
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.utils import try_import_pyspiel
pyspiel = try_import_pyspiel(error=True)

class OpenSpielEnv(MultiAgentEnv):

    def __init__(self, env):
        if False:
            print('Hello World!')
        super().__init__()
        self.env = env
        self._skip_env_checking = True
        self.num_agents = self.env.num_players()
        self.type = self.env.get_type()
        self.state = None
        self.observation_space = Box(float('-inf'), float('inf'), (self.env.observation_tensor_size(),))
        self.action_space = Discrete(self.env.num_distinct_actions())

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        if False:
            for i in range(10):
                print('nop')
        self.state = self.env.new_initial_state()
        return (self._get_obs(), {})

    def step(self, action):
        if False:
            print('Hello World!')
        self._solve_chance_nodes()
        penalties = {}
        if str(self.type.dynamics) == 'Dynamics.SEQUENTIAL':
            curr_player = self.state.current_player()
            assert curr_player in action
            try:
                self.state.apply_action(action[curr_player])
            except pyspiel.SpielError:
                self.state.apply_action(np.random.choice(self.state.legal_actions()))
                penalties[curr_player] = -0.1
            rewards = {ag: r for (ag, r) in enumerate(self.state.returns())}
        else:
            assert self.state.current_player() == -2
            self.state.apply_actions([action[ag] for ag in range(self.num_agents)])
        obs = self._get_obs()
        rewards = {ag: r for (ag, r) in enumerate(self.state.returns())}
        for (ag, penalty) in penalties.items():
            rewards[ag] += penalty
        is_terminated = self.state.is_terminal()
        terminateds = dict({ag: is_terminated for ag in range(self.num_agents)}, **{'__all__': is_terminated})
        truncateds = dict({ag: False for ag in range(self.num_agents)}, **{'__all__': False})
        return (obs, rewards, terminateds, truncateds, {})

    def render(self, mode=None) -> None:
        if False:
            print('Hello World!')
        if mode == 'human':
            print(self.state)

    def _get_obs(self):
        if False:
            print('Hello World!')
        self._solve_chance_nodes()
        if self.state.is_terminal():
            return {}
        if str(self.type.dynamics) == 'Dynamics.SEQUENTIAL':
            curr_player = self.state.current_player()
            return {curr_player: np.reshape(self.state.observation_tensor(), [-1])}
        else:
            assert self.state.current_player() == -2
            return {ag: np.reshape(self.state.observation_tensor(ag), [-1]) for ag in range(self.num_agents)}

    def _solve_chance_nodes(self):
        if False:
            i = 10
            return i + 15
        while self.state.is_chance_node():
            assert self.state.current_player() == -1
            (actions, probs) = zip(*self.state.chance_outcomes())
            action = np.random.choice(actions, p=probs)
            self.state.apply_action(action)