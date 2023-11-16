"""An RL agent wrapper for the MCTS bot."""
import numpy as np
from open_spiel.python import rl_agent
import pyspiel

class MCTSAgent(rl_agent.AbstractAgent):
    """MCTS agent class.

  Important note: this agent requires the environment to provide the full state
  in its TimeStep objects. Hence, the environment must be created with the
  use_full_state flag set to True, and the state must be serializable.
  """

    def __init__(self, player_id, num_actions, mcts_bot, name='mcts_agent'):
        if False:
            return 10
        assert num_actions > 0
        self._player_id = player_id
        self._mcts_bot = mcts_bot
        self._num_actions = num_actions

    def step(self, time_step, is_evaluation=False):
        if False:
            while True:
                i = 10
        if time_step.last():
            return
        assert 'serialized_state' in time_step.observations
        (_, state) = pyspiel.deserialize_game_and_state(time_step.observations['serialized_state'])
        probs = np.zeros(self._num_actions)
        action = self._mcts_bot.step(state)
        probs[action] = 1.0
        return rl_agent.StepOutput(action=action, probs=probs)