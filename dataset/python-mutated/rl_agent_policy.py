"""Joint policy denoted by the RL agents of a game."""
from typing import Dict
from open_spiel.python import policy
from open_spiel.python import rl_agent
from open_spiel.python import rl_environment

class JointRLAgentPolicy(policy.Policy):
    """Joint policy denoted by the RL agents of a game.

  Given a list of RL agents of players for a game, this class can be used derive
  the corresponding (joint) policy. In particular, the distribution over
  possible actions will be those that are returned by the step() method of
  the RL agents given the state.
  """

    def __init__(self, game, agents: Dict[int, rl_agent.AbstractAgent], use_observation: bool):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the joint RL agent policy.\n\n    Args:\n      game: The game.\n      agents: Dictionary of agents keyed by the player IDs.\n      use_observation: If true then observation tensor will be used as the\n        `info_state` in the step() calls; otherwise, information state tensor\n        will be used. See `use_observation` property of\n        rl_environment.Environment.\n    '
        player_ids = list(sorted(agents.keys()))
        super().__init__(game, player_ids)
        self._agents = agents
        self._obs = {'info_state': [None] * game.num_players(), 'legal_actions': [None] * game.num_players()}
        self._use_observation = use_observation

    def action_probabilities(self, state, player_id=None):
        if False:
            return 10
        if state.is_simultaneous_node():
            assert player_id is not None, 'Player ID should be specified.'
        elif player_id is None:
            player_id = state.current_player()
        else:
            assert player_id == state.current_player()
        player_id = int(player_id)
        legal_actions = state.legal_actions(player_id)
        self._obs['current_player'] = player_id
        self._obs['info_state'][player_id] = state.observation_tensor(player_id) if self._use_observation else state.information_state_tensor(player_id)
        self._obs['legal_actions'][player_id] = legal_actions
        info_state = rl_environment.TimeStep(observations=self._obs, rewards=None, discounts=None, step_type=None)
        p = self._agents[player_id].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict

class RLAgentPolicy(JointRLAgentPolicy):
    """A policy for a specific agent trained in an RL environment."""

    def __init__(self, game, agent: rl_agent.AbstractAgent, player_id: int, use_observation: bool):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the RL agent policy.\n\n    Args:\n      game: The game.\n      agent: RL agent.\n      player_id: ID of the player.\n      use_observation: See JointRLAgentPolicy above.\n    '
        self._player_id = player_id
        super().__init__(game, {player_id: agent}, use_observation)

    def action_probabilities(self, state, player_id=None):
        if False:
            return 10
        return super().action_probabilities(state, self._player_id if player_id is None else player_id)