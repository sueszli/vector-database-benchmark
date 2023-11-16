"""A bot that chooses uniformly at random from legal actions."""
import pyspiel

class UniformRandomBot(pyspiel.Bot):
    """Chooses uniformly at random from the available legal actions."""

    def __init__(self, player_id, rng):
        if False:
            i = 10
            return i + 15
        'Initializes a uniform-random bot.\n\n    Args:\n      player_id: The integer id of the player for this bot, e.g. `0` if acting\n        as the first player.\n      rng: A random number generator supporting a `choice` method, e.g.\n        `np.random`\n    '
        pyspiel.Bot.__init__(self)
        self._player_id = player_id
        self._rng = rng

    def restart_at(self, state):
        if False:
            i = 10
            return i + 15
        pass

    def player_id(self):
        if False:
            for i in range(10):
                print('nop')
        return self._player_id

    def provides_policy(self):
        if False:
            i = 10
            return i + 15
        return True

    def step_with_policy(self, state):
        if False:
            for i in range(10):
                print('nop')
        'Returns the stochastic policy and selected action in the given state.\n\n    Args:\n      state: The current state of the game.\n\n    Returns:\n      A `(policy, action)` pair, where policy is a `list` of\n      `(action, probability)` pairs for each legal action, with\n      `probability = 1/num_actions`\n      The `action` is selected uniformly at random from the legal actions,\n      or `pyspiel.INVALID_ACTION` if there are no legal actions available.\n    '
        legal_actions = state.legal_actions(self._player_id)
        if not legal_actions:
            return ([], pyspiel.INVALID_ACTION)
        p = 1 / len(legal_actions)
        policy = [(action, p) for action in legal_actions]
        action = self._rng.choice(legal_actions)
        return (policy, action)

    def step(self, state):
        if False:
            return 10
        return self.step_with_policy(state)[1]