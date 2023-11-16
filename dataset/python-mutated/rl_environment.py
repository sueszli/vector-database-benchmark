"""Reinforcement Learning (RL) Environment for Open Spiel.

This module wraps Open Spiel Python interface providing an RL-friendly API. It
covers both turn-based and simultaneous move games. Interactions between agents
and the underlying game occur mostly through the `reset` and `step` methods,
which return a `TimeStep` structure (see its docstrings for more info).

The following example illustrates the interaction dynamics. Consider a 2-player
Kuhn Poker (turn-based game). Agents have access to the `observations` (a dict)
field from `TimeSpec`, containing the following members:
 * `info_state`: list containing the game information state for each player. The
   size of the list always correspond to the number of players. E.g.:
   [[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]].
 * `legal_actions`: list containing legal action ID lists (one for each player).
   E.g.: [[0, 1], [0]], which corresponds to actions 0 and 1 being valid for
   player 0 (the 1st player) and action 0 being valid for player 1 (2nd player).
 * `current_player`: zero-based integer representing the player to make a move.

At each `step` call, the environment expects a singleton list with the action
(as it's a turn-based game), e.g.: [1]. This (zero-based) action must correspond
to the player specified at `current_player`. The game (which is at decision
node) will process the action and take as many steps necessary to cover chance
nodes, halting at a new decision or final node. Finally, a new `TimeStep`is
returned to the agent.

Simultaneous-move games follow analogous dynamics. The only differences is the
environment expects a list of actions, one per player. Note the `current_player`
field is "irrelevant" here, admitting a constant value defined in spiel.h, which
defaults to -2 (module level constant `SIMULTANEOUS_PLAYER_ID`).

See open_spiel/python/examples/rl_example.py for example usages.
"""
import collections
import enum
from absl import logging
import numpy as np
import pyspiel
SIMULTANEOUS_PLAYER_ID = pyspiel.PlayerId.SIMULTANEOUS

class TimeStep(collections.namedtuple('TimeStep', ['observations', 'rewards', 'discounts', 'step_type'])):
    """Returned with every call to `step` and `reset`.

  A `TimeStep` contains the data emitted by a game at each step of interaction.
  A `TimeStep` holds an `observation` (list of dicts, one per player),
  associated lists of `rewards`, `discounts` and a `step_type`.

  The first `TimeStep` in a sequence will have `StepType.FIRST`. The final
  `TimeStep` will have `StepType.LAST`. All other `TimeStep`s in a sequence will
  have `StepType.MID.

  Attributes:
    observations: a list of dicts containing observations per player.
    rewards: A list of scalars (one per player), or `None` if `step_type` is
      `StepType.FIRST`, i.e. at the start of a sequence.
    discounts: A list of discount values in the range `[0, 1]` (one per player),
      or `None` if `step_type` is `StepType.FIRST`.
    step_type: A `StepType` enum value.
  """
    __slots__ = ()

    def first(self):
        if False:
            return 10
        return self.step_type == StepType.FIRST

    def mid(self):
        if False:
            i = 10
            return i + 15
        return self.step_type == StepType.MID

    def last(self):
        if False:
            i = 10
            return i + 15
        return self.step_type == StepType.LAST

    def is_simultaneous_move(self):
        if False:
            i = 10
            return i + 15
        return self.observations['current_player'] == SIMULTANEOUS_PLAYER_ID

    def current_player(self):
        if False:
            i = 10
            return i + 15
        return self.observations['current_player']

class StepType(enum.Enum):
    """Defines the status of a `TimeStep` within a sequence."""
    FIRST = 0
    MID = 1
    LAST = 2

    def first(self):
        if False:
            for i in range(10):
                print('nop')
        return self is StepType.FIRST

    def mid(self):
        if False:
            return 10
        return self is StepType.MID

    def last(self):
        if False:
            return 10
        return self is StepType.LAST

def registered_games():
    if False:
        return 10
    return pyspiel.registered_games()

class ChanceEventSampler(object):
    """Default sampler for external chance events."""

    def __init__(self, seed=None):
        if False:
            print('Hello World!')
        self.seed(seed)

    def seed(self, seed=None):
        if False:
            for i in range(10):
                print('nop')
        self._rng = np.random.RandomState(seed)

    def __call__(self, state):
        if False:
            while True:
                i = 10
        'Sample a chance event in the given state.'
        (actions, probs) = zip(*state.chance_outcomes())
        return self._rng.choice(actions, p=probs)

class ObservationType(enum.Enum):
    """Defines what kind of observation to use."""
    OBSERVATION = 0
    INFORMATION_STATE = 1

class Environment(object):
    """Open Spiel reinforcement learning environment class."""

    def __init__(self, game, discount=1.0, chance_event_sampler=None, observation_type=None, include_full_state=False, mfg_distribution=None, mfg_population=None, enable_legality_check=False, **kwargs):
        if False:
            print('Hello World!')
        "Constructor.\n\n    Args:\n      game: [string, pyspiel.Game] Open Spiel game name or game instance.\n      discount: float, discount used in non-initial steps. Defaults to 1.0.\n      chance_event_sampler: optional object with `sample_external_events` method\n        to sample chance events.\n      observation_type: what kind of observation to use. If not specified, will\n        default to INFORMATION_STATE unless the game doesn't provide it.\n      include_full_state: whether or not to include the full serialized\n        OpenSpiel state in the observations (sometimes useful for debugging).\n      mfg_distribution: the distribution over states if the game is a mean field\n        game.\n      mfg_population: The Mean Field Game population to consider.\n      enable_legality_check: Check the legality of the move before stepping.\n      **kwargs: dict, additional settings passed to the Open Spiel game.\n    "
        self._chance_event_sampler = chance_event_sampler or ChanceEventSampler()
        self._include_full_state = include_full_state
        self._mfg_distribution = mfg_distribution
        self._mfg_population = mfg_population
        self._enable_legality_check = enable_legality_check
        if isinstance(game, str):
            if kwargs:
                game_settings = {key: val for (key, val) in kwargs.items()}
                logging.info('Using game settings: %s', game_settings)
                self._game = pyspiel.load_game(game, game_settings)
            else:
                logging.info('Using game string: %s', game)
                self._game = pyspiel.load_game(game)
        else:
            logging.info('Using game instance: %s', game.get_type().short_name)
            self._game = game
        self._num_players = self._game.num_players()
        self._state = None
        self._should_reset = True
        self._discounts = [discount] * self._num_players
        if observation_type is None:
            if self._game.get_type().provides_information_state_tensor:
                observation_type = ObservationType.INFORMATION_STATE
            else:
                observation_type = ObservationType.OBSERVATION
        if observation_type == ObservationType.OBSERVATION:
            if not self._game.get_type().provides_observation_tensor:
                raise ValueError(f'observation_tensor not supported by {game}')
        elif observation_type == ObservationType.INFORMATION_STATE:
            if not self._game.get_type().provides_information_state_tensor:
                raise ValueError(f'information_state_tensor not supported by {game}')
        self._use_observation = observation_type == ObservationType.OBSERVATION
        if self._game.get_type().dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD:
            assert mfg_distribution is not None
            assert mfg_population is not None
            assert 0 <= mfg_population < self._num_players

    def seed(self, seed=None):
        if False:
            print('Hello World!')
        self._chance_event_sampler.seed(seed)

    def get_time_step(self):
        if False:
            while True:
                i = 10
        'Returns a `TimeStep` without updating the environment.\n\n    Returns:\n      A `TimeStep` namedtuple containing:\n        observation: list of dicts containing one observations per player, each\n          corresponding to `observation_spec()`.\n        reward: list of rewards at this timestep, or None if step_type is\n          `StepType.FIRST`.\n        discount: list of discounts in the range [0, 1], or None if step_type is\n          `StepType.FIRST`.\n        step_type: A `StepType` value.\n    '
        observations = {'info_state': [], 'legal_actions': [], 'current_player': [], 'serialized_state': []}
        rewards = []
        step_type = StepType.LAST if self._state.is_terminal() else StepType.MID
        self._should_reset = step_type == StepType.LAST
        cur_rewards = self._state.rewards()
        for player_id in range(self.num_players):
            rewards.append(cur_rewards[player_id])
            observations['info_state'].append(self._state.observation_tensor(player_id) if self._use_observation else self._state.information_state_tensor(player_id))
            observations['legal_actions'].append(self._state.legal_actions(player_id))
        observations['current_player'] = self._state.current_player()
        discounts = self._discounts
        if step_type == StepType.LAST:
            discounts = [0.0 for _ in discounts]
        if self._include_full_state:
            observations['serialized_state'] = pyspiel.serialize_game_and_state(self._game, self._state)
        if hasattr(self._state, 'last_info'):
            observations['info'] = self._state.last_info
        return TimeStep(observations=observations, rewards=rewards, discounts=discounts, step_type=step_type)

    def _check_legality(self, actions):
        if False:
            i = 10
            return i + 15
        if self.is_turn_based:
            legal_actions = self._state.legal_actions()
            if actions[0] not in legal_actions:
                raise RuntimeError(f'step() called on illegal action {actions[0]}')
        else:
            for p in range(len(actions)):
                legal_actions = self._state.legal_actions(p)
                if legal_actions and actions[p] not in legal_actions:
                    raise RuntimeError(f'step() by player {p} called on illegal ' + f'action: {actions[p]}')

    def step(self, actions):
        if False:
            while True:
                i = 10
        'Updates the environment according to `actions` and returns a `TimeStep`.\n\n    If the environment returned a `TimeStep` with `StepType.LAST` at the\n    previous step, this call to `step` will start a new sequence and `actions`\n    will be ignored.\n\n    This method will also start a new sequence if called after the environment\n    has been constructed and `reset` has not been called. Again, in this case\n    `actions` will be ignored.\n\n    Args:\n      actions: a list containing one action per player, following specifications\n        defined in `action_spec()`.\n\n    Returns:\n      A `TimeStep` namedtuple containing:\n        observation: list of dicts containing one observations per player, each\n          corresponding to `observation_spec()`.\n        reward: list of rewards at this timestep, or None if step_type is\n          `StepType.FIRST`.\n        discount: list of discounts in the range [0, 1], or None if step_type is\n          `StepType.FIRST`.\n        step_type: A `StepType` value.\n    '
        assert len(actions) == self.num_actions_per_step, 'Invalid number of actions! Expected {}'.format(self.num_actions_per_step)
        if self._should_reset:
            return self.reset()
        if self._enable_legality_check:
            self._check_legality(actions)
        if self.is_turn_based:
            self._state.apply_action(actions[0])
        else:
            self._state.apply_actions(actions)
        self._sample_external_events()
        return self.get_time_step()

    def reset(self):
        if False:
            i = 10
            return i + 15
        'Starts a new sequence and returns the first `TimeStep` of this sequence.\n\n    Returns:\n      A `TimeStep` namedtuple containing:\n        observations: list of dicts containing one observations per player, each\n          corresponding to `observation_spec()`.\n        rewards: list of rewards at this timestep, or None if step_type is\n          `StepType.FIRST`.\n        discounts: list of discounts in the range [0, 1], or None if step_type\n          is `StepType.FIRST`.\n        step_type: A `StepType` value.\n    '
        self._should_reset = False
        if self._game.get_type().dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD and self._num_players > 1:
            self._state = self._game.new_initial_state_for_population(self._mfg_population)
        else:
            self._state = self._game.new_initial_state()
        self._sample_external_events()
        observations = {'info_state': [], 'legal_actions': [], 'current_player': [], 'serialized_state': []}
        for player_id in range(self.num_players):
            observations['info_state'].append(self._state.observation_tensor(player_id) if self._use_observation else self._state.information_state_tensor(player_id))
            observations['legal_actions'].append(self._state.legal_actions(player_id))
        observations['current_player'] = self._state.current_player()
        if self._include_full_state:
            observations['serialized_state'] = pyspiel.serialize_game_and_state(self._game, self._state)
        return TimeStep(observations=observations, rewards=None, discounts=None, step_type=StepType.FIRST)

    def _sample_external_events(self):
        if False:
            return 10
        'Sample chance events until we get to a decision node.'
        while self._state.is_chance_node() or self._state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
            if self._state.is_chance_node():
                outcome = self._chance_event_sampler(self._state)
                self._state.apply_action(outcome)
            if self._state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
                dist_to_register = self._state.distribution_support()
                dist = [self._mfg_distribution.value_str(str_state, default_value=0.0) for str_state in dist_to_register]
                self._state.update_distribution(dist)

    def observation_spec(self):
        if False:
            print('Hello World!')
        'Defines the observation per player provided by the environment.\n\n    Each dict member will contain its expected structure and shape. E.g.: for\n    Kuhn Poker {"info_state": (6,), "legal_actions": (2,), "current_player": (),\n                "serialized_state": ()}\n\n    Returns:\n      A specification dict describing the observation fields and shapes.\n    '
        return dict(info_state=tuple([self._game.observation_tensor_size() if self._use_observation else self._game.information_state_tensor_size()]), legal_actions=(self._game.num_distinct_actions(),), current_player=(), serialized_state=())

    def action_spec(self):
        if False:
            i = 10
            return i + 15
        'Defines per player action specifications.\n\n    Specifications include action boundaries and their data type.\n    E.g.: for Kuhn Poker {"num_actions": 2, "min": 0, "max":1, "dtype": int}\n\n    Returns:\n      A specification dict containing per player action properties.\n    '
        return dict(num_actions=self._game.num_distinct_actions(), min=0, max=self._game.num_distinct_actions() - 1, dtype=int)

    @property
    def use_observation(self):
        if False:
            for i in range(10):
                print('nop')
        "Returns whether the environment is using the game's observation.\n\n    If false, it is using the game's information state.\n    "
        return self._use_observation

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self._game.get_type().short_name

    @property
    def num_players(self):
        if False:
            for i in range(10):
                print('nop')
        return self._game.num_players()

    @property
    def num_actions_per_step(self):
        if False:
            print('Hello World!')
        return 1 if self.is_turn_based else self.num_players

    @property
    def is_turn_based(self):
        if False:
            while True:
                i = 10
        return self._game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL or self._game.get_type().dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD

    @property
    def max_game_length(self):
        if False:
            return 10
        return self._game.max_game_length()

    @property
    def is_chance_node(self):
        if False:
            for i in range(10):
                print('nop')
        return self._state.is_chance_node()

    @property
    def game(self):
        if False:
            i = 10
            return i + 15
        return self._game

    def set_state(self, new_state):
        if False:
            for i in range(10):
                print('nop')
        'Updates the game state.'
        assert new_state.get_game() == self.game, 'State must have been created by the same game.'
        self._state = new_state

    @property
    def get_state(self):
        if False:
            while True:
                i = 10
        return self._state

    @property
    def mfg_distribution(self):
        if False:
            for i in range(10):
                print('nop')
        return self._mfg_distribution

    def update_mfg_distribution(self, mfg_distribution):
        if False:
            for i in range(10):
                print('nop')
        'Updates the distribution over the states of the mean field game.'
        assert self._game.get_type().dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD
        self._mfg_distribution = mfg_distribution