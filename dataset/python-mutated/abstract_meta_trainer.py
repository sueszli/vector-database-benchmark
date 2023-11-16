"""Abstract class for meta trainers (Generalized PSRO, RNR, ...)

Meta-algorithm with modular behaviour, allowing implementation of PSRO, RNR, and
other variations.
"""
import numpy as np
from open_spiel.python.algorithms.psro_v2 import meta_strategies
from open_spiel.python.algorithms.psro_v2 import strategy_selectors
from open_spiel.python.algorithms.psro_v2 import utils
_DEFAULT_STRATEGY_SELECTION_METHOD = 'probabilistic'
_DEFAULT_META_STRATEGY_METHOD = 'prd'

def _process_string_or_callable(string_or_callable, dictionary):
    if False:
        for i in range(10):
            print('nop')
    'Process a callable or a string representing a callable.\n\n  Args:\n    string_or_callable: Either a string or a callable\n    dictionary: Dictionary of shape {string_reference: callable}\n\n  Returns:\n    string_or_callable if string_or_callable is a callable ; otherwise,\n    dictionary[string_or_callable]\n\n  Raises:\n    NotImplementedError: If string_or_callable is of the wrong type, or has an\n      unexpected value (Not present in dictionary).\n  '
    if callable(string_or_callable):
        return string_or_callable
    try:
        return dictionary[string_or_callable]
    except KeyError as e:
        raise NotImplementedError('Input type / value not supported. Accepted types: string, callable. Acceptable string values : {}. Input provided : {}'.format(list(dictionary.keys()), string_or_callable)) from e

def sample_episode(state, policies):
    if False:
        i = 10
        return i + 15
    'Samples an episode using policies, starting from state.\n\n  Args:\n    state: Pyspiel state representing the current state.\n    policies: List of policy representing the policy executed by each player.\n\n  Returns:\n    The result of the call to returns() of the final state in the episode.\n        Meant to be a win/loss integer.\n  '
    if state.is_terminal():
        return np.array(state.returns(), dtype=np.float32)
    if state.is_simultaneous_node():
        actions = [None] * state.num_players()
        for player in range(state.num_players()):
            state_policy = policies[player](state, player)
            (outcomes, probs) = zip(*state_policy.items())
            actions[player] = utils.random_choice(outcomes, probs)
        state.apply_actions(actions)
        return sample_episode(state, policies)
    if state.is_chance_node():
        (outcomes, probs) = zip(*state.chance_outcomes())
    else:
        player = state.current_player()
        state_policy = policies[player](state)
        (outcomes, probs) = zip(*state_policy.items())
    state.apply_action(utils.random_choice(outcomes, probs))
    return sample_episode(state, policies)

class AbstractMetaTrainer(object):
    """Abstract class implementing meta trainers.

  If a trainer is something that computes a best response to given environment &
  agents, a meta trainer will compute which best responses to compute (Against
  what, how, etc)
  This class can support PBT, Hyperparameter Evolution, etc.
  """

    def __init__(self, game, oracle, initial_policies=None, meta_strategy_method=_DEFAULT_META_STRATEGY_METHOD, training_strategy_selector=_DEFAULT_STRATEGY_SELECTION_METHOD, symmetric_game=False, number_policies_selected=1, **kwargs):
        if False:
            i = 10
            return i + 15
        'Abstract Initialization for meta trainers.\n\n    Args:\n      game: A pyspiel game object.\n      oracle: An oracle object, from an implementation of the AbstractOracle\n        class.\n      initial_policies: A list of initial policies, to set up a default for\n        training. Resorts to tabular policies if not set.\n      meta_strategy_method: String, or callable taking a MetaTrainer object and\n        returning a list of meta strategies (One list entry per player).\n        String value can be:\n              - "uniform": Uniform distribution on policies.\n              - "nash": Taking nash distribution. Only works for 2 player, 0-sum\n                games.\n              - "prd": Projected Replicator Dynamics, as described in Lanctot et\n                Al.\n      training_strategy_selector: A callable or a string. If a callable, takes\n        as arguments: - An instance of `PSROSolver`, - a\n          `number_policies_selected` integer. and returning a list of\n          `num_players` lists of selected policies to train from.\n        When a string, supported values are:\n              - "top_k_probabilites": selects the first\n                \'number_policies_selected\' policies with highest selection\n                probabilities.\n              - "probabilistic": randomly selects \'number_policies_selected\'\n                with probabilities determined by the meta strategies.\n              - "exhaustive": selects every policy of every player.\n              - "rectified": only selects strategies that have nonzero chance of\n                being selected.\n              - "uniform": randomly selects \'number_policies_selected\' policies\n                with uniform probabilities.\n      symmetric_game: Whether to consider the current game as symmetric (True)\n        game or not (False).\n      number_policies_selected: Maximum number of new policies to train for each\n        player at each PSRO iteration.\n      **kwargs: kwargs for meta strategy computation and training strategy\n        selection\n    '
        self._iterations = 0
        self._game = game
        self._oracle = oracle
        self._num_players = self._game.num_players()
        self.symmetric_game = symmetric_game
        self._game_num_players = self._num_players
        self._num_players = 1 if symmetric_game else self._num_players
        self._number_policies_selected = number_policies_selected
        meta_strategy_method = _process_string_or_callable(meta_strategy_method, meta_strategies.META_STRATEGY_METHODS)
        print('Using {} as strategy method.'.format(meta_strategy_method))
        self._training_strategy_selector = _process_string_or_callable(training_strategy_selector, strategy_selectors.TRAINING_STRATEGY_SELECTORS)
        print('Using {} as training strategy selector.'.format(self._training_strategy_selector))
        self._meta_strategy_method = meta_strategy_method
        self._kwargs = kwargs
        self._initialize_policy(initial_policies)
        self._initialize_game_state()
        self.update_meta_strategies()

    def _initialize_policy(self, initial_policies):
        if False:
            print('Hello World!')
        return NotImplementedError('initialize_policy not implemented. Initial policies passed as arguments : {}'.format(initial_policies))

    def _initialize_game_state(self):
        if False:
            print('Hello World!')
        return NotImplementedError('initialize_game_state not implemented.')

    def iteration(self, seed=None):
        if False:
            i = 10
            return i + 15
        'Main trainer loop.\n\n    Args:\n      seed: Seed for random BR noise generation.\n    '
        self._iterations += 1
        self.update_agents()
        self.update_empirical_gamestate(seed=seed)
        self.update_meta_strategies()

    def update_meta_strategies(self):
        if False:
            while True:
                i = 10
        self._meta_strategy_probabilities = self._meta_strategy_method(self)
        if self.symmetric_game:
            self._meta_strategy_probabilities = [self._meta_strategy_probabilities[0]]

    def update_agents(self):
        if False:
            print('Hello World!')
        return NotImplementedError('update_agents not implemented.')

    def update_empirical_gamestate(self, seed=None):
        if False:
            print('Hello World!')
        return NotImplementedError('update_empirical_gamestate not implemented. Seed passed as argument : {}'.format(seed))

    def sample_episodes(self, policies, num_episodes):
        if False:
            while True:
                i = 10
        'Samples episodes and averages their returns.\n\n    Args:\n      policies: A list of policies representing the policies executed by each\n        player.\n      num_episodes: Number of episodes to execute to estimate average return of\n        policies.\n\n    Returns:\n      Average episode return over num episodes.\n    '
        totals = np.zeros(self._num_players)
        for _ in range(num_episodes):
            totals += sample_episode(self._game.new_initial_state(), policies).reshape(-1)
        return totals / num_episodes

    def get_meta_strategies(self):
        if False:
            return 10
        'Returns the Nash Equilibrium distribution on meta game matrix.'
        meta_strategy_probabilities = self._meta_strategy_probabilities
        if self.symmetric_game:
            meta_strategy_probabilities = self._game_num_players * meta_strategy_probabilities
        return [np.copy(a) for a in meta_strategy_probabilities]

    def get_meta_game(self):
        if False:
            print('Hello World!')
        'Returns the meta game matrix.'
        meta_games = self._meta_games
        return [np.copy(a) for a in meta_games]

    def get_policies(self):
        if False:
            for i in range(10):
                print('nop')
        "Returns the players' policies."
        policies = self._policies
        if self.symmetric_game:
            policies = self._game_num_players * policies
        return policies

    def get_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        return self._kwargs