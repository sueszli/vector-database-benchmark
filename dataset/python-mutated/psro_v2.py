"""Modular implementations of the PSRO meta algorithm.

Allows the use of Restricted Nash Response, Nash Response, Uniform Response,
and other modular matchmaking selection components users can add.

This version works for N player, general sum games.

One iteration of the algorithm consists of:

1) Computing the selection probability vector (or meta-strategy) for current
strategies of each player, given their payoff.
2) [optional] Generating a mask over joint policies that restricts which policy
to train against, ie. rectify the set of policies trained against. (This
operation is designated by "rectify" in the code)
3) From every strategy used, generating a new best response strategy against the
meta-strategy-weighted, potentially rectified, mixture of strategies using an
oracle.
4) Updating meta game matrix with new game results.

"""
import itertools
import numpy as np
from open_spiel.python import policy
from open_spiel.python.algorithms.psro_v2 import abstract_meta_trainer
from open_spiel.python.algorithms.psro_v2 import strategy_selectors
from open_spiel.python.algorithms.psro_v2 import utils
TRAIN_TARGET_SELECTORS = {'': None, 'rectified': strategy_selectors.rectified_selector}

class PSROSolver(abstract_meta_trainer.AbstractMetaTrainer):
    """A general implementation PSRO.

  PSRO is the algorithm described in (Lanctot et Al., 2017,
  https://arxiv.org/pdf/1711.00832.pdf ).

  Subsequent work regarding PSRO's matchmaking and training has been performed
  by David Balduzzi, who introduced Restricted Nash Response (RNR), Nash
  Response (NR) and Uniform Response (UR).
  RNR is Algorithm 4 in (Balduzzi, 2019, "Open-ended Learning in Symmetric
  Zero-sum Games"). NR, Nash response, is algorithm 3.
  Balduzzi et Al., 2019, https://arxiv.org/pdf/1901.08106.pdf

  This implementation allows one to modularly choose different meta strategy
  computation methods, or other user-written ones.
  """

    def __init__(self, game, oracle, sims_per_entry, initial_policies=None, rectifier='', training_strategy_selector=None, meta_strategy_method='alpharank', sample_from_marginals=False, number_policies_selected=1, n_noisy_copies=0, alpha_noise=0.0, beta_noise=0.0, **kwargs):
        if False:
            i = 10
            return i + 15
        'Initialize the PSRO solver.\n\n    Arguments:\n      game: The open_spiel game object.\n      oracle: Callable that takes as input: - game - policy - policies played -\n        array representing the probability of playing policy i - other kwargs\n        and returns a new best response.\n      sims_per_entry: Number of simulations to run to estimate each element of\n        the game outcome matrix.\n      initial_policies: A list of initial policies for each player, from which\n        the optimization process will start.\n      rectifier: A string indicating the rectifying method. Can be :\n              - "" or None: Train against potentially all strategies.\n              - "rectified": Train only against strategies beaten by current\n                strategy.\n      training_strategy_selector: Callable taking (PSROSolver,\n        \'number_policies_selected\') and returning a list of list of selected\n        strategies to train from - this usually means copying weights and\n        rectifying with respect to the selected strategy\'s performance (One list\n        entry per player), or string selecting pre-implemented methods.\n        String value can be:\n              - "top_k_probabilites": selects the first\n              \'number_policies_selected\' policies with highest selection\n              probabilities.\n              - "probabilistic": randomly selects \'number_policies_selected\'\n                with probabilities determined by the meta strategies.\n              - "exhaustive": selects every policy of every player.\n              - "rectified": only selects strategies that have nonzero chance of\n                being selected.\n              - "uniform": randomly selects \'number_policies_selected\'\n                policies with uniform probabilities.\n      meta_strategy_method: String or callable taking a GenPSROSolver object and\n        returning two lists ; one list of meta strategies (One list entry per\n        player), and one list of joint strategies.\n        String value can be:\n              - alpharank: AlphaRank distribution on policies.\n              - "uniform": Uniform distribution on policies.\n              - "nash": Taking nash distribution. Only works for 2 player, 0-sum\n                games.\n              - "prd": Projected Replicator Dynamics, as described in Lanctot et\n                Al.\n      sample_from_marginals: A boolean, specifying whether to sample from\n        marginal (True) or joint (False) meta-strategy distributions.\n      number_policies_selected: Number of policies to return for each player.\n\n      n_noisy_copies: Number of noisy copies of each agent after training. 0 to\n        ignore this.\n      alpha_noise: lower bound on alpha noise value (Mixture amplitude.)\n      beta_noise: lower bound on beta noise value (Softmax temperature.)\n      **kwargs: kwargs for meta strategy computation and training strategy\n        selection.\n    '
        self._sims_per_entry = sims_per_entry
        print('Using {} sims per entry.'.format(sims_per_entry))
        self._rectifier = TRAIN_TARGET_SELECTORS.get(rectifier, None)
        self._rectify_training = self._rectifier
        print('Rectifier : {}'.format(rectifier))
        self._meta_strategy_probabilities = np.array([])
        self._non_marginalized_probabilities = np.array([])
        print('Perturbating oracle outputs : {}'.format(n_noisy_copies > 0))
        self._n_noisy_copies = n_noisy_copies
        self._alpha_noise = alpha_noise
        self._beta_noise = beta_noise
        self._policies = []
        self._new_policies = []
        if not meta_strategy_method or meta_strategy_method == 'alpharank':
            meta_strategy_method = utils.alpharank_strategy
        print('Sampling from marginals : {}'.format(sample_from_marginals))
        self.sample_from_marginals = sample_from_marginals
        super(PSROSolver, self).__init__(game, oracle, initial_policies, meta_strategy_method, training_strategy_selector, number_policies_selected=number_policies_selected, **kwargs)

    def _initialize_policy(self, initial_policies):
        if False:
            i = 10
            return i + 15
        if self.symmetric_game:
            self._policies = [[]]
            self._new_policies = [[initial_policies[0]] if initial_policies else [policy.UniformRandomPolicy(self._game)]]
        else:
            self._policies = [[] for _ in range(self._num_players)]
            self._new_policies = [[initial_policies[k]] if initial_policies else [policy.UniformRandomPolicy(self._game)] for k in range(self._num_players)]

    def _initialize_game_state(self):
        if False:
            while True:
                i = 10
        effective_payoff_size = self._game_num_players
        self._meta_games = [np.array(utils.empty_list_generator(effective_payoff_size)) for _ in range(effective_payoff_size)]
        self.update_empirical_gamestate(seed=None)

    def get_joint_policy_ids(self):
        if False:
            i = 10
            return i + 15
        'Returns a list of integers enumerating all joint meta strategies.'
        return utils.get_strategy_profile_ids(self._meta_games)

    def get_joint_policies_from_id_list(self, selected_policy_ids):
        if False:
            print('Hello World!')
        'Returns a list of joint policies from a list of integer IDs.\n\n    Args:\n      selected_policy_ids: A list of integer IDs corresponding to the\n        meta-strategies, with duplicate entries allowed.\n\n    Returns:\n      selected_joint_policies: A list, with each element being a joint policy\n        instance (i.e., a list of policies, one per player).\n    '
        policies = self.get_policies()
        selected_joint_policies = utils.get_joint_policies_from_id_list(self._meta_games, policies, selected_policy_ids)
        return selected_joint_policies

    def update_meta_strategies(self):
        if False:
            return 10
        'Recomputes the current meta strategy of each player.\n\n    Given new payoff tables, we call self._meta_strategy_method to update the\n    meta-probabilities.\n    '
        if self.symmetric_game:
            self._policies = self._policies * self._game_num_players
        (self._meta_strategy_probabilities, self._non_marginalized_probabilities) = self._meta_strategy_method(solver=self, return_joint=True)
        if self.symmetric_game:
            self._policies = [self._policies[0]]
            self._meta_strategy_probabilities = [self._meta_strategy_probabilities[0]]

    def get_policies_and_strategies(self):
        if False:
            while True:
                i = 10
        'Returns current policy sampler, policies and meta-strategies of the game.\n\n    If strategies are rectified, we automatically switch to returning joint\n    strategies.\n\n    Returns:\n      sample_strategy: A strategy sampling function\n      total_policies: A list of list of policies, one list per player.\n      probabilities_of_playing_policies: the meta strategies, either joint or\n        marginalized.\n    '
        sample_strategy = utils.sample_strategy_marginal
        probabilities_of_playing_policies = self.get_meta_strategies()
        if self._rectify_training or not self.sample_from_marginals:
            sample_strategy = utils.sample_strategy_joint
            probabilities_of_playing_policies = self._non_marginalized_probabilities
        total_policies = self.get_policies()
        return (sample_strategy, total_policies, probabilities_of_playing_policies)

    def _restrict_target_training(self, current_player, ind, total_policies, probabilities_of_playing_policies, restrict_target_training_bool, epsilon=1e-12):
        if False:
            while True:
                i = 10
        'Rectifies training.\n\n    Args:\n      current_player: the current player.\n      ind: Current strategy index of the player.\n      total_policies: all policies available to all players.\n      probabilities_of_playing_policies: meta strategies.\n      restrict_target_training_bool: Boolean specifying whether to restrict\n        training. If False, standard meta strategies are returned. Otherwise,\n        restricted joint strategies are returned.\n      epsilon: threshold below which we consider 0 sum of probabilities.\n\n    Returns:\n      Probabilities of playing each joint strategy (If rectifying) / probability\n      of each player playing each strategy (Otherwise - marginal probabilities)\n    '
        true_shape = tuple([len(a) for a in total_policies])
        if not restrict_target_training_bool:
            return probabilities_of_playing_policies
        else:
            kept_probas = self._rectifier(self, current_player, ind)
            probability = probabilities_of_playing_policies.reshape(true_shape)
            probability = probability * kept_probas
            prob_sum = np.sum(probability)
            if prob_sum <= epsilon:
                probability = probabilities_of_playing_policies
            else:
                probability /= prob_sum
            return probability

    def update_agents(self):
        if False:
            while True:
                i = 10
        'Updates policies for each player at the same time by calling the oracle.\n\n    The resulting policies are appended to self._new_policies.\n    '
        (used_policies, used_indexes) = self._training_strategy_selector(self, self._number_policies_selected)
        (sample_strategy, total_policies, probabilities_of_playing_policies) = self.get_policies_and_strategies()
        training_parameters = [[] for _ in range(self._num_players)]
        for current_player in range(self._num_players):
            if self.sample_from_marginals:
                currently_used_policies = used_policies[current_player]
                current_indexes = used_indexes[current_player]
            else:
                currently_used_policies = [joint_policy[current_player] for joint_policy in used_policies]
                current_indexes = used_indexes[current_player]
            for i in range(len(currently_used_policies)):
                pol = currently_used_policies[i]
                ind = current_indexes[i]
                new_probabilities = self._restrict_target_training(current_player, ind, total_policies, probabilities_of_playing_policies, self._rectify_training)
                new_parameter = {'policy': pol, 'total_policies': total_policies, 'current_player': current_player, 'probabilities_of_playing_policies': new_probabilities}
                training_parameters[current_player].append(new_parameter)
        if self.symmetric_game:
            self._policies = self._game_num_players * self._policies
            self._num_players = self._game_num_players
            training_parameters = [training_parameters[0]]
        self._new_policies = self._oracle(self._game, training_parameters, strategy_sampler=sample_strategy, using_joint_strategies=self._rectify_training or not self.sample_from_marginals)
        if self.symmetric_game:
            self._policies = [self._policies[0]]
            self._num_players = 1

    def update_empirical_gamestate(self, seed=None):
        if False:
            return 10
        'Given new agents in _new_policies, update meta_games through simulations.\n\n    Args:\n      seed: Seed for environment generation.\n\n    Returns:\n      Meta game payoff matrix.\n    '
        if seed is not None:
            np.random.seed(seed=seed)
        assert self._oracle is not None
        if self.symmetric_game:
            self._policies = self._game_num_players * self._policies
            self._new_policies = self._game_num_players * self._new_policies
            self._num_players = self._game_num_players
        updated_policies = [self._policies[k] + self._new_policies[k] for k in range(self._num_players)]
        total_number_policies = [len(updated_policies[k]) for k in range(self._num_players)]
        number_older_policies = [len(self._policies[k]) for k in range(self._num_players)]
        number_new_policies = [len(self._new_policies[k]) for k in range(self._num_players)]
        meta_games = [np.full(tuple(total_number_policies), np.nan) for k in range(self._num_players)]
        older_policies_slice = tuple([slice(len(self._policies[k])) for k in range(self._num_players)])
        for k in range(self._num_players):
            meta_games[k][older_policies_slice] = self._meta_games[k]
        for current_player in range(self._num_players):
            range_iterators = [range(total_number_policies[k]) for k in range(current_player)] + [range(number_new_policies[current_player])] + [range(total_number_policies[k]) for k in range(current_player + 1, self._num_players)]
            for current_index in itertools.product(*range_iterators):
                used_index = list(current_index)
                used_index[current_player] += number_older_policies[current_player]
                if np.isnan(meta_games[current_player][tuple(used_index)]):
                    estimated_policies = [updated_policies[k][current_index[k]] for k in range(current_player)] + [self._new_policies[current_player][current_index[current_player]]] + [updated_policies[k][current_index[k]] for k in range(current_player + 1, self._num_players)]
                    if self.symmetric_game:
                        utility_estimates = self.sample_episodes(estimated_policies, self._sims_per_entry)
                        player_permutations = list(itertools.permutations(list(range(self._num_players))))
                        for permutation in player_permutations:
                            used_tuple = tuple([used_index[i] for i in permutation])
                            for player in range(self._num_players):
                                if np.isnan(meta_games[player][used_tuple]):
                                    meta_games[player][used_tuple] = 0.0
                                meta_games[player][used_tuple] += utility_estimates[permutation[player]] / len(player_permutations)
                    else:
                        utility_estimates = self.sample_episodes(estimated_policies, self._sims_per_entry)
                        for k in range(self._num_players):
                            meta_games[k][tuple(used_index)] = utility_estimates[k]
        if self.symmetric_game:
            self._policies = [self._policies[0]]
            self._new_policies = [self._new_policies[0]]
            updated_policies = [updated_policies[0]]
            self._num_players = 1
        self._meta_games = meta_games
        self._policies = updated_policies
        return meta_games

    def get_meta_game(self):
        if False:
            print('Hello World!')
        'Returns the meta game matrix.'
        return self._meta_games

    @property
    def meta_games(self):
        if False:
            while True:
                i = 10
        return self._meta_games

    def get_policies(self):
        if False:
            return 10
        "Returns a list, each element being a list of each player's policies."
        policies = self._policies
        if self.symmetric_game:
            policies = self._game_num_players * self._policies
        return policies

    def get_and_update_non_marginalized_meta_strategies(self, update=True):
        if False:
            i = 10
            return i + 15
        'Returns the Nash Equilibrium distribution on meta game matrix.'
        if update:
            self.update_meta_strategies()
        return self._non_marginalized_probabilities

    def get_strategy_computation_and_selection_kwargs(self):
        if False:
            print('Hello World!')
        return self._strategy_computation_and_selection_kwargs