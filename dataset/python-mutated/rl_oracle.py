"""An Oracle for any RL algorithm.

An Oracle for any RL algorithm following the OpenSpiel Policy API.
"""
import numpy as np
from open_spiel.python.algorithms.psro_v2 import optimization_oracle
from open_spiel.python.algorithms.psro_v2 import utils

def update_episodes_per_oracles(episodes_per_oracle, played_policies_indexes):
    if False:
        return 10
    'Updates the current episode count per policy.\n\n  Args:\n    episodes_per_oracle: List of list of number of episodes played per policy.\n      One list per player.\n    played_policies_indexes: List with structure (player_index, policy_index) of\n      played policies whose count needs updating.\n\n  Returns:\n    Updated count.\n  '
    for (player_index, policy_index) in played_policies_indexes:
        episodes_per_oracle[player_index][policy_index] += 1
    return episodes_per_oracle

def freeze_all(policies_per_player):
    if False:
        for i in range(10):
            print('nop')
    'Freezes all policies within policy_per_player.\n\n  Args:\n    policies_per_player: List of list of number of policies.\n  '
    for policies in policies_per_player:
        for pol in policies:
            pol.freeze()

def random_count_weighted_choice(count_weight):
    if False:
        for i in range(10):
            print('nop')
    "Returns a randomly sampled index i with P ~ 1 / (count_weight[i] + 1).\n\n  Allows random sampling to prioritize indexes that haven't been sampled as many\n  times as others.\n\n  Args:\n    count_weight: A list of counts to sample an index from.\n\n  Returns:\n    Randomly-sampled index.\n  "
    indexes = list(range(len(count_weight)))
    p = np.array([1 / (weight + 1) for weight in count_weight])
    p /= np.sum(p)
    chosen_index = np.random.choice(indexes, p=p)
    return chosen_index

class RLOracle(optimization_oracle.AbstractOracle):
    """Oracle handling Approximate Best Responses computation."""

    def __init__(self, env, best_response_class, best_response_kwargs, number_training_episodes=1000.0, self_play_proportion=0.0, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Init function for the RLOracle.\n\n    Args:\n      env: rl_environment instance.\n      best_response_class: class of the best response.\n      best_response_kwargs: kwargs of the best response.\n      number_training_episodes: (Minimal) number of training episodes to run\n        each best response through. May be higher for some policies.\n      self_play_proportion: Float, between 0 and 1. Defines the probability that\n        a non-currently-training player will actually play (one of) its\n        currently training strategy (Which will be trained as well).\n      **kwargs: kwargs\n    '
        self._env = env
        self._best_response_class = best_response_class
        self._best_response_kwargs = best_response_kwargs
        self._self_play_proportion = self_play_proportion
        self._number_training_episodes = number_training_episodes
        super(RLOracle, self).__init__(**kwargs)

    def sample_episode(self, unused_time_step, agents, is_evaluation=False):
        if False:
            i = 10
            return i + 15
        time_step = self._env.reset()
        cumulative_rewards = 0.0
        while not time_step.last():
            if time_step.is_simultaneous_move():
                action_list = []
                for agent in agents:
                    output = agent.step(time_step, is_evaluation=is_evaluation)
                    action_list.append(output.action)
                time_step = self._env.step(action_list)
                cumulative_rewards += np.array(time_step.rewards)
            else:
                player_id = time_step.observations['current_player']
                agent_output = agents[player_id].step(time_step, is_evaluation=is_evaluation)
                action_list = [agent_output.action]
                time_step = self._env.step(action_list)
                cumulative_rewards += np.array(time_step.rewards)
        if not is_evaluation:
            for agent in agents:
                agent.step(time_step)
        return cumulative_rewards

    def _has_terminated(self, episodes_per_oracle):
        if False:
            i = 10
            return i + 15
        return np.all(episodes_per_oracle.reshape(-1) > self._number_training_episodes)

    def sample_policies_for_episode(self, new_policies, training_parameters, episodes_per_oracle, strategy_sampler):
        if False:
            while True:
                i = 10
        "Randomly samples a set of policies to run during the next episode.\n\n    Note : sampling is biased to select players & strategies that haven't\n    trained as much as the others.\n\n    Args:\n      new_policies: The currently training policies, list of list, one per\n        player.\n      training_parameters: List of list of training parameters dictionaries, one\n        list per player, one dictionary per training policy.\n      episodes_per_oracle: List of list of integers, computing the number of\n        episodes trained on by each policy. Used to weight the strategy\n        sampling.\n      strategy_sampler: Sampling function that samples a joint strategy given\n        probabilities.\n\n    Returns:\n      Sampled list of policies (One policy per player), index of currently\n      training policies in the list.\n    "
        num_players = len(training_parameters)
        episodes_per_player = [sum(episodes) for episodes in episodes_per_oracle]
        chosen_player = random_count_weighted_choice(episodes_per_player)
        agent_chosen_ind = np.random.randint(0, len(training_parameters[chosen_player]))
        agent_chosen_dict = training_parameters[chosen_player][agent_chosen_ind]
        new_policy = new_policies[chosen_player][agent_chosen_ind]
        total_policies = agent_chosen_dict['total_policies']
        probabilities_of_playing_policies = agent_chosen_dict['probabilities_of_playing_policies']
        episode_policies = strategy_sampler(total_policies, probabilities_of_playing_policies)
        live_agents_player_index = [(chosen_player, agent_chosen_ind)]
        for player in range(num_players):
            if player == chosen_player:
                episode_policies[player] = new_policy
                assert not new_policy.is_frozen()
            elif np.random.binomial(1, self._self_play_proportion):
                agent_index = random_count_weighted_choice(episodes_per_oracle[player])
                self_play_agent = new_policies[player][agent_index]
                episode_policies[player] = self_play_agent
                live_agents_player_index.append((player, agent_index))
            else:
                assert episode_policies[player].is_frozen()
        return (episode_policies, live_agents_player_index)

    def _rollout(self, game, agents, **oracle_specific_execution_kwargs):
        if False:
            while True:
                i = 10
        self.sample_episode(None, agents, is_evaluation=False)

    def generate_new_policies(self, training_parameters):
        if False:
            while True:
                i = 10
        'Generates new policies to be trained into best responses.\n\n    Args:\n      training_parameters: list of list of training parameter dictionaries, one\n        list per player.\n\n    Returns:\n      List of list of the new policies, following the same structure as\n      training_parameters.\n    '
        new_policies = []
        for player in range(len(training_parameters)):
            player_parameters = training_parameters[player]
            new_pols = []
            for param in player_parameters:
                current_pol = param['policy']
                if isinstance(current_pol, self._best_response_class):
                    new_pol = current_pol.copy_with_noise(self._kwargs.get('sigma', 0.0))
                else:
                    new_pol = self._best_response_class(self._env, player, **self._best_response_kwargs)
                    new_pol.unfreeze()
                new_pols.append(new_pol)
            new_policies.append(new_pols)
        return new_policies

    def __call__(self, game, training_parameters, strategy_sampler=utils.sample_strategy, **oracle_specific_execution_kwargs):
        if False:
            while True:
                i = 10
        'Call method for oracle, returns best responses against a set of policies.\n\n    Args:\n      game: The game on which the optimization process takes place.\n      training_parameters: A list of list of dictionaries (One list per player),\n        each dictionary containing the following fields :\n        - policy : the policy from which to start training.\n        - total_policies: A list of all policy.Policy strategies used for\n          training, including the one for the current player.\n        - current_player: Integer representing the current player.\n        - probabilities_of_playing_policies: A list of arrays representing, per\n          player, the probabilities of playing each policy in total_policies for\n          the same player.\n      strategy_sampler: Callable that samples strategies from total_policies\n        using probabilities_of_playing_policies. It only samples one joint\n        set of policies for all players. Implemented to be able to take into\n        account joint probabilities of action (For Alpharank)\n      **oracle_specific_execution_kwargs: Other set of arguments, for\n        compatibility purposes. Can for example represent whether to Rectify\n        Training or not.\n\n    Returns:\n      A list of list, one for each member of training_parameters, of (epsilon)\n      best responses.\n    '
        episodes_per_oracle = [[0 for _ in range(len(player_params))] for player_params in training_parameters]
        episodes_per_oracle = np.array(episodes_per_oracle)
        new_policies = self.generate_new_policies(training_parameters)
        while not self._has_terminated(episodes_per_oracle):
            (agents, indexes) = self.sample_policies_for_episode(new_policies, training_parameters, episodes_per_oracle, strategy_sampler)
            self._rollout(game, agents, **oracle_specific_execution_kwargs)
            episodes_per_oracle = update_episodes_per_oracles(episodes_per_oracle, indexes)
        freeze_all(new_policies)
        return new_policies