import logging
import re
from abc import ABCMeta
from collections import defaultdict
from typing import Any, DefaultDict, Dict
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.annotations import ExperimentalAPI, override
from ray.rllib.utils.numpy import softmax
from ray.rllib.utils.typing import PolicyID, ResultDict
logger = logging.getLogger(__name__)

@ExperimentalAPI
class LeagueBuilder(metaclass=ABCMeta):

    def __init__(self, algo: Algorithm, algo_config: AlgorithmConfig):
        if False:
            i = 10
            return i + 15
        'Initializes a LeagueBuilder instance.\n\n        Args:\n            algo: The Algorithm object by which this league builder is used.\n                Algorithm calls `build_league()` after each training step.\n            algo_config: The (not yet validated) config to be\n                used on the Algorithm. Child classes of `LeagueBuilder`\n                should preprocess this to add e.g. multiagent settings\n                to this config.\n        '
        self.algo = algo
        self.config = algo_config

    def build_league(self, result: ResultDict) -> None:
        if False:
            i = 10
            return i + 15
        'Method containing league-building logic. Called after train step.\n\n        Args:\n            result: The most recent result dict with all necessary stats in\n                it (e.g. episode rewards) to perform league building\n                operations.\n        '
        raise NotImplementedError

    def __getstate__(self) -> Dict[str, Any]:
        if False:
            return 10
        'Returns a state dict, mapping str keys to state variables.\n\n        Returns:\n            The current state dict of this LeagueBuilder.\n        '
        return {}

@ExperimentalAPI
class NoLeagueBuilder(LeagueBuilder):
    """A LeagueBuilder that does nothing.

    Useful for simple, non-league-building multi-agent setups.
    See e.g.
    `rllib/tuned_examples/alpha_star/multi-agent-cart-pole-alpha-star.yaml`
    """

    def build_league(self, result: ResultDict) -> None:
        if False:
            i = 10
            return i + 15
        pass

@ExperimentalAPI
class AlphaStarLeagueBuilder(LeagueBuilder):

    def __init__(self, algo: Algorithm, algo_config: AlgorithmConfig, num_random_policies: int=2, num_learning_league_exploiters: int=4, num_learning_main_exploiters: int=4, win_rate_threshold_for_new_snapshot: float=0.8, keep_new_snapshot_training_prob: float=0.0, prob_league_exploiter_match: float=0.33, prob_main_exploiter_match: float=0.33, prob_main_exploiter_playing_against_learning_main: float=0.5):
        if False:
            return 10
        "Initializes a AlphaStarLeagueBuilder instance.\n\n        The following match types are possible:\n        LE: A learning (not snapshot) league_exploiter vs any snapshot policy.\n        ME: A learning (not snapshot) main exploiter vs any main.\n        M: Main self-play (main vs main).\n\n        Args:\n            algo: The Algorithm object by which this league builder is used.\n                Algorithm calls `build_league()` after each training step to reconfigure\n                the league structure (e.g. to add/remove policies).\n            algo_config: The (not yet validated) config to be\n                used on the Algorithm. Child classes of `LeagueBuilder`\n                should preprocess this to add e.g. multiagent settings\n                to this config.\n            num_random_policies: The number of random policies to add to the\n                league. This must be an even number (including 0) as these\n                will be evenly distributed amongst league- and main- exploiters.\n            num_learning_league_exploiters: The number of initially learning\n                league-exploiters to create.\n            num_learning_main_exploiters: The number of initially learning\n                main-exploiters to create.\n            win_rate_threshold_for_new_snapshot: The win-rate to be achieved\n                for a learning policy to get snapshot'd (forked into `self` +\n                a new learning or non-learning copy of `self`).\n            keep_new_snapshot_training_prob: The probability with which a new\n                snapshot should keep training. Note that the policy from which\n                this snapshot is taken will continue to train regardless.\n            prob_league_exploiter_match: Probability of an episode to become a\n                league-exploiter vs snapshot match.\n            prob_main_exploiter_match: Probability of an episode to become a\n                main-exploiter vs main match.\n            prob_main_exploiter_playing_against_learning_main: Probability of\n                a main-exploiter vs (training!) main match.\n        "
        super().__init__(algo, algo_config)
        self.win_rate_threshold_for_new_snapshot = win_rate_threshold_for_new_snapshot
        self.keep_new_snapshot_training_prob = keep_new_snapshot_training_prob
        self.prob_league_exploiter_match = prob_league_exploiter_match
        self.prob_main_exploiter_match = prob_main_exploiter_match
        self.prob_main_exploiter_playing_against_learning_main = prob_main_exploiter_playing_against_learning_main
        self.win_rates: DefaultDict[PolicyID, float] = defaultdict(float)
        assert num_random_policies % 2 == 0, "ERROR: `num_random_policies` must be even number (we'll distribute these evenly amongst league- and main-exploiters)!"
        self.config._is_frozen = False
        assert self.config.policies is None, 'ERROR: `config.policies` should be None (not pre-defined by user)! AlphaStarLeagueBuilder will construct this itself.'
        policies = {}
        self.main_policies = 1
        self.league_exploiters = num_learning_league_exploiters + num_random_policies / 2
        self.main_exploiters = num_learning_main_exploiters + num_random_policies / 2
        policies['main_0'] = PolicySpec()
        policies_to_train = ['main_0']
        i = -1
        for i in range(num_random_policies // 2):
            policies[f'league_exploiter_{i}'] = PolicySpec(policy_class=RandomPolicy)
            policies[f'main_exploiter_{i}'] = PolicySpec(policy_class=RandomPolicy)
        for j in range(num_learning_league_exploiters):
            pid = f'league_exploiter_{j + i + 1}'
            policies[pid] = PolicySpec()
            policies_to_train.append(pid)
        for j in range(num_learning_league_exploiters):
            pid = f'main_exploiter_{j + i + 1}'
            policies[pid] = PolicySpec()
            policies_to_train.append(pid)
        self.config.policy_mapping_fn = lambda agent_id, episode, worker, **kw: 'main_0' if episode.episode_id % 2 == agent_id else 'main_exploiter_0'
        self.config.policies = policies
        self.config.policies_to_train = policies_to_train
        self.config.freeze()

    @override(LeagueBuilder)
    def build_league(self, result: ResultDict) -> None:
        if False:
            return 10
        local_worker = self.algo.workers.local_worker()
        if 'evaluation' in result:
            hist_stats = result['evaluation']['hist_stats']
        else:
            hist_stats = result['hist_stats']
        trainable_policies = local_worker.get_policies_to_train()
        non_trainable_policies = set(local_worker.policy_map.keys()) - trainable_policies
        logger.info(f'League building after iter {self.algo.iteration}:')
        for (policy_id, rew) in hist_stats.items():
            mo = re.match('^policy_(.+)_reward$', policy_id)
            if mo is None:
                continue
            policy_id = mo.group(1)
            won = 0
            for r in rew:
                if r > 0.0:
                    won += 1
            win_rate = won / len(rew)
            self.win_rates[policy_id] = win_rate
            if policy_id not in trainable_policies:
                continue
            logger.info(f'\t{policy_id} win-rate={win_rate} -> ')
            if win_rate >= self.win_rate_threshold_for_new_snapshot:
                is_main = re.match('^main(_\\d+)?$', policy_id)
                keep_training_p = self.keep_new_snapshot_training_prob
                keep_training = False if is_main else np.random.choice([True, False], p=[keep_training_p, 1.0 - keep_training_p])
                if policy_id.startswith('league_ex'):
                    new_pol_id = re.sub('_\\d+$', f'_{self.league_exploiters}', policy_id)
                    self.league_exploiters += 1
                elif policy_id.startswith('main_ex'):
                    new_pol_id = re.sub('_\\d+$', f'_{self.main_exploiters}', policy_id)
                    self.main_exploiters += 1
                else:
                    new_pol_id = re.sub('_\\d+$', f'_{self.main_policies}', policy_id)
                    self.main_policies += 1
                if keep_training:
                    trainable_policies.add(new_pol_id)
                else:
                    non_trainable_policies.add(new_pol_id)
                logger.info(f'adding new opponents to the mix ({new_pol_id}; trainable={keep_training}).')
                num_main_policies = self.main_policies
                probs_match_types = [self.prob_league_exploiter_match, self.prob_main_exploiter_match, 1.0 - self.prob_league_exploiter_match - self.prob_main_exploiter_match]
                prob_playing_learning_main = self.prob_main_exploiter_playing_against_learning_main

                def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                    if False:
                        return 10
                    type_ = np.random.choice(['LE', 'ME', 'M'], p=probs_match_types)
                    if type_ == 'LE':
                        if episode.episode_id % 2 == agent_id:
                            league_exploiter = np.random.choice([p for p in trainable_policies if p.startswith('league_ex')])
                            logger.debug(f'Episode {episode.episode_id}: AgentID {agent_id} played by {league_exploiter} (training)')
                            return league_exploiter
                        else:
                            all_opponents = list(non_trainable_policies)
                            probs = softmax([worker.global_vars['win_rates'][pid] for pid in all_opponents])
                            opponent = np.random.choice(all_opponents, p=probs)
                            logger.debug(f'Episode {episode.episode_id}: AgentID {agent_id} played by {opponent} (frozen)')
                            return opponent
                    elif type_ == 'ME':
                        if episode.episode_id % 2 == agent_id:
                            main_exploiter = np.random.choice([p for p in trainable_policies if p.startswith('main_ex')])
                            logger.debug(f'Episode {episode.episode_id}: AgentID {agent_id} played by {main_exploiter} (training)')
                            return main_exploiter
                        else:
                            if num_main_policies == 1 or np.random.random() < prob_playing_learning_main:
                                main = 'main_0'
                                training = 'training'
                            else:
                                all_opponents = [f'main_{p}' for p in list(range(1, num_main_policies))]
                                probs = softmax([worker.global_vars['win_rates'][pid] for pid in all_opponents])
                                main = np.random.choice(all_opponents, p=probs)
                                training = 'frozen'
                            logger.debug(f'Episode {episode.episode_id}: AgentID {agent_id} played by {main} ({training})')
                            return main
                    else:
                        logger.debug(f'Episode {episode.episode_id}: main_0 vs main_0')
                        return 'main_0'
                state = self.algo.get_policy(policy_id).get_state()
                self.algo.add_policy(policy_id=new_pol_id, policy_cls=type(self.algo.get_policy(policy_id)), policy_state=state, policy_mapping_fn=policy_mapping_fn, policies_to_train=trainable_policies)
            else:
                logger.info('not good enough; will keep learning ...')

    def __getstate__(self) -> Dict[str, Any]:
        if False:
            return 10
        return {'win_rates': self.win_rates, 'main_policies': self.main_policies, 'league_exploiters': self.league_exploiters, 'main_exploiters': self.main_exploiters}

    def __setstate__(self, state) -> None:
        if False:
            print('Hello World!')
        self.win_rates = state['win_rates']
        self.main_policies = state['main_policies']
        self.league_exploiters = state['league_exploiters']
        self.main_exploiters = state['main_exploiters']