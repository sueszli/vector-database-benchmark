from SelfPlayExp import SelfPlayExp
import os
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from callbacks import *
from shared import evaluate_policy_simple
import bach_utils.os as utos
from bach_utils.shared import *
from bach_utils.heatmapvis import *
from bach_utils.json_parser import ExperimentParser
import random

class PPOMod(PPO):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(PPOMod, self).__init__(*args, **kwargs)

class SelfPlayTesting(SelfPlayExp):

    def __init__(self, seed_value=None, render_sleep_time=0.001):
        if False:
            print('Hello World!')
        super(SelfPlayTesting, self).__init__()
        self.seed_value = seed_value
        self.load_prefix = 'history_'
        self.deterministic = True
        self.warn = True
        self.render = None
        self.crosstest_flag = None
        self.render_sleep_time = render_sleep_time

    def _import_original_configs(self):
        if False:
            return 10
        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs['name']
            testing_config = self.testing_configs[agent_name]
            agent_config_file_path = os.path.join(testing_config['path'], 'experiment_config.json')
            if os.path.isfile(agent_config_file_path):
                self.clilog.info(f'Parse from json file in {agent_config_file_path}')
                (_experiment_configs, _agents_configs, _evaluation_configs, _testing_configs, _merged_config) = ExperimentParser.load(agent_config_file_path)
                agent_original_config = _agents_configs[k]
                self.agents_configs[k] = agent_original_config

    def _init_testing(self, experiment_filename, logdir, wandb):
        if False:
            print('Hello World!')
        super(SelfPlayTesting, self)._init_exp(experiment_filename, logdir, wandb)
        self._import_original_configs()
        self.render = self.testing_configs.get('render', True)
        self.crosstest_flag = self.testing_configs.get('crosstest', False)
        self.clilog.info(f'----- Load testing conditions')
        self._load_testing_conditions(experiment_filename)

    def _init_argparse(self):
        if False:
            i = 10
            return i + 15
        super(SelfPlayTesting, self)._init_argparse(description='Self-play experiment testing script', help='The experiemnt configuration file path and name which the experiment should be loaded')

    def _generate_log_dir(self):
        if False:
            print('Hello World!')
        super(SelfPlayTesting, self)._generate_log_dir(dir_postfix='test')

    def _load_testing_conditions(self, path):
        if False:
            i = 10
            return i + 15
        self.testing_conditions = {}
        self.testing_modes = {}
        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs['name']
            testing_config = self.testing_configs[agent_name]
            agent_testing_path = os.path.join(path, agent_name) if testing_config['path'] is None else testing_config['path']
            agent_testing_path = os.path.join(agent_testing_path, self.testing_configs[agent_name]['dirname'])
            mode = testing_config['mode']
            self.testing_conditions[agent_name] = {'path': agent_testing_path}
            self.testing_modes[agent_name] = mode
            num_rounds = self.experiment_configs['num_rounds']
            if mode == 'limit':
                self.testing_conditions[agent_name]['limits'] = [0, testing_config['gens'], testing_config['freq']]
            elif mode == 'limit_s':
                self.testing_conditions[agent_name]['limits'] = [testing_config['gens'], num_rounds - 1, testing_config['freq']]
            elif mode == 'limit_e':
                self.testing_conditions[agent_name]['limits'] = [0, testing_config['gens'], testing_config['freq']]
            elif mode == 'gen':
                self.testing_conditions[agent_name]['limits'] = [testing_config['gens'], testing_config['gens'], testing_config['freq']]
            elif mode == 'all':
                self.testing_conditions[agent_name]['limits'] = [0, num_rounds - 1, testing_config['freq']]
            elif mode == 'random':
                self.testing_conditions[agent_name]['limits'] = [None, None, testing_config['freq']]
            elif mode == 'round':
                self.testing_conditions[agent_name]['limits'] = [0, num_rounds - 1, testing_config['freq']]
            self.clilog.debug(self.testing_conditions[agent_name]['limits'])

    def _get_opponent_algorithm_class(self, agent_configs):
        if False:
            while True:
                i = 10
        algorithm_class = None
        opponent_algorithm_class_cfg = agent_configs.get('opponent_rl_algorithm', agent_configs['rl_algorithm'])
        if opponent_algorithm_class_cfg == 'PPO':
            algorithm_class = PPOMod
        elif opponent_algorithm_class_cfg == 'SAC':
            algorithm_class = SAC
        return algorithm_class

    def _init_envs(self):
        if False:
            return 10
        self.envs = {}
        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs['name']
            algorithm_class = self._get_opponent_algorithm_class(agent_configs)
            env = super(SelfPlayTesting, self).create_env(key=k, name='Testing', opponent_archive=None, algorithm_class=algorithm_class, gui=True)
            self.envs[agent_name] = env

    def _init_archives(self):
        if False:
            return 10
        raise NotImplementedError('_init_archives() not implemented')

    def _init_models(self):
        if False:
            i = 10
            return i + 15
        self.models = {}
        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs['name']
            algorithm_class = None
            if agent_configs['rl_algorithm'] == 'PPO':
                algorithm_class = PPOMod
            elif agent_configs['rl_algorithm'] == 'SAC':
                algorithm_class = SAC
            self.models[agent_name] = algorithm_class

    def render_callback(self, ret):
        if False:
            return 10
        return ret

    def _run_one_evaluation(self, agent_conifgs_key, sampled_agent, sampled_opponents, n_eval_episodes, render_extra_info, env=None, render=None, agent_model=None, seed_value=None, return_episode_rewards=False):
        if False:
            i = 10
            return i + 15
        self.clilog.info('----------------------------------------')
        self.clilog.info(render_extra_info)
        self.make_deterministic(cuda_check=False)
        if env is None and agent_model is None:
            agent_configs = self.agents_configs[agent_conifgs_key]
            opponent_algorithm_class = self._get_opponent_algorithm_class(agent_configs)
            (env, seed_value) = super(SelfPlayTesting, self).create_env(key=agent_conifgs_key, name='Testing', opponent_archive=None, algorithm_class=opponent_algorithm_class, seed_value=seed_value, ret_seed=True, gui=True)
            algorithm_class = None
            if agent_configs['rl_algorithm'] == 'PPO':
                algorithm_class = PPOMod
            elif agent_configs['rl_algorithm'] == 'SAC':
                algorithm_class = SAC
            self.clilog.debug(f'loading agent model: {sampled_agent}, {algorithm_class}, {env}')
            agent_model = algorithm_class.load(sampled_agent, env)
        (mean_reward, std_reward, win_rate, std_win_rate, render_ret) = evaluate_policy_simple(agent_model, env, n_eval_episodes=n_eval_episodes, render=self.render if render is None else render, deterministic=self.deterministic, return_episode_rewards=return_episode_rewards, warn=self.warn, callback=None, sampled_opponents=sampled_opponents, render_extra_info=render_extra_info, render_callback=self.render_callback, sleep_time=self.render_sleep_time)
        (mean_reward_, std_reward_, win_rate_, std_win_rate_) = (mean_reward, std_reward, win_rate, std_win_rate)
        if return_episode_rewards:
            mean_reward_ = np.mean(mean_reward)
            std_reward_ = np.std(mean_reward)
            win_rate_ = np.mean(win_rate)
        self.clilog.info(f'{render_extra_info} -> win rate: {100 * win_rate_:.2f}% +/- {std_win_rate_:.2f}\trewards: {mean_reward_:.2f} +/- {std_reward_:.2f}')
        env.close()
        return (mean_reward, std_reward, win_rate, std_win_rate, render_ret)

    def _test_round_by_round(self, key, n_eval_episodes):
        if False:
            while True:
                i = 10
        agent_configs = self.agents_configs[key]
        agent_name = agent_configs['name']
        opponent_name = agent_configs['opponent_name']
        for round_num in range(0, self.experiment_configs['num_rounds'], self.testing_conditions[agent_name]['limits'][2]):
            startswith_keyword = f'{self.load_prefix}{round_num}_'
            agent_latest = utos.get_latest(self.testing_conditions[agent_name]['path'], startswith=startswith_keyword)
            if len(agent_latest) == 0:
                continue
            sampled_agent = os.path.join(self.testing_conditions[agent_name]['path'], agent_latest[0])
            opponent_latest = utos.get_latest(self.testing_conditions[opponent_name]['path'], startswith=startswith_keyword)
            if len(opponent_latest) == 0:
                continue
            sampled_opponent = os.path.join(self.testing_conditions[opponent_name]['path'], opponent_latest[0])
            sampled_opponents = [sampled_opponent]
            self._run_one_evaluation(key, sampled_agent, sampled_opponents, n_eval_episodes, f'{round_num} vs {round_num}')

    def _test_different_rounds(self, key, n_eval_episodes):
        if False:
            return 10
        agent_configs = self.agents_configs[key]
        agent_name = agent_configs['name']
        opponent_name = agent_configs['opponent_name']
        for i in range(self.testing_conditions[agent_name]['limits'][0], self.testing_conditions[agent_name]['limits'][1] + 1, self.testing_conditions[agent_name]['limits'][2]):
            agent_startswith_keyword = f'{self.load_prefix}{i}_'
            agent_latest = utos.get_latest(self.testing_conditions[agent_name]['path'], startswith=agent_startswith_keyword)
            if len(agent_latest) == 0:
                continue
            sampled_agent = os.path.join(self.testing_conditions[agent_name]['path'], agent_latest[0])
            for j in range(self.testing_conditions[opponent_name]['limits'][0], self.testing_conditions[opponent_name]['limits'][1] + 1, self.testing_conditions[opponent_name]['limits'][2]):
                opponent_startswith_keyword = f'{self.load_prefix}{j}_'
                opponent_latest = utos.get_latest(self.testing_conditions[opponent_name]['path'], startswith=opponent_startswith_keyword)
                if len(opponent_latest) == 0:
                    continue
                sampled_opponent = os.path.join(self.testing_conditions[opponent_name]['path'], opponent_latest[0])
                sampled_opponents = [sampled_opponent]
                self._run_one_evaluation(key, sampled_agent, sampled_opponents, n_eval_episodes, f'{i} vs {j}')

    def get_latest_agent_path(self, idx, path, population_idx):
        if False:
            while True:
                i = 10
        agent_startswith_keyword = f'{self.load_prefix}{idx}_'
        agent_latest = utos.get_latest(path, startswith=agent_startswith_keyword, population_idx=population_idx)
        ret = True
        if len(agent_latest) == 0:
            ret = False
        latest_agent = os.path.join(path, agent_latest[0])
        return (ret, latest_agent)

    def _compute_performance(self, agent, opponent, key, n_eval_episodes=1, n_seeds=1, negative_score_flag=False, render=False, render_extra_info=None):
        if False:
            return 10

        def normalize_performance(min_val, max_val, performance, negative_score_flag):
            if False:
                i = 10
                return i + 15
            if negative_score_flag:
                performance = min(0, performance)
                return (max_val - abs(performance)) / max_val
            else:
                performance = max(0, performance)
                return performance / max_val
        (episodes_reward, episodes_length, win_rates, std_win_rate, render_ret) = self._run_one_evaluation(key, agent, [opponent], n_eval_episodes, render=render, render_extra_info=f'{agent} vs {opponent}' if render_extra_info is None else render_extra_info, return_episode_rewards=True)
        length = np.mean(episodes_length)
        limits = [0, 1000]
        normalized_length = normalize_performance(*limits, length, negative_score_flag)
        self.clilog.debug(f'Nomralized: {normalized_length}, {length}')
        return normalized_length
        (mean_reward, std_reward, win_rate, std_win_rate, render_ret) = self._run_one_evaluation(key, agent, [opponent], n_eval_episodes, render=render, render_extra_info=f'{agent} vs {opponent}' if render_extra_info is None else render_extra_info)
        reward = np.mean(mean_reward)
        limits = self.testing_configs.get('crosstest_rewards_limits')
        normalized_reward = normalize_performance(*limits, reward, negative_score_flag)
        print(f'Nomralized: {normalized_reward}, {reward}')
        return normalized_reward

    def _get_best_agent(self, num_rounds, search_radius, paths, key, num_population, min_gamma_val=0.05, n_eval_episodes=1, n_seeds=1, render=False, negative_score_flag=False):
        if False:
            for i in range(10):
                print('nop')
        (agent_num_rounds, opponent_num_rounds) = num_rounds[:]
        (agent_path, opponent_path) = paths[:]
        self.clilog.info('##################################################################')
        self.clilog.info(f'## Getting the best model for {key}')
        best_rewards = []
        freq = self.testing_configs.get('crosstest_freq')
        opponents_rounds_idx = [i for i in range(0, opponent_num_rounds, freq)]
        if not opponent_num_rounds - 1 in opponents_rounds_idx:
            opponents_rounds_idx.append(opponent_num_rounds - 1)
        gamma = min_gamma_val ** (1 / len(opponents_rounds_idx))
        for agent_population_idx in range(num_population):
            for agent_idx in range(agent_num_rounds - search_radius - 1, agent_num_rounds):
                (ret, agent) = self.get_latest_agent_path(agent_idx, agent_path, agent_population_idx)
                if not ret:
                    continue
                rewards = []
                for opponent_population_idx_ in range(1):
                    opponent_population_indices = [random.randint(0, num_population - 1) for _ in range(len(opponents_rounds_idx))]
                    for (i, opponent_idx) in enumerate(opponents_rounds_idx):
                        opponent_population_idx = opponent_population_indices[i]
                        (ret, sampled_opponent) = self.get_latest_agent_path(opponent_idx, opponent_path, opponent_population_idx)
                        if not ret:
                            continue
                        mean_reward = self._compute_performance(agent, sampled_opponent, key, n_eval_episodes, n_seeds, negative_score_flag, render, render_extra_info=f'{agent_idx}({agent_population_idx}) vs {opponent_idx} ({opponent_population_idx})')
                        weight = 1
                        weighted_reward = weight * mean_reward
                        self.clilog.debug(f'Weight: {weight}\tPerformance: {mean_reward}\tWeighted Performance: {weighted_reward}')
                        rewards.append(weighted_reward)
                mean_reward = np.mean(np.array(rewards))
                best_rewards.append([agent_population_idx, agent_idx, mean_reward])
        best_rewards = np.array(best_rewards)
        self.clilog.debug(best_rewards)
        best_agent_idx = np.argmax(best_rewards[:, 2])
        agent_idx = int(best_rewards[best_agent_idx, 1])
        agent_population_idx = int(best_rewards[best_agent_idx, 0])
        self.clilog.info(f'Best agent: idx {agent_idx}, population {agent_population_idx}')
        startswith_keyword = f'{self.load_prefix}{agent_idx}_'
        agent_latest = utos.get_latest(agent_path, startswith=startswith_keyword, population_idx=agent_population_idx)
        best_agent = os.path.join(agent_path, agent_latest[0])
        return best_agent

    def crosstest(self, n_eval_episodes, n_seeds):
        if False:
            while True:
                i = 10
        self.clilog.info(f'---------------- Running Crosstest ----------------')
        num_rounds = self.testing_configs.get('crosstest_num_rounds')
        (num_rounds1, num_rounds2) = (num_rounds[0], num_rounds[1])
        search_radius = self.testing_configs.get('crosstest_search_radius')
        self.clilog.info(f'Num. rounds: {num_rounds1}, {num_rounds2}')
        approaches_path = self.testing_configs.get('crosstest_approaches_path')
        (approach1_path, approach2_path) = (approaches_path[0], approaches_path[1])
        self.clilog.info(f'Paths:\n{approach1_path}\n{approach2_path}')
        names = [self.agents_configs[k]['name'] for k in self.agents_configs.keys()]
        (agent_name, opponent_name) = (names[0], names[1])
        self.clilog.info(f'names: {agent_name}, {opponent_name}')
        agent1_path = os.path.join(approach1_path, agent_name)
        opponent1_path = os.path.join(approach1_path, opponent_name)
        agent2_path = os.path.join(approach2_path, agent_name)
        opponent2_path = os.path.join(approach2_path, opponent_name)
        self.clilog.info(f'Agent1 path: {agent1_path}')
        self.clilog.info(f'Opponenet1 path: {opponent1_path}')
        self.clilog.info(f'Agent2 path: {agent2_path}')
        self.clilog.info(f'Opponenet2 path: {opponent2_path}')
        (num_population1, num_population2) = self.testing_configs.get('crosstest_populations')
        self.clilog.info(f'Num. populations: {num_population1}, {num_population2}')
        n_eval_episodes_best_agent = self.testing_configs.get('n_eval_episodes_best_agent', 1)
        best_agent1 = self._get_best_agent([num_rounds1, num_rounds1], search_radius, [agent1_path, opponent1_path], agent_name, num_population1, n_eval_episodes=n_eval_episodes_best_agent, negative_score_flag=True, n_seeds=n_seeds)
        self.clilog.info(f'Best agent1: {best_agent1}')
        best_opponent1 = self._get_best_agent([num_rounds1, num_rounds1], search_radius, [opponent1_path, agent1_path], opponent_name, num_population1, n_eval_episodes=n_eval_episodes_best_agent, negative_score_flag=False, n_seeds=n_seeds)
        self.clilog.info(f'Best opponent1: {best_opponent1}')
        best_agent2 = self._get_best_agent([num_rounds2, num_rounds2], search_radius, [agent2_path, opponent2_path], agent_name, num_population2, n_eval_episodes=n_eval_episodes_best_agent, negative_score_flag=True, n_seeds=n_seeds)
        self.clilog.info(f'Best agent2: {best_agent2}')
        best_opponent2 = self._get_best_agent([num_rounds2, num_rounds2], search_radius, [opponent2_path, agent2_path], opponent_name, num_population2, n_eval_episodes=n_eval_episodes_best_agent, negative_score_flag=False, n_seeds=n_seeds)
        self.clilog.info(f'Best opponent2: {best_opponent2}')
        self.clilog.info('###############################################################')
        self.clilog.info(f'# Best agent1: {best_agent1}')
        self.clilog.info(f'# Best opponent1: {best_opponent1}')
        self.clilog.info(f'# Best agent2: {best_agent2}')
        self.clilog.info(f'# Best opponent2: {best_opponent2}')
        self.clilog.info('###############################################################')
        render = self.testing_configs.get('render')
        self.clilog.info(f'################# Agent1 vs Opponent2 #################')
        perf_agent1_opponent2 = self._compute_performance(best_agent1, best_opponent2, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=render)
        self.clilog.info(f'################# Agent1 vs Opponent1 #################')
        perf_agent1_opponent1 = self._compute_performance(best_agent1, best_opponent1, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=render)
        self.clilog.info(f'################# Agent2 vs Opponent2 #################')
        perf_agent2_opponent2 = self._compute_performance(best_agent2, best_opponent2, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=render)
        self.clilog.info(f'################# Agent2 vs Opponent1 #################')
        perf_agent2_opponent1 = self._compute_performance(best_agent2, best_opponent1, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=render)
        self.clilog.info(f'################# Opponent1 vs Agent2 #################')
        perf_opponent1_agent2 = self._compute_performance(best_opponent1, best_agent2, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=render)
        self.clilog.info(f'################# Opponent1 vs Agent1 #################')
        perf_opponent1_agent1 = self._compute_performance(best_opponent1, best_agent1, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=render)
        self.clilog.info(f'################# Opponent1 vs Agent2 #################')
        perf_opponent2_agent2 = self._compute_performance(best_opponent2, best_agent2, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=render)
        self.clilog.info(f'################# Opponent2 vs Agent1 #################')
        perf_opponent2_agent1 = self._compute_performance(best_opponent2, best_agent1, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=render)
        perf_agent1 = perf_agent1_opponent2 - perf_agent1_opponent1
        perf_agent2 = perf_agent2_opponent2 - perf_agent2_opponent1
        perf_opponent1 = perf_opponent1_agent2 - perf_opponent1_agent1
        perf_opponent2 = perf_opponent2_agent2 - perf_opponent2_agent1
        gain1 = perf_agent1 + perf_opponent1
        gain2 = perf_agent2 + perf_opponent2
        self.clilog.info('-----------------------------------------------------------------')
        self.clilog.info(f'perf_agent1: {perf_agent1}\tperf_opponent1: {perf_opponent1}\tgain1: {gain1}')
        self.clilog.info(f'perf_agent2: {perf_agent2}\tperf_opponent2: {perf_opponent2}\tgain2: {gain2}')
        self.clilog.info(f'perf_agent: {perf_agent1 + perf_agent2}\tperf_opponent: {perf_opponent1 + perf_opponent2}\tgain(sum): {gain1 + gain2}')
        gain = [gain1, gain2, gain1 + gain2]
        perf_agent = [perf_agent1, perf_agent2, perf_agent1 + perf_agent2]
        perf_opponent = [perf_opponent1, perf_opponent2, perf_opponent1 + perf_opponent2]
        for i in range(3):
            self.clilog.info(f' ----- Part {i + 1} ----- ')
            eps = 0.001
            if perf_agent[i] > 0:
                self.clilog.info(f'Configuration 1 is better {1} to generate preys (path: {approach1_path})')
            elif -eps <= perf_agent[i] <= eps:
                self.clilog.info(f'Configuration 1 & 2 are close to each other to generate preys')
            else:
                self.clilog.info(f'Configuration 2 is better {2} to generate preys (path: {approach2_path})')
            if perf_opponent[i] > 0:
                self.clilog.info(f'Configuration 1 is better {1} to generate predators (path: {approach1_path})')
            elif -eps <= perf_opponent[i] <= eps:
                self.clilog.info(f'Configuration 1 & 2 are close to each other to generate predators')
            else:
                self.clilog.info(f'Configuration 2 is better {2} to generate predators (path: {approach2_path})')
            if gain[i] > 0:
                self.clilog.info(f'Configuration 1 is better {1} (path: {approach1_path})')
            elif -eps <= gain[i] <= eps:
                self.clilog.info(f'Configuration 1 & 2 are close to each other')
            else:
                self.clilog.info(f'Configuration 2 is better {2} (path: {approach2_path})')

    def test(self, experiment_filename=None, logdir=False, wandb=False, n_eval_episodes=1):
        if False:
            while True:
                i = 10
        self._init_testing(experiment_filename=experiment_filename, logdir=logdir, wandb=wandb)
        self.render_sleep_time = self.render_sleep_time if self.args.rendersleep <= 0 else self.args.rendersleep
        n_eval_episodes_configs = self.testing_configs.get('repetition', None)
        n_eval_episodes = n_eval_episodes_configs if n_eval_episodes_configs is not None else n_eval_episodes
        if self.crosstest_flag:
            n_seeds = self.testing_configs.get('n_seeds', 1)
            self.crosstest(n_eval_episodes, n_seeds)
        else:
            already_evaluated_agents = []
            self.clilog.debug(self.testing_modes)
            keys = self.agents_configs.keys()
            keys = ['pred', 'prey']
            for k in keys:
                agent_configs = self.agents_configs[k]
                agent_name = agent_configs['name']
                agent_opponent_joint = sorted([agent_name, agent_configs['opponent_name']])
                if 'round' in self.testing_modes.values():
                    if agent_opponent_joint in already_evaluated_agents:
                        continue
                    self._test_round_by_round(k, n_eval_episodes)
                else:
                    self._test_different_rounds(k, n_eval_episodes)
                already_evaluated_agents.append(agent_opponent_joint)