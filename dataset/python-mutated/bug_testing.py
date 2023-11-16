import os
from stable_baselines3 import PPO
from callbacks import *
from shared import normalize_reward, evaluate_policy_simple
import bach_utils.os as utos
from bach_utils.shared import *
from SelfPlayExp import SelfPlayExp
from bach_utils.heatmapvis import *

class PPOMod(PPO):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(PPOMod, self).__init__(*args, **kwargs)

    @staticmethod
    def load(model_path, env):
        if False:
            for i in range(10):
                print('nop')
        custom_objects = {'lr_schedule': lambda x: 0.003, 'clip_range': lambda x: 0.02}
        return PPO.load(model_path, env, custom_objects=custom_objects)

class SelfPlayTesting(SelfPlayExp):

    def __init__(self, seed_value=None, render_sleep_time=0.01):
        if False:
            print('Hello World!')
        super(SelfPlayTesting, self).__init__()
        self.seed_value = seed_value
        self.load_prefix = 'history_'
        self.deterministic = False
        self.warn = True
        self.render = None
        self.crosstest_flag = None
        self.render_sleep_time = render_sleep_time

    def _init_testing(self, experiment_filename, logdir, wandb):
        if False:
            return 10
        super(SelfPlayTesting, self)._init_exp(experiment_filename, logdir, wandb)
        self.render = self.testing_configs.get('render', True)
        self.crosstest_flag = self.testing_configs.get('crosstest', False)
        print(f'----- Load testing conditions')
        self._load_testing_conditions(experiment_filename)

    def _init_argparse(self):
        if False:
            i = 10
            return i + 15
        super(SelfPlayTesting, self)._init_argparse(description='Self-play experiment testing script', help='The experiemnt configuration file path and name which the experiment should be loaded')

    def _generate_log_dir(self):
        if False:
            for i in range(10):
                print('nop')
        super(SelfPlayTesting, self)._generate_log_dir(dir_postfix='test')

    def _load_testing_conditions(self, path):
        if False:
            return 10
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
            print(self.testing_conditions[agent_name]['limits'])

    def _init_envs(self):
        if False:
            i = 10
            return i + 15
        self.envs = {}
        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs['name']
            env = super(SelfPlayTesting, self).create_env(key=k, name='Testing', opponent_archive=None, algorithm_class=PPOMod)
            self.envs[agent_name] = env

    def _init_archives(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('_init_archives() not implemented')

    def _init_models(self):
        if False:
            return 10
        self.models = {}
        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs['name']
            self.models[agent_name] = PPOMod

    def render_callback(self, ret):
        if False:
            for i in range(10):
                print('nop')
        return ret

    def _run_one_evaluation(self, agent_conifgs_key, sampled_agent, sampled_opponents, n_eval_episodes, render_extra_info, env=None, render=None, agent_model=None, seed_value=None, return_episode_rewards=False):
        if False:
            return 10
        print('----------------------------------------')
        print(render_extra_info)
        self.make_deterministic(cuda_check=False)
        if env is None and agent_model is None:
            env = super(SelfPlayTesting, self).create_env(key=agent_conifgs_key, name='Testing', opponent_archive=None, algorithm_class=PPOMod, seed_value=seed_value)
            agent_model = PPOMod.load(sampled_agent, env)
        (mean_reward, std_reward, win_rate, std_win_rate, render_ret) = evaluate_policy_simple(agent_model, env, n_eval_episodes=n_eval_episodes, render=self.render if render is None else render, deterministic=self.deterministic, return_episode_rewards=return_episode_rewards, warn=self.warn, callback=None, sampled_opponents=sampled_opponents, render_extra_info=render_extra_info, render_callback=self.render_callback, sleep_time=self.render_sleep_time)
        print(f'{render_extra_info} -> win rate: {100 * win_rate:.2f}% +/- {std_win_rate:.2f}\trewards: {mean_reward:.2f} +/- {std_reward:.2f}')
        env.close()
        return (mean_reward, std_reward, win_rate, std_win_rate, render_ret)

    def _test_round_by_round(self, key, n_eval_episodes):
        if False:
            for i in range(10):
                print('nop')
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
            print('Hello World!')
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

    def _compute_single_round_gain_score(self, round_nums, n_eval_episodes, seed=3):
        if False:
            return 10
        gain_evaluation_models = self.testing_configs['gain_evaluation_models']
        agents_list = {k: [] for k in self.agents_configs.keys()}
        for (model_idx, model_path) in enumerate(gain_evaluation_models):
            round_num = round_nums[model_idx]
            agent_startswith_keyword = f'{self.load_prefix}{round_num}_'
            for agent_type in self.agents_configs.keys():
                path = os.path.join(model_path, agent_type)
                agent_latest = utos.get_latest(path, startswith=agent_startswith_keyword)
                if len(agent_latest) == 0:
                    continue
                sampled_agent = os.path.join(path, agent_latest[0])
                agents_list[agent_type].append((sampled_agent, model_idx, round_num))
        gain_list = []
        allowed_pairs = [(1, 0, 0), (-1, 0, 1), (-1, 0, 0), (+1, 1, 0)]
        for (factor, agent_idx, opponent_idx) in allowed_pairs:
            agent_type = 'pred'
            opponent_type = 'prey'
            (sampled_agent, agent_idx_ret, round_num1) = agents_list[agent_type][agent_idx]
            (sampled_opponents, opponent_idx_ret, round_num2) = agents_list[opponent_type][opponent_idx]
            sampled_opponents = [sampled_opponents]
            assert agent_idx == agent_idx_ret
            assert opponent_idx == opponent_idx_ret
            print('###########')
            print(f'Pair: {(agent_idx, opponent_idx)}')
            (mean_reward, std_reward, win_rate, std_win_rate, render_ret) = self._run_one_evaluation(agent_type, sampled_agent, sampled_opponents, n_eval_episodes, f'{agent_type}({agent_idx}):{round_num1} vs {opponent_type}({opponent_idx}):{round_num2}', seed_value=seed)
            score = normalize_reward(mean_reward, mn=-1010, mx=0)
            print(f'Score (Normalized reward): {score}')
            gain = factor * score
            print(f'Gain: {gain}')
            gain_list.append(gain)
        return sum(gain_list)

    def _compute_gain_score(self, n_eval_episodes, n_seeds):
        if False:
            print('Hello World!')
        print('###############################################')
        print(' ------------- Compute Gain Score -------------')
        num_rounds = self.experiment_configs['num_rounds']
        round_axis = [i for i in range(0, num_rounds, self.testing_configs['gain_score_freq'])]
        if round_axis[-1] != num_rounds - 1:
            round_axis.append(num_rounds - 1)
        gain_matrix = np.zeros([len(round_axis) for _ in self.agents_configs.keys()])
        for (ei, i) in enumerate(round_axis):
            for (ej, j) in enumerate(round_axis):
                print('--------------------------------------------')
                print(f'Compute Gain score round: {i} vs {j}')
                gain_scores = []
                for seed_idx in range(n_seeds):
                    print(f'Seed iteration: {seed_idx}')
                    gain_score = self._compute_single_round_gain_score([i, j], n_eval_episodes=n_eval_episodes, seed='random')
                    gain_scores.append(gain_score)
                gain_matrix[ei, ej] = np.mean(gain_scores)
        print('####################################################')
        print(f'Gain score {np.mean(gain_matrix):.4f} +/- {np.std(gain_matrix):.4f}')
        print('####################################################')
        HeatMapVisualizer.visPlotly(gain_matrix, xrange=round_axis, yrange=round_axis)

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
        (mean_reward, std_reward, win_rate, std_win_rate, render_ret) = self._run_one_evaluation(key, agent, [opponent], n_eval_episodes, render=render, render_extra_info=f'{agent} vs {opponent}' if render_extra_info is None else render_extra_info)
        reward = np.mean(mean_reward)
        limits = self.testing_configs.get('crosstest_rewards_limits')
        normalized_reward = normalize_performance(*limits, reward, negative_score_flag)
        print(f'Nomralized: {normalized_reward}, {reward}')
        return normalized_reward

    def _get_best_agent(self, num_rounds, search_radius, paths, key, num_population, min_gamma_val=0.05, n_eval_episodes=1, n_seeds=1, render=False, negative_score_flag=False):
        if False:
            while True:
                i = 10
        (agent_num_rounds, opponent_num_rounds) = num_rounds[:]
        (agent_path, opponent_path) = paths[:]
        print(f'## Getting the best model for {key}')
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
                for opponent_population_idx in range(num_population):
                    print(f'POP: {agent_population_idx}, {opponent_population_idx}')
                    for (i, opponent_idx) in enumerate(opponents_rounds_idx):
                        (ret, sampled_opponent) = self.get_latest_agent_path(opponent_idx, opponent_path, opponent_population_idx)
                        if not ret:
                            continue
                        mean_reward = self._compute_performance(agent, sampled_opponent, key, n_eval_episodes, n_seeds, negative_score_flag, render, render_extra_info=f'{agent_idx}({agent_population_idx}) vs {opponent_idx} ({opponent_population_idx})')
                        weight = gamma ** (len(opponents_rounds_idx) - i)
                        weighted_reward = weight * mean_reward
                        print(f'Weight: {weight}\tPerformance: {mean_reward}\tWeighted Performance: {weighted_reward}')
                        rewards.append(weighted_reward)
                mean_reward = np.mean(np.array(rewards))
                best_rewards.append([agent_population_idx, agent_idx, mean_reward])
        best_rewards = np.array(best_rewards)
        print(best_rewards)
        best_agent_idx = np.argmax(best_rewards[:, 2])
        agent_idx = int(best_rewards[best_agent_idx, 1])
        agent_population_idx = int(best_rewards[best_agent_idx, 0])
        print(f'Best agent: idx {agent_idx}, population {agent_population_idx}')
        startswith_keyword = f'{self.load_prefix}{agent_idx}_'
        agent_latest = utos.get_latest(agent_path, startswith=startswith_keyword, population_idx=agent_population_idx)
        best_agent = os.path.join(agent_path, agent_latest[0])
        return best_agent

    def crosstest(self, n_eval_episodes, n_seeds):
        if False:
            for i in range(10):
                print('nop')
        print(f'---------------- Running Crosstest ----------------')
        num_rounds = self.testing_configs.get('crosstest_num_rounds')
        (num_rounds1, num_rounds2) = (num_rounds[0], num_rounds[1])
        search_radius = self.testing_configs.get('crosstest_search_radius')
        print(f'Num. rounds: {num_rounds1}, {num_rounds2}')
        approaches_path = self.testing_configs.get('crosstest_approaches_path')
        (approach1_path, approach2_path) = (approaches_path[0], approaches_path[1])
        print(f'Paths:\n{approach1_path}\n{approach2_path}')
        names = [self.agents_configs[k]['name'] for k in self.agents_configs.keys()]
        (agent_name, opponent_name) = (names[0], names[1])
        print(f'names: {agent_name}, {opponent_name}')
        agent1_path = os.path.join(approach1_path, agent_name)
        opponent1_path = os.path.join(approach1_path, opponent_name)
        agent2_path = os.path.join(approach2_path, agent_name)
        opponent2_path = os.path.join(approach2_path, opponent_name)
        print(f'Agent1 path: {agent1_path}')
        print(f'Opponenet1 path: {opponent1_path}')
        print(f'Agent2 path: {agent2_path}')
        print(f'Opponenet2 path: {opponent2_path}')
        (num_population1, num_population2) = self.testing_configs.get('crosstest_populations')
        print(f'Num. populations: {num_population1}, {num_population2}')
        best_agent1 = self._get_best_agent([num_rounds1, num_rounds1], search_radius, [agent1_path, opponent1_path], agent_name, num_population1, n_eval_episodes=1, negative_score_flag=True, n_seeds=n_seeds)
        print(f'Best agent1: {best_agent1}')
        best_opponent1 = self._get_best_agent([num_rounds1, num_rounds1], search_radius, [opponent1_path, agent1_path], opponent_name, num_population1, n_eval_episodes=1, negative_score_flag=False, n_seeds=n_seeds)
        print(f'Best opponent1: {best_opponent1}')
        best_agent2 = self._get_best_agent([num_rounds2, num_rounds2], search_radius, [agent2_path, opponent2_path], agent_name, num_population2, n_eval_episodes=1, negative_score_flag=True, n_seeds=n_seeds)
        print(f'Best agent2: {best_agent2}')
        best_opponent2 = self._get_best_agent([num_rounds2, num_rounds2], search_radius, [opponent2_path, agent2_path], opponent_name, num_population2, n_eval_episodes=1, negative_score_flag=False, n_seeds=n_seeds)
        print(f'Best opponent2: {best_opponent2}')
        print('###############################################################')
        print(f'# Best agent1: {best_agent1}')
        print(f'# Best opponent1: {best_opponent1}')
        print(f'# Best agent2: {best_agent2}')
        print(f'# Best opponent2: {best_opponent2}')
        print('###############################################################')
        print(f'################# Agent1 vs Opponent2 #################')
        perf_agent1_opponent2 = self._compute_performance(best_agent1, best_opponent2, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=True)
        print(f'################# Agent1 vs Opponent1 #################')
        perf_agent1_opponent1 = self._compute_performance(best_agent1, best_opponent1, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=True)
        print(f'################# Agent2 vs Opponent2 #################')
        perf_agent2_opponent2 = self._compute_performance(best_agent2, best_opponent2, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=True)
        print(f'################# Agent2 vs Opponent1 #################')
        perf_agent2_opponent1 = self._compute_performance(best_agent2, best_opponent1, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=True)
        print(f'################# Opponent1 vs Agent2 #################')
        perf_opponent1_agent2 = self._compute_performance(best_opponent1, best_agent2, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=True)
        print(f'################# Opponent1 vs Agent1 #################')
        perf_opponent1_agent1 = self._compute_performance(best_opponent1, best_agent1, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=True)
        print(f'################# Opponent1 vs Agent2 #################')
        perf_opponent2_agent2 = self._compute_performance(best_opponent2, best_agent2, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=True)
        print(f'################# Opponent2 vs Agent1 #################')
        perf_opponent2_agent1 = self._compute_performance(best_opponent2, best_agent1, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=True)
        perf_agent = perf_agent1_opponent2 - perf_agent1_opponent1 + perf_agent2_opponent2 - perf_agent2_opponent1
        perf_opponent = perf_opponent1_agent2 - perf_opponent1_agent1 + perf_opponent2_agent2 - perf_opponent2_agent1
        gain = perf_agent + perf_opponent
        print('-----------------------------------------------------------------')
        print(f'perf_agent: {perf_agent}\tperf_opponent: {perf_opponent}\tgain: {gain}')
        eps = 0.001
        if perf_agent > 0:
            print(f'Configuration 1 is better {1} to generate predators (path: {approach1_path})')
        elif -eps <= perf_agent <= eps:
            print(f'Configuration 1 & 2 are close to each other to generate predators')
        else:
            print(f'Configuration 2 is better {2} to generate predators (path: {approach2_path})')
        if perf_opponent > 0:
            print(f'Configuration 1 is better {1} to generate preys (path: {approach1_path})')
        elif -eps <= perf_opponent <= eps:
            print(f'Configuration 1 & 2 are close to each other to generate prey')
        else:
            print(f'Configuration 2 is better {2} to generate preys (path: {approach2_path})')
        if gain > 0:
            print(f'Configuration 1 is better {1} (path: {approach1_path})')
            return 1
        elif -eps <= gain <= eps:
            print(f'Configuration 1 & 2 are close to each other')
        else:
            print(f'Configuration 2 is better {2} (path: {approach2_path})')
            return 2

    def test(self, experiment_filename=None, logdir=False, wandb=False, n_eval_episodes=1):
        if False:
            i = 10
            return i + 15
        self._init_testing(experiment_filename=experiment_filename, logdir=logdir, wandb=wandb)
        n_eval_episodes_configs = self.testing_configs.get('repetition', None)
        n_eval_episodes = n_eval_episodes_configs if n_eval_episodes_configs is not None else n_eval_episodes
        if self.crosstest_flag:
            n_seeds = self.testing_configs.get('n_seeds', 1)
            self.crosstest(n_eval_episodes, n_seeds)
        else:
            already_evaluated_agents = []
            for k in self.agents_configs.keys():
                agent_configs = self.agents_configs[k]
                agent_name = agent_configs['name']
                agent_opponent_joint = sorted([agent_name, agent_configs['opponent_name']])
                if agent_opponent_joint in already_evaluated_agents:
                    continue
                if self.testing_modes[agent_name] == 'round':
                    self._test_round_by_round(k, n_eval_episodes)
                else:
                    self._test_different_rounds(k, n_eval_episodes)
                already_evaluated_agents.append(agent_opponent_joint)

    def _bug_compute_performance(self, agent, opponent, key, n_eval_episodes=1, n_seeds=1, negative_score_flag=False, render=False, render_extra_info=None):
        if False:
            i = 10
            return i + 15
        (mean_reward, std_reward, win_rate, std_win_rate, render_ret) = self._run_one_evaluation(key, agent, [opponent], n_eval_episodes, render=render, render_extra_info=f'{agent} vs {opponent}' if render_extra_info is None else render_extra_info)
        return mean_reward

    def _bug_compute_performance2(self, agent, opponent, key, n_eval_episodes=1, n_seeds=1, negative_score_flag=False, render=False, render_extra_info=None, agent_model=None, env=None):
        if False:
            while True:
                i = 10
        (mean_reward, std_reward, win_rate, std_win_rate, render_ret) = self._run_one_evaluation(key, agent, [opponent], n_eval_episodes, agent_model=agent_model, env=env, render=render, render_extra_info=f'{agent} vs {opponent}' if render_extra_info is None else render_extra_info)
        return mean_reward

    def bug(self):
        if False:
            print('Hello World!')
        agent = '/home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-crosstest-models/standard1-01.11.2022_21.57.03/pred/history_46_lastreward_m_-230.2_s_1251328_p_0'
        opponent = '/home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-crosstest-models/standard1-01.11.2022_21.57.03/prey/history_47_lastreward_m_802.6_s_1277952_p_0'
        agent_name = 'pred'
        opponent_name = 'prey'
        self.agents_configs = {}
        self.agents_configs['pred'] = {'name': 'pred', 'env_class': 'SelfPlayPredEnv'}
        self.agents_configs['prey'] = {'name': 'prey', 'env_class': 'SelfPlayPreyEnv'}
        self.seed_value = 3
        self.render_sleep_time = 0.0001
        perf_agent_opponent = self._bug_compute_performance(agent, opponent, agent_name, n_eval_episodes=50, n_seeds=1, negative_score_flag=True, render=False)
        perf_opponent_agent = self._bug_compute_performance(opponent, agent, opponent_name, n_eval_episodes=50, n_seeds=1, negative_score_flag=False, render=False)
if __name__ == '__main__':
    testing = SelfPlayTesting()
    testing.bug()