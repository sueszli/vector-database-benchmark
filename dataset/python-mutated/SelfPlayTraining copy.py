from SelfPlayExp import SelfPlayExp
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from archive import ArchiveSB3 as Archive
from callbacks import *
from wandb.integration.sb3 import WandbCallback
import wandb
import gym_predprey
import gym
from gym.envs.registration import register
from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPredEnv
from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPreyEnv
from gym_pz_predprey.envs.SelfPlayPZPredPrey import SelfPlayPZPredEnv
from gym_pz_predprey.envs.SelfPlayPZPredPrey import SelfPlayPZPreyEnv
from gym_predprey_drones.envs.SelfPlayPredPreyDrones1v1 import SelfPlayPredDroneEnv
from gym_predprey_drones.envs.SelfPlayPredPreyDrones1v1 import SelfPlayPreyDroneEnv
from PolicyNetworks import get_policy_arch
from shared import *
from copy import deepcopy
from bach_utils.shared import *
from bach_utils.sorting import population_key, round_key, checkpoint_key, sort_steps
from bach_utils.json_parser import ExperimentParser
import numpy.ma as ma
import threading
THREADED = True

class SelfPlayTraining(SelfPlayExp):

    def __init__(self, seed_value=None):
        if False:
            print('Hello World!')
        super(SelfPlayTraining, self).__init__()
        self.envs = None
        self.eval_envs = None
        self.evalsave_callbacks = None
        self.archives = None
        self.models = None
        self.opponent_selection_callbacks = None
        self.wandb_callbacks = None
        self.seed_value = seed_value
        self.deterministic = False
        self.THREADED = THREADED

    def _init_argparse(self):
        if False:
            i = 10
            return i + 15
        super(SelfPlayTraining, self)._init_argparse(description='Self-play experiment training script', help='The experiemnt configuration file path and name which the experiment should be loaded')

    def _generate_log_dir(self):
        if False:
            for i in range(10):
                print('nop')
        return super(SelfPlayTraining, self)._generate_log_dir(dir_postfix='train')

    def _init_archives(self):
        if False:
            for i in range(10):
                print('nop')
        self.archives = {}
        population_size = self.experiment_configs['population_size']
        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs['name']
            eval_opponent_selection = agent_configs['eval_opponent_selection']
            opponent_selection = agent_configs['opponent_selection']
            self.archives[agent_name] = Archive(sorting_keys=[eval_opponent_selection, opponent_selection], sorting=True, moving_least_freq_flag=False, save_path=os.path.join(self.log_dir, agent_name), delta=agent_configs.get('delta_latest', 0) * population_size)

    def _init_envs(self):
        if False:
            i = 10
            return i + 15
        self.envs = {}
        self.eval_envs = {}
        population_size = self.experiment_configs['population_size']
        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs['name']
            self.envs[agent_name] = []
            self.eval_envs[agent_name] = []
            for population_num in range(population_size):
                opponent_name = agent_configs['opponent_name']
                opponent_archive = self.archives[opponent_name]
                sampling_parameters = {'opponent_selection': agent_configs['opponent_selection'], 'sample_path': os.path.join(self.log_dir, opponent_name), 'randomly_reseed_sampling': agent_configs.get('randomly_reseed_sampling', False)}
                algorithm_class = None
                opponent_algorithm_class_cfg = agent_configs.get('opponent_rl_algorithm', agent_configs['rl_algorithm'])
                if opponent_algorithm_class_cfg == 'PPO':
                    algorithm_class = PPO
                elif opponent_algorithm_class_cfg == 'SAC':
                    algorithm_class = SAC
                self.envs[agent_name].append(super(SelfPlayTraining, self).create_env(key=k, name='Training', algorithm_class=algorithm_class, opponent_archive=opponent_archive, sample_after_reset=agent_configs['sample_after_reset'], sampling_parameters=sampling_parameters, seed_value=self.seed_value + population_num))
                self.eval_envs[agent_name].append(super(SelfPlayTraining, self).create_env(key=k, name='Evaluation', algorithm_class=algorithm_class, opponent_archive=opponent_archive, sample_after_reset=False, sampling_parameters=None, seed_value=self.seed_value + population_num))

    def _init_models(self):
        if False:
            i = 10
            return i + 15
        self.models = {}
        population_size = self.experiment_configs['population_size']
        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs['name']
            self.models[agent_name] = []
            for population_num in range(population_size):
                policy_kwargs = get_policy_arch(str(agent_configs.get('policy_arch', None)))
                policy = None
                if agent_configs['rl_algorithm'] == 'PPO':
                    policy = PPO(agent_configs['policy'], self.envs[agent_name][population_num], clip_range=agent_configs['clip_range'], ent_coef=agent_configs['ent_coef'], learning_rate=agent_configs['lr'], batch_size=agent_configs['batch_size'], gamma=agent_configs['gamma'], verbose=2, tensorboard_log=os.path.join(self.log_dir, agent_name), n_epochs=agent_configs['n_epochs'], n_steps=agent_configs.get('n_steps', 2048), seed=self.seed_value + population_num, policy_kwargs=policy_kwargs)
                elif agent_configs['rl_algorithm'] == 'SAC':
                    policy = SAC(agent_configs['policy'], self.envs[agent_name][population_num], buffer_size=agent_configs['buffer_size'], learning_rate=agent_configs['lr'], batch_size=agent_configs['batch_size'], gamma=agent_configs['gamma'], verbose=agent_configs.get('verbose', 2), tensorboard_log=os.path.join(self.log_dir, agent_name), seed=self.seed_value + population_num, policy_kwargs=policy_kwargs)
                self.models[agent_name].append(policy)

    def _init_callbacks(self):
        if False:
            for i in range(10):
                print('nop')
        self.opponent_selection_callbacks = {}
        self.evalsave_callbacks = {}
        self.wandb_callbacks = {}
        population_size = self.experiment_configs['population_size']
        self.eval_matrix_method = {}
        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs['name']
            opponent_name = agent_configs['opponent_name']
            opponent_sample_path = os.path.join(self.log_dir, opponent_name)
            agent_path = os.path.join(self.log_dir, agent_name)
            self.eval_matrix_method[agent_name] = agent_configs.get('eval_matrix_method', 'reward')
            self.evalsave_callbacks[agent_name] = []
            self.opponent_selection_callbacks[agent_name] = []
            for population_num in range(population_size):
                enable_evaluation_matrix = True
                self.evalsave_callbacks[agent_name].append(EvalSaveCallback(eval_env=self.eval_envs[agent_name][population_num], log_path=agent_path, eval_freq=int(agent_configs['eval_freq']), n_eval_episodes=agent_configs['num_eval_episodes'], deterministic=self.deterministic, save_path=agent_path, eval_metric=agent_configs['eval_metric'], eval_opponent_selection=agent_configs['eval_opponent_selection'], eval_sample_path=opponent_sample_path, save_freq=int(agent_configs['save_freq']), archive={'self': self.archives[agent_name], 'opponent': self.archives[opponent_name]}, agent_name=agent_name, num_rounds=self.experiment_configs['num_rounds'], seed_value=self.seed_value, enable_evaluation_matrix=enable_evaluation_matrix, randomly_reseed_sampling=agent_configs.get('randomly_reseed_sampling', False), eval_matrix_method=self.eval_matrix_method[agent_name]))
                self.evalsave_callbacks[agent_name][-1].population_idx = population_num
                self.opponent_selection_callbacks[agent_name].append(TrainingOpponentSelectionCallback(sample_path=opponent_sample_path, env=self.envs[agent_name][population_num], opponent_selection=agent_configs['opponent_selection'], sample_after_rollout=agent_configs['sample_after_rollout'], sample_after_reset=agent_configs['sample_after_reset'], num_sampled_per_round=agent_configs['num_sampled_opponent_per_round'], archive=self.archives[opponent_name], randomly_reseed_sampling=agent_configs.get('randomly_reseed_sampling', False)))
            self.wandb_callbacks[agent_name] = WandbCallback()

    def _init_training(self, experiment_filename):
        if False:
            print('Hello World!')
        super(SelfPlayTraining, self)._init_exp(experiment_filename, True, True)
        self.clilog.info(f'----- Initialize archives, envs, models, callbacks')
        self._init_archives()
        self._init_envs()
        self._init_models()
        self._init_callbacks()

    def _create_agents_names_list(self):
        if False:
            i = 10
            return i + 15
        agents_order = self.experiment_configs['agents_order']
        agents_names_list = [None for i in range(len(agents_order.keys()))]
        for (k, v) in agents_order.items():
            agents_names_list[int(k)] = v
        return agents_names_list

    def _change_archives(self, agent_name, archive):
        if False:
            i = 10
            return i + 15
        self.archives[agent_name].change_archive_core(archive)

    def _population_thread_func(self, agent_name, population_num):
        if False:
            for i in range(10):
                print('nop')
        self.models[agent_name][population_num].learn(total_timesteps=int(self.agents_configs[agent_name]['num_timesteps']), callback=[self.opponent_selection_callbacks[agent_name][population_num], self.evalsave_callbacks[agent_name][population_num], self.wandb_callbacks[agent_name]], reset_num_timesteps=False)

    def _agent_thread_func(self, agent_name, population_size, round_num):
        if False:
            for i in range(10):
                print('nop')
        opponent_name = self.agents_configs[agent_name]['opponent_name']
        if self.experiment_configs.get('parallel_alternate_training', True):
            self.archives[opponent_name].change_archive_core(self.old_archives[opponent_name])
        threads = []
        for population_num in range(population_size):
            self.clilog.info(f'------------------- Train {agent_name}, round: {round_num},  population: {population_num}--------------------')
            self.clilog.debug(f'Model mem id: {self.models[agent_name][population_num]}')
            if self.THREADED:
                thread = threading.Thread(target=self._population_thread_func, args=(agent_name, population_num))
                threads.append(thread)
                thread.start()
            else:
                self._population_thread_func(agent_name, population_num)
        for (e, thread) in enumerate(threads):
            thread.join()
        self.new_archives[agent_name] = deepcopy(self.archives[agent_name])

    def train(self, experiment_filename=None):
        if False:
            for i in range(10):
                print('nop')
        self._init_training(experiment_filename=experiment_filename)
        num_rounds = self.experiment_configs['num_rounds']
        population_size = self.experiment_configs['population_size']
        agents_names_list = self._create_agents_names_list()
        self.old_archives = {}
        self.new_archives = {}
        old_THREADED = deepcopy(self.THREADED)
        self.THREADED = False
        for round_num in range(num_rounds):
            wandb.log({f'progress (round_num)': round_num})
            for (i, agent_name) in enumerate(agents_names_list):
                for population_num in range(population_size):
                    self.evalsave_callbacks[agent_name][population_num].set_name_prefix(f'history_{round_num}')
                self.old_archives[agent_name] = deepcopy(self.archives[agent_name])
            threads = []
            for (agent_idx, agent_name) in enumerate(agents_names_list):
                if self.THREADED:
                    thread = threading.Thread(target=self._agent_thread_func, args=(agent_name, population_size, round_num))
                    threads.append(thread)
                    thread.start()
                else:
                    self._agent_thread_func(agent_name, population_size, round_num)
            for (e, thread) in enumerate(threads):
                thread.join()
            if self.experiment_configs.get('parallel_alternate_training', True):
                for agent_name in agents_names_list:
                    self.archives[agent_name].change_archive_core(self.new_archives[agent_name])
            if old_THREADED:
                self.THREADED = True
            for (j, agent_name) in enumerate(agents_names_list):
                agent_config = self.agents_configs[agent_name]
                opponent_name = agent_config['opponent_name']
                num_heatmap_eval_episodes = agent_config['num_heatmap_eval_episodes']
                final_save_freq = agent_config['final_save_freq']
                heatmap_log_freq = agent_config['heatmap_log_freq']
                aggregate_eval_matrix = agent_config['aggregate_eval_matrix']
                if aggregate_eval_matrix:
                    self.clilog.info('--------------------------------------------------------------')
                    self.clilog.info(f'Round: {round_num} -> Aggregate HeatMap Evaluation for current round version of {agent_name} vs {opponent_name}')
                    for population_num in range(population_size):
                        if agent_config['rl_algorithm'] == 'PPO':
                            algorithm_class = PPO
                        elif agent_config['rl_algorithm'] == 'SAC':
                            algorithm_class = SAC
                        self.evalsave_callbacks[agent_name][population_num].compute_eval_matrix_aggregate(prefix='history_', round_num=round_num, n_eval_rep=num_heatmap_eval_episodes, algorithm_class=algorithm_class, opponents_path=os.path.join(self.log_dir, opponent_name), agents_path=os.path.join(self.log_dir, agent_name))
                if aggregate_eval_matrix and (round_num % heatmap_log_freq == 0 or round_num == num_rounds - 1):
                    evaluation_matrices = []
                    for population_num in range(population_size):
                        evaluation_matrix = self.evalsave_callbacks[agent_name][population_num].evaluation_matrix
                        evaluation_matrix = evaluation_matrix if j % 2 == 0 else evaluation_matrix.T
                        evaluation_matrices.append(evaluation_matrix)
                    mean_evaluation_matrix = np.mean(evaluation_matrices, axis=0)
                    std_evaluation_matrix = np.std(evaluation_matrices, axis=0)
                    if round_num == num_rounds - 1:
                        wandb.log({f'{agent_name}/heatmap': wandb.plots.HeatMap([i for i in range(num_rounds)], [i for i in range(num_rounds)], mean_evaluation_matrix, show_text=True)})
                        wandb.log({f'{agent_name}/std_heatmap': wandb.plots.HeatMap([i for i in range(num_rounds)], [i for i in range(num_rounds)], std_evaluation_matrix, show_text=True)})
                    wandb.log({f'{agent_name}/mid_eval/heatmap': wandb.plots.HeatMap([i for i in range(num_rounds)], [i for i in range(num_rounds)], mean_evaluation_matrix, show_text=False)})
                    wandb.log({f'{agent_name}/mid_eval/std_heatmap': wandb.plots.HeatMap([i for i in range(num_rounds)], [i for i in range(num_rounds)], std_evaluation_matrix, show_text=False)})
                    np.save(os.path.join(self.log_dir, agent_name, 'evaluation_matrix'), mean_evaluation_matrix)
                    wandb.save(os.path.join(self.log_dir, agent_name, 'evaluation_matrix') + '.npy')
                if round_num % final_save_freq == 0 or round_num == num_rounds - 1:
                    self.clilog.info(f'------------------- Models saving freq --------------------')
                    for population_num in range(population_size):
                        self.models[agent_name][population_num].save(os.path.join(self.log_dir, agent_name, f'final_model_pop{population_num}'))
                    self.models[agent_name][-1].save(os.path.join(self.log_dir, agent_name, 'final_model'))
        for (j, agent_name) in enumerate(agents_names_list):
            agent_config = self.agents_configs[agent_name]
            aggregate_eval_matrix = agent_config['aggregate_eval_matrix']
            opponent_name = agent_config['opponent_name']
            self.clilog.info(f'------------------- Prepare freq log used by {agent_name} ({opponent_name} archive) --------------------')
            num_heatmap_eval_episodes = agent_config['num_heatmap_eval_episodes']
            eval_matrix_testing_freq = agent_config['eval_matrix_testing_freq']
            (freq_keys, freq_values) = self.archives[opponent_name].get_freq()
            freq_dict = dict(zip(freq_keys, freq_values))
            max_checkpoint_num = 1
            for population_num in range(population_size):
                max_checkpoint_num = max(max_checkpoint_num, self.evalsave_callbacks[opponent_name][population_num].max_checkpoint_num)
            max_checkpoint_num = 1
            freq_matrix = np.zeros((population_size, max_checkpoint_num * num_rounds))
            sorted_keys = sort_steps(list(freq_keys))
            axis = [[i for i in range(num_rounds)], [i for i in range(population_size)]]
            for (i, key) in enumerate(sorted_keys):
                population_num = population_key(key)
                round_num = round_key(key)
                checkpoint_num = checkpoint_key(key)
                val = freq_dict[key]
                freq_matrix[population_num, round_num] += val
            wandb.log({f'{agent_name}vs({opponent_name}_archive)/freq_heatmap': wandb.plots.HeatMap(axis[0], axis[1], freq_matrix, show_text=True)})
            wandb.log({f'{agent_name}vs({opponent_name}_archive)/freq_heatmap_no_text': wandb.plots.HeatMap(axis[0], axis[1], freq_matrix, show_text=False)})
            mean_freq_heatmap = np.mean(freq_matrix, axis=0)
            std_freq_heatmap = np.std(freq_matrix, axis=0)
            stat_freq_heatmap = np.vstack((mean_freq_heatmap, std_freq_heatmap))
            wandb.log({f'{agent_name}vs({opponent_name}_archive)/stat_freq_heatmap': wandb.plots.HeatMap(axis[0], ['mean', 'std'], stat_freq_heatmap, show_text=True)})
        self.evaluation_configs['log_dir'] = self.log_dir
        self.evaluation_configs['log_main_dir'] = self.log_main_dir
        for (j, agent_name) in enumerate(agents_names_list):
            self.clilog.info(f' ----------- Evaluation for {agent_name} -----------')
            if (j + 1) % 2:
                self.clilog.warn('Note the score is inversed for length, it is not length but time elapsed')
            agent_config = self.agents_configs[agent_name]
            aggregate_eval_matrix = agent_config['aggregate_eval_matrix']
            if not aggregate_eval_matrix:
                opponent_name = agent_config['opponent_name']
                num_heatmap_eval_episodes = agent_config['num_heatmap_eval_episodes']
                eval_matrix_testing_freq = agent_config['eval_matrix_testing_freq']
                maximize_indicator = True
                evaluation_matrices = []
                best_agents_population = {}
                best_agent_search_radius = agent_config.get('best_agent_search_radius', num_rounds)
                for population_num in range(population_size):
                    self.clilog.info(f'Full evaluation matrix for {agent_name} (population: {population_num})')
                    algorithm_class = None
                    if agent_config['rl_algorithm'] == 'PPO':
                        algorithm_class = PPO
                    elif agent_config['rl_algorithm'] == 'SAC':
                        algorithm_class = SAC
                    (axis, agent_names) = self.evalsave_callbacks[agent_name][population_num].compute_eval_matrix(prefix='history_', num_rounds=num_rounds, n_eval_rep=num_heatmap_eval_episodes, algorithm_class=algorithm_class, opponents_path=os.path.join(self.log_dir, opponent_name), agents_path=os.path.join(self.log_dir, agent_name), freq=eval_matrix_testing_freq, population_size=population_size, negative_indicator=(j + 1) % 2)
                    evaluation_matrix = self.evalsave_callbacks[agent_name][population_num].evaluation_matrix
                    evaluation_matrix = evaluation_matrix if j % 2 == 0 else evaluation_matrix.T
                    evaluation_matrices.append(evaluation_matrix)
                    num_eval_rounds = len(axis[0])
                    best_agent_search_radius = min(best_agent_search_radius, num_eval_rounds)
                    mask = np.ones((num_eval_rounds, num_eval_rounds))
                    mask_initial_idx = num_eval_rounds - best_agent_search_radius
                    mask[mask_initial_idx:, :] = np.zeros((best_agent_search_radius, num_eval_rounds))
                    agent_names = np.array(agent_names)
                    eval_mask = mask if j % 2 == 0 else mask.T
                    shape = (best_agent_search_radius, num_eval_rounds)
                    masked_evaluation_matrix = evaluation_matrix[eval_mask == 0].reshape(shape)
                    masked_evaluation_matrix = masked_evaluation_matrix if j % 2 == 0 else masked_evaluation_matrix.T
                    agent_names = agent_names[mask[:, 0] == 0]
                    (best_agent_name, best_agent_score) = get_best_agent_from_eval_mat(masked_evaluation_matrix, agent_names, axis=j, maximize=maximize_indicator)
                    best_agents_population[best_agent_name] = best_agent_score
                (best_agent_name, best_agent_score) = get_best_agent_from_vector(list(best_agents_population.values()), list(best_agents_population.keys()), maximize=maximize_indicator)
                self.evaluation_configs[agent_name] = {'best_agent_name': best_agent_name, 'best_agent_score': best_agent_score, 'best_agent_method': self.eval_matrix_method[agent_name]}
                self.clilog.info(f'Best agent for {agent_name} -> {best_agent_name}, score: {best_agent_score}')
                mean_evaluation_matrix = np.mean(evaluation_matrices, axis=0)
                std_evaluation_matrix = np.std(evaluation_matrices, axis=0)
                wandb.log({f'{agent_name}/heatmap': wandb.plots.HeatMap(axis[0], axis[1], mean_evaluation_matrix, show_text=True)})
                wandb.log({f'{agent_name}/mid_eval/heatmap': wandb.plots.HeatMap(axis[0], axis[1], mean_evaluation_matrix, show_text=False)})
                wandb.log({f'{agent_name}/std_heatmap': wandb.plots.HeatMap(axis[0], axis[1], std_evaluation_matrix, show_text=True)})
                np.save(os.path.join(self.log_dir, agent_name, 'evaluation_matrix_axis_x'), axis[0])
                np.save(os.path.join(self.log_dir, agent_name, 'evaluation_matrix_axis_y'), axis[1])
                np.save(os.path.join(self.log_dir, agent_name, 'evaluation_matrix'), mean_evaluation_matrix)
                wandb.save(os.path.join(self.log_dir, agent_name, 'evaluation_matrix') + '.npy')
                wandb.save(os.path.join(self.log_dir, agent_name, 'evaluation_matrix_axis_x') + '.npy')
                wandb.save(os.path.join(self.log_dir, agent_name, 'evaluation_matrix_axis_y') + '.npy')
            self.clilog.info('Save experiment configuration with ')
            log_file = os.path.join(self.log_dir, 'experiment_config.json')
            ExperimentParser.save(log_file, self.experiment_configs, self.agents_configs, self.evaluation_configs, self.testing_configs)
            wandb.save(log_file)
            post_eval_list = []
            for population_num in range(population_size):
                self.clilog.info('-----------------------------------------------------------------------')
                self.clilog.info(f'Post Evaluation for {agent_name} (population: {population_num})')
                self.clilog.info('-----------------------------------------------------------------------')
                eval_return_list = self.evalsave_callbacks[agent_name][population_num].post_eval(opponents_path=os.path.join(self.log_dir, self.agents_configs[agent_name]['opponent_name']), population_size=population_size)
                post_eval_list.append(eval_return_list)
                self.envs[agent_name][population_num].close()
                self.eval_envs[agent_name][population_num].close()
            mean_post_eval = np.mean(post_eval_list, axis=0)
            std_post_eval = np.std(post_eval_list, axis=0)
            data = [[x, y] for (x, y) in zip([i for i in range(len(mean_post_eval))], mean_post_eval)]
            table = wandb.Table(data=data, columns=['opponent idx', 'win-rate'])
            std_data = [[x, y] for (x, y) in zip([i for i in range(len(std_post_eval))], std_post_eval)]
            std_table = wandb.Table(data=std_data, columns=['opponent idx', 'win-rate'])
            wandb.log({f'{agent_name}/post_eval/table': wandb.plot.line(table, 'opponent idx', 'win-rate', title=f'Post evaluation {agent_name}')})
            wandb.log({f'{agent_name}/post_eval/std_table': wandb.plot.line(std_table, 'opponent idx', 'win-rate', title=f'Std Post evaluation {agent_name}')})
if __name__ == '__main__':
    from SelfPlayTraining_threaded import SelfPlayTraining
    training = SelfPlayTraining()
    training.train()