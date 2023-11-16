from typing import Optional, Any, List
from collections import namedtuple
from easydict import EasyDict
import copy
import numpy as np
import torch
from ding.envs import BaseEnvManager
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY, one_time_warning, get_rank, get_world_size, broadcast_object_list, allreduce_data
from ding.torch_utils import to_tensor, to_ndarray
from .base_serial_collector import ISerialCollector, CachePool, TrajBuffer, INF, to_tensor_transitions

@SERIAL_COLLECTOR_REGISTRY.register('sample')
class SampleSerialCollector(ISerialCollector):
    """
    Overview:
        Sample collector(n_sample), a sample is one training sample for updating model,
        it is usually like <s, a, s', r, d>(one transition)
        while is a trajectory with many transitions, which is often used in RNN-model.
    Interfaces:
        __init__, reset, reset_env, reset_policy, collect, close
    Property:
        envstep
    """
    config = dict(deepcopy_obs=False, transform_obs=False, collect_print_freq=100)

    def __init__(self, cfg: EasyDict, env: BaseEnvManager=None, policy: namedtuple=None, tb_logger: 'SummaryWriter'=None, exp_name: Optional[str]='default_experiment', instance_name: Optional[str]='collector') -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Initialization method.\n        Arguments:\n            - cfg (:obj:`EasyDict`): Config dict\n            - env (:obj:`BaseEnvManager`): the subclass of vectorized env_manager(BaseEnvManager)\n            - policy (:obj:`namedtuple`): the api namedtuple of collect_mode policy\n            - tb_logger (:obj:`SummaryWriter`): tensorboard handle\n        '
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._collect_print_freq = cfg.collect_print_freq
        self._deepcopy_obs = cfg.deepcopy_obs
        self._transform_obs = cfg.transform_obs
        self._cfg = cfg
        self._timer = EasyTimer()
        self._end_flag = False
        self._rank = get_rank()
        self._world_size = get_world_size()
        if self._rank == 0:
            if tb_logger is not None:
                (self._logger, _) = build_logger(path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False)
                self._tb_logger = tb_logger
            else:
                (self._logger, self._tb_logger) = build_logger(path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name)
        else:
            (self._logger, _) = build_logger(path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False)
            self._tb_logger = None
        self.reset(policy, env)

    def reset_env(self, _env: Optional[BaseEnvManager]=None) -> None:
        if False:
            return 10
        '\n        Overview:\n            Reset the environment.\n            If _env is None, reset the old environment.\n            If _env is not None, replace the old environment in the collector with the new passed                 in environment and launch.\n        Arguments:\n            - env (:obj:`Optional[BaseEnvManager]`): instance of the subclass of vectorized                 env_manager(BaseEnvManager)\n        '
        if _env is not None:
            self._env = _env
            self._env.launch()
            self._env_num = self._env.env_num
        else:
            self._env.reset()

    def reset_policy(self, _policy: Optional[namedtuple]=None) -> None:
        if False:
            return 10
        '\n        Overview:\n            Reset the policy.\n            If _policy is None, reset the old policy.\n            If _policy is not None, replace the old policy in the collector with the new passed in policy.\n        Arguments:\n            - policy (:obj:`Optional[namedtuple]`): the api namedtuple of collect_mode policy\n        '
        assert hasattr(self, '_env'), 'please set env first'
        if _policy is not None:
            self._policy = _policy
            self._policy_cfg = self._policy.get_attribute('cfg')
            self._default_n_sample = _policy.get_attribute('n_sample')
            self._traj_len_inf = self._policy_cfg.traj_len_inf
            self._unroll_len = _policy.get_attribute('unroll_len')
            self._on_policy = _policy.get_attribute('on_policy')
            if self._default_n_sample is not None and (not self._traj_len_inf):
                self._traj_len = max(self._unroll_len, self._default_n_sample // self._env_num + int(self._default_n_sample % self._env_num != 0))
                self._logger.debug('Set default n_sample mode(n_sample({}), env_num({}), traj_len({}))'.format(self._default_n_sample, self._env_num, self._traj_len))
            else:
                self._traj_len = INF
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple]=None, _env: Optional[BaseEnvManager]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Reset the environment and policy.\n            If _env is None, reset the old environment.\n            If _env is not None, replace the old environment in the collector with the new passed                 in environment and launch.\n            If _policy is None, reset the old policy.\n            If _policy is not None, replace the old policy in the collector with the new passed in policy.\n        Arguments:\n            - policy (:obj:`Optional[namedtuple]`): the api namedtuple of collect_mode policy\n            - env (:obj:`Optional[BaseEnvManager]`): instance of the subclass of vectorized                 env_manager(BaseEnvManager)\n        '
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)
        if self._policy_cfg.type == 'dreamer_command':
            self._states = None
            self._resets = np.array([False for i in range(self._env_num)])
        self._obs_pool = CachePool('obs', self._env_num, deepcopy=self._deepcopy_obs)
        self._policy_output_pool = CachePool('policy_output', self._env_num)
        maxlen = self._traj_len if self._traj_len != INF else None
        self._traj_buffer = {env_id: TrajBuffer(maxlen=maxlen, deepcopy=self._deepcopy_obs) for env_id in range(self._env_num)}
        self._env_info = {env_id: {'time': 0.0, 'step': 0, 'train_sample': 0} for env_id in range(self._env_num)}
        self._episode_info = []
        self._total_envstep_count = 0
        self._total_episode_count = 0
        self._total_train_sample_count = 0
        self._total_duration = 0
        self._last_train_iter = 0
        self._end_flag = False

    def _reset_stat(self, env_id: int) -> None:
        if False:
            return 10
        "\n        Overview:\n            Reset the collector's state. Including reset the traj_buffer, obs_pool, policy_output_pool                and env_info. Reset these states according to env_id. You can refer to base_serial_collector                to get more messages.\n        Arguments:\n            - env_id (:obj:`int`): the id where we need to reset the collector's state\n        "
        self._traj_buffer[env_id].clear()
        self._obs_pool.reset(env_id)
        self._policy_output_pool.reset(env_id)
        self._env_info[env_id] = {'time': 0.0, 'step': 0, 'train_sample': 0}

    @property
    def envstep(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Print the total envstep count.\n        Return:\n            - envstep (:obj:`int`): the total envstep count\n        '
        return self._total_envstep_count

    def close(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Close the collector. If end_flag is False, close the environment, flush the tb_logger                and close the tb_logger.\n        '
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        if self._tb_logger:
            self._tb_logger.flush()
            self._tb_logger.close()

    def __del__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Execute the close command and close the collector. __del__ is automatically called to                 destroy the collector instance when the collector finishes its work\n        '
        self.close()

    def collect(self, n_sample: Optional[int]=None, train_iter: int=0, drop_extra: bool=True, random_collect: bool=False, record_random_collect: bool=True, policy_kwargs: Optional[dict]=None, level_seeds: Optional[List]=None) -> List[Any]:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Collect `n_sample` data with policy_kwargs, which is already trained `train_iter` iterations.\n        Arguments:\n            - n_sample (:obj:`int`): The number of collecting data sample.\n            - train_iter (:obj:`int`): The number of training iteration when calling collect method.\n            - drop_extra (:obj:`bool`): Whether to drop extra return_data more than `n_sample`.\n            - record_random_collect (:obj:`bool`) :Whether to output logs of random collect.\n            - policy_kwargs (:obj:`dict`): The keyword args for policy forward.\n            - level_seeds (:obj:`dict`): Used in PLR, represents the seed of the environment that                 generate the data\n        Returns:\n            - return_data (:obj:`List`): A list containing training samples.\n        '
        if n_sample is None:
            if self._default_n_sample is None:
                raise RuntimeError('Please specify collect n_sample')
            else:
                n_sample = self._default_n_sample
        if n_sample % self._env_num != 0:
            one_time_warning('Please make sure env_num is divisible by n_sample: {}/{}, '.format(n_sample, self._env_num) + 'which may cause convergence problems in a few algorithms')
        if policy_kwargs is None:
            policy_kwargs = {}
        collected_sample = 0
        collected_step = 0
        collected_episode = 0
        return_data = []
        while collected_sample < n_sample:
            with self._timer:
                obs = self._env.ready_obs
                self._obs_pool.update(obs)
                if self._transform_obs:
                    obs = to_tensor(obs, dtype=torch.float32)
                if self._policy_cfg.type == 'dreamer_command' and (not random_collect):
                    policy_output = self._policy.forward(obs, **policy_kwargs, reset=self._resets, state=self._states)
                    self._states = [output['state'] for output in policy_output.values()]
                else:
                    policy_output = self._policy.forward(obs, **policy_kwargs)
                self._policy_output_pool.update(policy_output)
                actions = {env_id: output['action'] for (env_id, output) in policy_output.items()}
                actions = to_ndarray(actions)
                timesteps = self._env.step(actions)
            interaction_duration = self._timer.value / len(timesteps)
            for (env_id, timestep) in timesteps.items():
                with self._timer:
                    if timestep.info.get('abnormal', False):
                        self._env.reset({env_id: None})
                        self._policy.reset([env_id])
                        self._reset_stat(env_id)
                        self._logger.info('Env{} returns a abnormal step, its info is {}'.format(env_id, timestep.info))
                        continue
                    if self._policy_cfg.type == 'dreamer_command' and (not random_collect):
                        self._resets[env_id] = timestep.done
                    if self._policy_cfg.type == 'ngu_command':
                        transition = self._policy.process_transition(self._obs_pool[env_id], self._policy_output_pool[env_id], timestep, env_id)
                    else:
                        transition = self._policy.process_transition(self._obs_pool[env_id], self._policy_output_pool[env_id], timestep)
                        if level_seeds is not None:
                            transition['seed'] = level_seeds[env_id]
                    transition['collect_iter'] = train_iter
                    self._traj_buffer[env_id].append(transition)
                    self._env_info[env_id]['step'] += 1
                    collected_step += 1
                    if timestep.done or len(self._traj_buffer[env_id]) == self._traj_len:
                        transitions = to_tensor_transitions(self._traj_buffer[env_id], not self._deepcopy_obs)
                        train_sample = self._policy.get_train_sample(transitions)
                        return_data.extend(train_sample)
                        self._env_info[env_id]['train_sample'] += len(train_sample)
                        collected_sample += len(train_sample)
                        self._traj_buffer[env_id].clear()
                self._env_info[env_id]['time'] += self._timer.value + interaction_duration
                if timestep.done:
                    collected_episode += 1
                    reward = timestep.info['eval_episode_return']
                    info = {'reward': reward, 'time': self._env_info[env_id]['time'], 'step': self._env_info[env_id]['step'], 'train_sample': self._env_info[env_id]['train_sample']}
                    self._episode_info.append(info)
                    self._policy.reset([env_id])
                    self._reset_stat(env_id)
        collected_duration = sum([d['time'] for d in self._episode_info])
        if self._world_size > 1:
            collected_sample = allreduce_data(collected_sample, 'sum')
            collected_step = allreduce_data(collected_step, 'sum')
            collected_episode = allreduce_data(collected_episode, 'sum')
            collected_duration = allreduce_data(collected_duration, 'sum')
        self._total_envstep_count += collected_step
        self._total_episode_count += collected_episode
        self._total_duration += collected_duration
        self._total_train_sample_count += collected_sample
        if record_random_collect:
            self._output_log(train_iter)
        else:
            self._episode_info.clear()
        if self._on_policy:
            for env_id in range(self._env_num):
                self._reset_stat(env_id)
        if drop_extra:
            return return_data[:n_sample]
        else:
            return return_data

    def _output_log(self, train_iter: int) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Print the output log information. You can refer to the docs of `Best Practice` to understand             the training generated logs and tensorboards.\n        Arguments:\n            - train_iter (:obj:`int`): the number of training iteration.\n        '
        if self._rank != 0:
            return
        if train_iter - self._last_train_iter >= self._collect_print_freq and len(self._episode_info) > 0:
            self._last_train_iter = train_iter
            episode_count = len(self._episode_info)
            envstep_count = sum([d['step'] for d in self._episode_info])
            train_sample_count = sum([d['train_sample'] for d in self._episode_info])
            duration = sum([d['time'] for d in self._episode_info])
            episode_return = [d['reward'] for d in self._episode_info]
            info = {'episode_count': episode_count, 'envstep_count': envstep_count, 'train_sample_count': train_sample_count, 'avg_envstep_per_episode': envstep_count / episode_count, 'avg_sample_per_episode': train_sample_count / episode_count, 'avg_envstep_per_sec': envstep_count / duration, 'avg_train_sample_per_sec': train_sample_count / duration, 'avg_episode_per_sec': episode_count / duration, 'reward_mean': np.mean(episode_return), 'reward_std': np.std(episode_return), 'reward_max': np.max(episode_return), 'reward_min': np.min(episode_return), 'total_envstep_count': self._total_envstep_count, 'total_train_sample_count': self._total_train_sample_count, 'total_episode_count': self._total_episode_count}
            self._episode_info.clear()
            self._logger.info('collect end:\n{}'.format('\n'.join(['{}: {}'.format(k, v) for (k, v) in info.items()])))
            for (k, v) in info.items():
                if k in ['each_reward']:
                    continue
                self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
                if k in ['total_envstep_count']:
                    continue
                self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, self._total_envstep_count)