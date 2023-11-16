from typing import Any, Union, List, Tuple, Dict, Callable, Optional
from multiprocessing import connection, get_context
from collections import namedtuple
from ditk import logging
import platform
import time
import copy
import gymnasium
import gym
import traceback
import torch
import pickle
import numpy as np
import treetensor.numpy as tnp
from easydict import EasyDict
from types import MethodType
from ding.data import ShmBufferContainer, ShmBuffer
from ding.envs.env import BaseEnvTimestep
from ding.utils import PropagatingThread, LockContextType, LockContext, ENV_MANAGER_REGISTRY, make_key_as_identifier, remove_illegal_item, CloudPickleWrapper
from .base_env_manager import BaseEnvManager, EnvState, timeout_wrapper

def is_abnormal_timestep(timestep: namedtuple) -> bool:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(timestep.info, dict):
        return timestep.info.get('abnormal', False)
    elif isinstance(timestep.info, list) or isinstance(timestep.info, tuple):
        return timestep.info[0].get('abnormal', False) or timestep.info[1].get('abnormal', False)
    else:
        raise TypeError('invalid env timestep type: {}'.format(type(timestep.info)))

@ENV_MANAGER_REGISTRY.register('async_subprocess')
class AsyncSubprocessEnvManager(BaseEnvManager):
    """
    Overview:
        Create an AsyncSubprocessEnvManager to manage multiple environments.
        Each Environment is run by a respective subprocess.
    Interfaces:
        seed, launch, ready_obs, step, reset, active_env
    """
    config = dict(episode_num=float('inf'), max_retry=5, step_timeout=None, auto_reset=True, retry_type='reset', reset_timeout=None, retry_waiting_time=0.1, shared_memory=True, copy_on_get=True, context='spawn' if platform.system().lower() == 'windows' else 'fork', wait_num=2, step_wait_timeout=0.01, connect_timeout=60, reset_inplace=False)

    def __init__(self, env_fn: List[Callable], cfg: EasyDict=EasyDict({})) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Initialize the AsyncSubprocessEnvManager.\n        Arguments:\n            - env_fn (:obj:`List[Callable]`): The function to create environment\n            - cfg (:obj:`EasyDict`): Config\n\n        .. note::\n\n            - wait_num: for each time the minimum number of env return to gather\n            - step_wait_timeout: for each time the minimum number of env return to gather\n        '
        super().__init__(env_fn, cfg)
        self._shared_memory = self._cfg.shared_memory
        self._copy_on_get = self._cfg.copy_on_get
        self._context = self._cfg.context
        self._wait_num = self._cfg.wait_num
        self._step_wait_timeout = self._cfg.step_wait_timeout
        self._lock = LockContext(LockContextType.THREAD_LOCK)
        self._connect_timeout = self._cfg.connect_timeout
        self._async_args = {'step': {'wait_num': min(self._wait_num, self._env_num), 'timeout': self._step_wait_timeout}}
        self._reset_inplace = self._cfg.reset_inplace
        if not self._auto_reset:
            assert not self._reset_inplace, 'reset_inplace is unavailable when auto_reset=False.'

    def _create_state(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Fork/spawn sub-processes(Call ``_create_env_subprocess``) and create pipes to transfer the data.\n        '
        self._env_episode_count = {env_id: 0 for env_id in range(self.env_num)}
        self._ready_obs = {env_id: None for env_id in range(self.env_num)}
        self._reset_param = {i: {} for i in range(self.env_num)}
        if self._shared_memory:
            obs_space = self._observation_space
            if isinstance(obs_space, (gym.spaces.Dict, gymnasium.spaces.Dict)):
                shape = {k: v.shape for (k, v) in obs_space.spaces.items()}
                dtype = {k: v.dtype for (k, v) in obs_space.spaces.items()}
            else:
                shape = obs_space.shape
                dtype = obs_space.dtype
            self._obs_buffers = {env_id: ShmBufferContainer(dtype, shape, copy_on_get=self._copy_on_get) for env_id in range(self.env_num)}
        else:
            self._obs_buffers = {env_id: None for env_id in range(self.env_num)}
        (self._pipe_parents, self._pipe_children) = ({}, {})
        self._subprocesses = {}
        for env_id in range(self.env_num):
            self._create_env_subprocess(env_id)
        self._waiting_env = {'step': set()}
        self._closed = False

    def _create_env_subprocess(self, env_id):
        if False:
            print('Hello World!')
        ctx = get_context(self._context)
        (self._pipe_parents[env_id], self._pipe_children[env_id]) = ctx.Pipe()
        self._subprocesses[env_id] = ctx.Process(target=self.worker_fn_robust, args=(self._pipe_parents[env_id], self._pipe_children[env_id], CloudPickleWrapper(self._env_fn[env_id]), self._obs_buffers[env_id], self.method_name_list, self._reset_timeout, self._step_timeout, self._reset_inplace), daemon=True, name='subprocess_env_manager{}_{}'.format(env_id, time.time()))
        self._subprocesses[env_id].start()
        self._pipe_children[env_id].close()
        self._env_states[env_id] = EnvState.INIT
        if self._env_replay_path is not None:
            self._pipe_parents[env_id].send(['enable_save_replay', [self._env_replay_path[env_id]], {}])
            self._pipe_parents[env_id].recv()

    @property
    def ready_env(self) -> List[int]:
        if False:
            while True:
                i = 10
        active_env = [i for (i, s) in self._env_states.items() if s == EnvState.RUN]
        return [i for i in active_env if i not in self._waiting_env['step']]

    @property
    def ready_obs(self) -> Dict[int, Any]:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Get the next observations.\n        Return:\n            A dictionary with observations and their environment IDs.\n        Note:\n            The observations are returned in np.ndarray.\n        Example:\n            >>>     obs_dict = env_manager.ready_obs\n            >>>     actions_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}\n        '
        no_done_env_idx = [i for (i, s) in self._env_states.items() if s != EnvState.DONE]
        sleep_count = 0
        while not any([self._env_states[i] == EnvState.RUN for i in no_done_env_idx]):
            if sleep_count != 0 and sleep_count % 10000 == 0:
                logging.warning('VEC_ENV_MANAGER: all the not done envs are resetting, sleep {} times'.format(sleep_count))
            time.sleep(0.001)
            sleep_count += 1
        return {i: self._ready_obs[i] for i in self.ready_env}

    @property
    def ready_imgs(self, render_mode: Optional[str]='rgb_array') -> Dict[int, Any]:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Get the next renderd frames.\n        Return:\n            A dictionary with rendered frames and their environment IDs.\n        Note:\n            The rendered frames are returned in np.ndarray.\n        '
        for i in self.ready_env:
            self._pipe_parents[i].send(['render', None, {'render_mode': render_mode}])
        data = {i: self._pipe_parents[i].recv() for i in self.ready_env}
        self._check_data(data)
        return data

    def launch(self, reset_param: Optional[Dict]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Set up the environments and their parameters.\n        Arguments:\n            - reset_param (:obj:`Optional[Dict]`): Dict of reset parameters for each environment, key is the env_id,                 value is the cooresponding reset parameters.\n        '
        assert self._closed, 'please first close the env manager'
        if reset_param is not None:
            assert len(reset_param) == len(self._env_fn)
        self._create_state()
        self.reset(reset_param)

    def reset(self, reset_param: Optional[Dict]=None) -> None:
        if False:
            return 10
        '\n        Overview:\n            Reset the environments their parameters.\n        Arguments:\n            - reset_param (:obj:`List`): Dict of reset parameters for each environment, key is the env_id,                 value is the cooresponding reset parameters.\n        '
        self._check_closed()
        if reset_param is None:
            reset_env_list = [env_id for env_id in range(self._env_num)]
        else:
            reset_env_list = reset_param.keys()
            for env_id in reset_param:
                self._reset_param[env_id] = reset_param[env_id]
        for env_id in reset_env_list:
            if env_id in self._waiting_env['step']:
                self._pipe_parents[env_id].recv()
                self._waiting_env['step'].remove(env_id)
        sleep_count = 0
        while any([self._env_states[i] == EnvState.RESET for i in reset_env_list]):
            if sleep_count != 0 and sleep_count % 10000 == 0:
                logging.warning('VEC_ENV_MANAGER: not all the envs finish resetting, sleep {} times'.format(sleep_count))
            time.sleep(0.001)
            sleep_count += 1
        reset_thread_list = []
        for (i, env_id) in enumerate(reset_env_list):
            if self._env_seed[env_id] is not None:
                try:
                    if self._env_dynamic_seed is not None:
                        self._pipe_parents[env_id].send(['seed', [self._env_seed[env_id], self._env_dynamic_seed], {}])
                    else:
                        self._pipe_parents[env_id].send(['seed', [self._env_seed[env_id]], {}])
                    ret = self._pipe_parents[env_id].recv()
                    self._check_data({env_id: ret})
                    self._env_seed[env_id] = None
                except BaseException as e:
                    logging.warning('subprocess reset set seed failed, ignore and continue... \n subprocess exception traceback: \n' + traceback.format_exc())
            self._env_states[env_id] = EnvState.RESET
            reset_thread = PropagatingThread(target=self._reset, args=(env_id,))
            reset_thread.daemon = True
            reset_thread_list.append(reset_thread)
        for t in reset_thread_list:
            t.start()
        for t in reset_thread_list:
            t.join()

    def _reset(self, env_id: int) -> None:
        if False:
            i = 10
            return i + 15

        def reset_fn():
            if False:
                return 10
            if self._pipe_parents[env_id].poll():
                recv_data = self._pipe_parents[env_id].recv()
                raise RuntimeError('unread data left before sending to the pipe: {}'.format(repr(recv_data)))
            if self._reset_param[env_id] is not None:
                assert isinstance(self._reset_param[env_id], dict), type(self._reset_param[env_id])
                self._pipe_parents[env_id].send(['reset', [], self._reset_param[env_id]])
            else:
                self._pipe_parents[env_id].send(['reset', [], None])
            if not self._pipe_parents[env_id].poll(self._connect_timeout):
                raise ConnectionError('env reset connection timeout')
            obs = self._pipe_parents[env_id].recv()
            self._check_data({env_id: obs}, close=False)
            if self._shared_memory:
                obs = self._obs_buffers[env_id].get()
            with self._lock:
                self._env_states[env_id] = EnvState.RUN
                self._ready_obs[env_id] = obs
        exceptions = []
        for _ in range(self._max_retry):
            try:
                reset_fn()
                return
            except BaseException as e:
                logging.info('subprocess exception traceback: \n' + traceback.format_exc())
                if self._retry_type == 'renew' or isinstance(e, pickle.UnpicklingError):
                    self._pipe_parents[env_id].close()
                    if self._subprocesses[env_id].is_alive():
                        self._subprocesses[env_id].terminate()
                    self._create_env_subprocess(env_id)
                exceptions.append(e)
                time.sleep(self._retry_waiting_time)
        logging.error('Env {} reset has exceeded max retries({})'.format(env_id, self._max_retry))
        runtime_error = RuntimeError('Env {} reset has exceeded max retries({}), and the latest exception is: {}'.format(env_id, self._max_retry, str(exceptions[-1])))
        runtime_error.__traceback__ = exceptions[-1].__traceback__
        if self._closed:
            return
        else:
            self.close()
            raise runtime_error

    def step(self, actions: Dict[int, Any]) -> Dict[int, namedtuple]:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Step all environments. Reset an env if done.\n        Arguments:\n            - actions (:obj:`Dict[int, Any]`): {env_id: action}\n        Returns:\n            - timesteps (:obj:`Dict[int, namedtuple]`): {env_id: timestep}. Timestep is a                 ``BaseEnvTimestep`` tuple with observation, reward, done, env_info.\n        Example:\n            >>>     actions_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}\n            >>>     timesteps = env_manager.step(actions_dict):\n            >>>     for env_id, timestep in timesteps.items():\n            >>>         pass\n\n        .. note:\n\n            - The env_id that appears in ``actions`` will also be returned in ``timesteps``.\n            - Each environment is run by a subprocess separately. Once an environment is done, it is reset immediately.\n            - Async subprocess env manager use ``connection.wait`` to poll.\n        '
        self._check_closed()
        env_ids = list(actions.keys())
        assert all([self._env_states[env_id] == EnvState.RUN for env_id in env_ids]), 'current env state are: {}, please check whether the requested env is in reset or done'.format({env_id: self._env_states[env_id] for env_id in env_ids})
        for (env_id, act) in actions.items():
            self._pipe_parents[env_id].send(['step', [act], None])
        timesteps = {}
        step_args = self._async_args['step']
        (wait_num, timeout) = (min(step_args['wait_num'], len(env_ids)), step_args['timeout'])
        rest_env_ids = list(set(env_ids).union(self._waiting_env['step']))
        ready_env_ids = []
        cur_rest_env_ids = copy.deepcopy(rest_env_ids)
        while True:
            rest_conn = [self._pipe_parents[env_id] for env_id in cur_rest_env_ids]
            (ready_conn, ready_ids) = AsyncSubprocessEnvManager.wait(rest_conn, min(wait_num, len(rest_conn)), timeout)
            cur_ready_env_ids = [cur_rest_env_ids[env_id] for env_id in ready_ids]
            assert len(cur_ready_env_ids) == len(ready_conn)
            for (env_id, p) in zip(cur_ready_env_ids, ready_conn):
                try:
                    timesteps.update({env_id: p.recv()})
                except pickle.UnpicklingError as e:
                    timestep = BaseEnvTimestep(None, None, None, {'abnormal': True})
                    timesteps.update({env_id: timestep})
                    self._pipe_parents[env_id].close()
                    if self._subprocesses[env_id].is_alive():
                        self._subprocesses[env_id].terminate()
                    self._create_env_subprocess(env_id)
            self._check_data(timesteps)
            ready_env_ids += cur_ready_env_ids
            cur_rest_env_ids = list(set(cur_rest_env_ids).difference(set(cur_ready_env_ids)))
            if any([not t.done for t in timesteps.values()]) or len(ready_conn) == len(rest_conn):
                break
        self._waiting_env['step']: set
        for env_id in rest_env_ids:
            if env_id in ready_env_ids:
                if env_id in self._waiting_env['step']:
                    self._waiting_env['step'].remove(env_id)
            else:
                self._waiting_env['step'].add(env_id)
        if self._shared_memory:
            for (i, (env_id, timestep)) in enumerate(timesteps.items()):
                timesteps[env_id] = timestep._replace(obs=self._obs_buffers[env_id].get())
        for (env_id, timestep) in timesteps.items():
            if is_abnormal_timestep(timestep):
                self._env_states[env_id] = EnvState.ERROR
                continue
            if timestep.done:
                self._env_episode_count[env_id] += 1
                if self._env_episode_count[env_id] < self._episode_num:
                    if self._auto_reset:
                        if self._reset_inplace:
                            self._env_states[env_id] = EnvState.RUN
                            self._ready_obs[env_id] = timestep.obs
                        else:
                            self._env_states[env_id] = EnvState.RESET
                            reset_thread = PropagatingThread(target=self._reset, args=(env_id,), name='regular_reset')
                            reset_thread.daemon = True
                            reset_thread.start()
                    else:
                        self._env_states[env_id] = EnvState.NEED_RESET
                else:
                    self._env_states[env_id] = EnvState.DONE
            else:
                self._ready_obs[env_id] = timestep.obs
        return timesteps

    @staticmethod
    def worker_fn(p: connection.Connection, c: connection.Connection, env_fn_wrapper: 'CloudPickleWrapper', obs_buffer: ShmBuffer, method_name_list: list, reset_inplace: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Subprocess's target function to run.\n        "
        torch.set_num_threads(1)
        env_fn = env_fn_wrapper.data
        env = env_fn()
        p.close()
        try:
            while True:
                try:
                    (cmd, args, kwargs) = c.recv()
                except EOFError:
                    c.close()
                    break
                try:
                    if cmd == 'getattr':
                        ret = getattr(env, args[0])
                    elif cmd in method_name_list:
                        if cmd == 'step':
                            timestep = env.step(*args, **kwargs)
                            if is_abnormal_timestep(timestep):
                                ret = timestep
                            else:
                                if reset_inplace and timestep.done:
                                    obs = env.reset()
                                    timestep = timestep._replace(obs=obs)
                                if obs_buffer is not None:
                                    obs_buffer.fill(timestep.obs)
                                    timestep = timestep._replace(obs=None)
                                ret = timestep
                        elif cmd == 'reset':
                            ret = env.reset(*args, **kwargs)
                            if obs_buffer is not None:
                                obs_buffer.fill(ret)
                                ret = None
                        elif args is None and kwargs is None:
                            ret = getattr(env, cmd)()
                        else:
                            ret = getattr(env, cmd)(*args, **kwargs)
                    else:
                        raise KeyError('not support env cmd: {}'.format(cmd))
                    c.send(ret)
                except Exception as e:
                    logging.warning('subprocess exception traceback: \n' + traceback.format_exc())
                    c.send(e.__class__('\nEnv Process Exception:\n' + ''.join(traceback.format_tb(e.__traceback__)) + repr(e)))
                if cmd == 'close':
                    c.close()
                    break
        except KeyboardInterrupt:
            c.close()

    @staticmethod
    def worker_fn_robust(parent, child, env_fn_wrapper, obs_buffer, method_name_list, reset_timeout=None, step_timeout=None, reset_inplace=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            A more robust version of subprocess's target function to run. Used by default.\n        "
        torch.set_num_threads(1)
        env_fn = env_fn_wrapper.data
        env = env_fn()
        parent.close()

        @timeout_wrapper(timeout=step_timeout)
        def step_fn(*args, **kwargs):
            if False:
                return 10
            timestep = env.step(*args, **kwargs)
            if is_abnormal_timestep(timestep):
                ret = timestep
            else:
                if reset_inplace and timestep.done:
                    obs = env.reset()
                    timestep = timestep._replace(obs=obs)
                if obs_buffer is not None:
                    obs_buffer.fill(timestep.obs)
                    timestep = timestep._replace(obs=None)
                ret = timestep
            return ret

        @timeout_wrapper(timeout=reset_timeout)
        def reset_fn(*args, **kwargs):
            if False:
                print('Hello World!')
            try:
                ret = env.reset(*args, **kwargs)
                if obs_buffer is not None:
                    obs_buffer.fill(ret)
                    ret = None
                return ret
            except BaseException as e:
                logging.warning('subprocess exception traceback: \n' + traceback.format_exc())
                env.close()
                raise e
        while True:
            try:
                (cmd, args, kwargs) = child.recv()
            except EOFError:
                child.close()
                break
            try:
                if cmd == 'getattr':
                    ret = getattr(env, args[0])
                elif cmd in method_name_list:
                    if cmd == 'step':
                        ret = step_fn(*args)
                    elif cmd == 'reset':
                        if kwargs is None:
                            kwargs = {}
                        ret = reset_fn(*args, **kwargs)
                    elif cmd == 'render':
                        from ding.utils import render
                        ret = render(env, **kwargs)
                    elif args is None and kwargs is None:
                        ret = getattr(env, cmd)()
                    else:
                        ret = getattr(env, cmd)(*args, **kwargs)
                else:
                    raise KeyError('not support env cmd: {}'.format(cmd))
                child.send(ret)
            except BaseException as e:
                logging.debug("Sub env '{}' error when executing {}".format(str(env), cmd))
                logging.warning('subprocess exception traceback: \n' + traceback.format_exc())
                child.send(e.__class__('\nEnv Process Exception:\n' + ''.join(traceback.format_tb(e.__traceback__)) + repr(e)))
            if cmd == 'close':
                child.close()
                break

    def _check_data(self, data: Dict, close: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        exceptions = []
        for (i, d) in data.items():
            if isinstance(d, BaseException):
                self._env_states[i] = EnvState.ERROR
                exceptions.append(d)
        if len(exceptions) > 0:
            if close:
                self.close()
            raise exceptions[0]

    def __getattr__(self, key: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        self._check_closed()
        if not hasattr(self._env_ref, key):
            raise AttributeError("env `{}` doesn't have the attribute `{}`".format(type(self._env_ref), key))
        if isinstance(getattr(self._env_ref, key), MethodType) and key not in self.method_name_list:
            raise RuntimeError("env getattr doesn't supports method({}), please override method_name_list".format(key))
        for (_, p) in self._pipe_parents.items():
            p.send(['getattr', [key], {}])
        data = {i: p.recv() for (i, p) in self._pipe_parents.items()}
        self._check_data(data)
        ret = [data[i] for i in self._pipe_parents.keys()]
        return ret

    def enable_save_replay(self, replay_path: Union[List[str], str]) -> None:
        if False:
            return 10
        "\n        Overview:\n            Set each env's replay save path.\n        Arguments:\n            - replay_path (:obj:`Union[List[str], str]`): List of paths for each environment;                 Or one path for all environments.\n        "
        if isinstance(replay_path, str):
            replay_path = [replay_path] * self.env_num
        self._env_replay_path = replay_path

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            CLose the env manager and release all related resources.\n        '
        if self._closed:
            return
        self._closed = True
        for (_, p) in self._pipe_parents.items():
            p.send(['close', None, None])
        for (env_id, p) in self._pipe_parents.items():
            if not p.poll(5):
                continue
            p.recv()
        for i in range(self._env_num):
            self._env_states[i] = EnvState.VOID
        for (_, p) in self._subprocesses.items():
            p.terminate()
        for (_, p) in self._pipe_parents.items():
            p.close()

    @staticmethod
    def wait(rest_conn: list, wait_num: int, timeout: Optional[float]=None) -> Tuple[list, list]:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Wait at least enough(len(ready_conn) >= wait_num) connections within timeout constraint.\n            If timeout is None and wait_num == len(ready_conn), means sync mode;\n            If timeout is not None, will return when len(ready_conn) >= wait_num and\n            this method takes more than timeout seconds.\n        '
        assert 1 <= wait_num <= len(rest_conn), 'please indicate proper wait_num: <wait_num: {}, rest_conn_num: {}>'.format(wait_num, len(rest_conn))
        rest_conn_set = set(rest_conn)
        ready_conn = set()
        start_time = time.time()
        while len(rest_conn_set) > 0:
            if len(ready_conn) >= wait_num and timeout:
                if time.time() - start_time >= timeout:
                    break
            finish_conn = set(connection.wait(rest_conn_set, timeout=timeout))
            ready_conn = ready_conn.union(finish_conn)
            rest_conn_set = rest_conn_set.difference(finish_conn)
        ready_ids = [rest_conn.index(c) for c in ready_conn]
        return (list(ready_conn), ready_ids)

@ENV_MANAGER_REGISTRY.register('subprocess')
class SyncSubprocessEnvManager(AsyncSubprocessEnvManager):
    config = dict(episode_num=float('inf'), max_retry=5, step_timeout=None, auto_reset=True, reset_timeout=None, retry_type='reset', retry_waiting_time=0.1, shared_memory=True, copy_on_get=True, context='spawn' if platform.system().lower() == 'windows' else 'fork', wait_num=float('inf'), step_wait_timeout=None, connect_timeout=60, reset_inplace=False)

    def step(self, actions: Dict[int, Any]) -> Dict[int, namedtuple]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Step all environments. Reset an env if done.\n        Arguments:\n            - actions (:obj:`Dict[int, Any]`): {env_id: action}\n        Returns:\n            - timesteps (:obj:`Dict[int, namedtuple]`): {env_id: timestep}. Timestep is a                 ``BaseEnvTimestep`` tuple with observation, reward, done, env_info.\n        Example:\n            >>>     actions_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}\n            >>>     timesteps = env_manager.step(actions_dict):\n            >>>     for env_id, timestep in timesteps.items():\n            >>>         pass\n\n        .. note::\n\n            - The env_id that appears in ``actions`` will also be returned in ``timesteps``.\n            - Each environment is run by a subprocess separately. Once an environment is done, it is reset immediately.\n        '
        self._check_closed()
        env_ids = list(actions.keys())
        assert all([self._env_states[env_id] == EnvState.RUN for env_id in env_ids]), 'current env state are: {}, please check whether the requested env is in reset or done'.format({env_id: self._env_states[env_id] for env_id in env_ids})
        for (env_id, act) in actions.items():
            self._pipe_parents[env_id].send(['step', [act], None])
        timesteps = {}
        ready_conn = [self._pipe_parents[env_id] for env_id in env_ids]
        for (env_id, p) in zip(env_ids, ready_conn):
            try:
                timesteps.update({env_id: p.recv()})
            except pickle.UnpicklingError as e:
                timestep = BaseEnvTimestep(None, None, None, {'abnormal': True})
                timesteps.update({env_id: timestep})
                self._pipe_parents[env_id].close()
                if self._subprocesses[env_id].is_alive():
                    self._subprocesses[env_id].terminate()
                self._create_env_subprocess(env_id)
        self._check_data(timesteps)
        if self._shared_memory:
            for (i, (env_id, timestep)) in enumerate(timesteps.items()):
                timesteps[env_id] = timestep._replace(obs=self._obs_buffers[env_id].get())
        for (env_id, timestep) in timesteps.items():
            if is_abnormal_timestep(timestep):
                self._env_states[env_id] = EnvState.ERROR
                continue
            if timestep.done:
                self._env_episode_count[env_id] += 1
                if self._env_episode_count[env_id] < self._episode_num:
                    if self._auto_reset:
                        if self._reset_inplace:
                            self._env_states[env_id] = EnvState.RUN
                            self._ready_obs[env_id] = timestep.obs
                        else:
                            self._env_states[env_id] = EnvState.RESET
                            reset_thread = PropagatingThread(target=self._reset, args=(env_id,), name='regular_reset')
                            reset_thread.daemon = True
                            reset_thread.start()
                    else:
                        self._env_states[env_id] = EnvState.NEED_RESET
                else:
                    self._env_states[env_id] = EnvState.DONE
            else:
                self._ready_obs[env_id] = timestep.obs
        return timesteps

@ENV_MANAGER_REGISTRY.register('subprocess_v2')
class SubprocessEnvManagerV2(SyncSubprocessEnvManager):
    """
    Overview:
        SyncSubprocessEnvManager for new task pipeline and interfaces coupled with treetensor.
    """

    @property
    def ready_obs(self) -> tnp.array:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Get the ready (next) observation in ``tnp.array`` type, which is uniform for both async/sync scenarios.\n        Return:\n            - ready_obs (:obj:`tnp.array`): A stacked treenumpy-type observation data.\n        Example:\n            >>> obs = env_manager.ready_obs\n            >>> action = model(obs)  # model input np obs and output np action\n            >>> timesteps = env_manager.step(action)\n        '
        no_done_env_idx = [i for (i, s) in self._env_states.items() if s != EnvState.DONE]
        sleep_count = 0
        while not any([self._env_states[i] == EnvState.RUN for i in no_done_env_idx]):
            if sleep_count != 0 and sleep_count % 10000 == 0:
                logging.warning('VEC_ENV_MANAGER: all the not done envs are resetting, sleep {} times'.format(sleep_count))
            time.sleep(0.001)
            sleep_count += 1
        return tnp.stack([tnp.array(self._ready_obs[i]) for i in self.ready_env])

    def step(self, actions: Union[List[tnp.ndarray], tnp.ndarray]) -> List[tnp.ndarray]:
        if False:
            return 10
        '\n        Overview:\n            Execute env step according to input actions. And reset an env if done.\n        Arguments:\n            - actions (:obj:`Union[List[tnp.ndarray], tnp.ndarray]`): actions came from outer caller like policy.\n        Returns:\n            - timesteps (:obj:`List[tnp.ndarray]`): Each timestep is a tnp.array with observation, reward, done,                 info, env_id.\n        '
        if isinstance(actions, tnp.ndarray):
            split_action = tnp.split(actions, actions.shape[0])
            split_action = [s.squeeze(0) for s in split_action]
        else:
            split_action = actions
        actions = {env_id: a for (env_id, a) in zip(self.ready_obs_id, split_action)}
        timesteps = super().step(actions)
        new_data = []
        for (env_id, timestep) in timesteps.items():
            (obs, reward, done, info) = timestep
            info = make_key_as_identifier(info)
            info = remove_illegal_item(info)
            new_data.append(tnp.array({'obs': obs, 'reward': reward, 'done': done, 'info': info, 'env_id': env_id}))
        return new_data