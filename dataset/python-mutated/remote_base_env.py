import gymnasium as gym
import logging
from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import ray
from ray.util import log_once
from ray.rllib.env.base_env import BaseEnv, _DUMMY_AGENT_ID, ASYNC_RESET_RETURN
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvType, MultiEnvDict
if TYPE_CHECKING:
    from ray.rllib.evaluation.rollout_worker import RolloutWorker
logger = logging.getLogger(__name__)

@PublicAPI
class RemoteBaseEnv(BaseEnv):
    """BaseEnv that executes its sub environments as @ray.remote actors.

    This provides dynamic batching of inference as observations are returned
    from the remote simulator actors. Both single and multi-agent child envs
    are supported, and envs can be stepped synchronously or asynchronously.

    NOTE: This class implicitly assumes that the remote envs are gym.Env's

    You shouldn't need to instantiate this class directly. It's automatically
    inserted when you use the `remote_worker_envs=True` option in your
    Algorithm's config.
    """

    def __init__(self, make_env: Callable[[int], EnvType], num_envs: int, multiagent: bool, remote_env_batch_wait_ms: int, existing_envs: Optional[List[ray.actor.ActorHandle]]=None, worker: Optional['RolloutWorker']=None, restart_failed_sub_environments: bool=False):
        if False:
            return 10
        'Initializes a RemoteVectorEnv instance.\n\n        Args:\n            make_env: Callable that produces a single (non-vectorized) env,\n                given the vector env index as only arg.\n            num_envs: The number of sub-environments to create for the\n                vectorization.\n            multiagent: Whether this is a multiagent env or not.\n            remote_env_batch_wait_ms: Time to wait for (ray.remote)\n                sub-environments to have new observations available when\n                polled. Only when none of the sub-environments is ready,\n                repeat the `ray.wait()` call until at least one sub-env\n                is ready. Then return only the observations of the ready\n                sub-environment(s).\n            existing_envs: Optional list of already created sub-environments.\n                These will be used as-is and only as many new sub-envs as\n                necessary (`num_envs - len(existing_envs)`) will be created.\n            worker: An optional RolloutWorker that owns the env. This is only\n                used if `remote_worker_envs` is True in your config and the\n                `on_sub_environment_created` custom callback needs to be\n                called on each created actor.\n            restart_failed_sub_environments: If True and any sub-environment (within\n                a vectorized env) throws any error during env stepping, the\n                Sampler will try to restart the faulty sub-environment. This is done\n                without disturbing the other (still intact) sub-environment and without\n                the RolloutWorker crashing.\n        '
        self.make_env = make_env
        self.num_envs = num_envs
        self.multiagent = multiagent
        self.poll_timeout = remote_env_batch_wait_ms / 1000
        self.worker = worker
        self.restart_failed_sub_environments = restart_failed_sub_environments
        existing_envs = existing_envs or []
        self.make_env_creates_actors = False
        self._observation_space = None
        self._action_space = None
        self.actors: Optional[List[ray.actor.ActorHandle]] = None
        if len(existing_envs) > 0 and isinstance(existing_envs[0], ray.actor.ActorHandle):
            self.make_env_creates_actors = True
            self.actors = existing_envs
            while len(self.actors) < self.num_envs:
                self.actors.append(self._make_sub_env(len(self.actors)))
        else:
            self.actors = [self._make_sub_env(i) for i in range(self.num_envs)]
            if len(existing_envs) > 0:
                self._observation_space = existing_envs[0].observation_space
                self._action_space = existing_envs[0].action_space
            else:
                (self._observation_space, self._action_space) = ray.get([self.actors[0].observation_space.remote(), self.actors[0].action_space.remote()])
        self.pending: Dict[ray.actor.ActorHandle] = {a.reset.remote(): a for a in self.actors}

    @override(BaseEnv)
    def poll(self) -> Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict]:
        if False:
            while True:
                i = 10
        (obs, rewards, terminateds, truncateds, infos) = ({}, {}, {}, {}, {})
        ready = []
        while not ready:
            (ready, _) = ray.wait(list(self.pending), num_returns=len(self.pending), timeout=self.poll_timeout)
        env_ids = set()
        for obj_ref in ready:
            actor = self.pending.pop(obj_ref)
            env_id = self.actors.index(actor)
            env_ids.add(env_id)
            try:
                ret = ray.get(obj_ref)
            except Exception as e:
                if self.restart_failed_sub_environments:
                    logger.exception(e.args[0])
                    self.try_restart(env_id)
                    ret = (e, {}, {'__all__': True}, {'__all__': False}, {})
                else:
                    raise e
            if self.make_env_creates_actors:
                (rew, terminated, truncated, info) = (None, None, None, None)
                if self.multiagent:
                    if isinstance(ret, tuple):
                        if len(ret) == 5:
                            (ob, rew, terminated, truncated, info) = ret
                        elif len(ret) == 2:
                            ob = ret[0]
                            info = ret[1]
                        else:
                            raise AssertionError('Your gymnasium.Env seems to NOT return the correct number of return values for `step()` (needs to return 5 values: obs, reward, terminated, truncated and info) or `reset()` (needs to return 2 values: obs and info)!')
                    else:
                        raise AssertionError('Your gymnasium.Env seems to only return a single value upon `reset()`! Must return 2 (obs AND infos).')
                elif isinstance(ret, tuple):
                    if len(ret) == 5:
                        ob = {_DUMMY_AGENT_ID: ret[0]}
                        rew = {_DUMMY_AGENT_ID: ret[1]}
                        terminated = {_DUMMY_AGENT_ID: ret[2], '__all__': ret[2]}
                        truncated = {_DUMMY_AGENT_ID: ret[3], '__all__': ret[3]}
                        info = {_DUMMY_AGENT_ID: ret[4]}
                    elif len(ret) == 2:
                        ob = {_DUMMY_AGENT_ID: ret[0]}
                        info = {_DUMMY_AGENT_ID: ret[1]}
                    else:
                        raise AssertionError('Your gymnasium.Env seems to NOT return the correct number of return values for `step()` (needs to return 5 values: obs, reward, terminated, truncated and info) or `reset()` (needs to return 2 values: obs and info)!')
                else:
                    raise AssertionError('Your gymnasium.Env seems to only return a single value upon `reset()`! Must return 2 (obs and infos).')
                if rew is None:
                    rew = {agent_id: 0 for agent_id in ob.keys()}
                    terminated = {'__all__': False}
                    truncated = {'__all__': False}
            else:
                (ob, rew, terminated, truncated, info) = ret
            obs[env_id] = ob
            rewards[env_id] = rew
            terminateds[env_id] = terminated
            truncateds[env_id] = truncated
            infos[env_id] = info
        logger.debug(f'Got obs batch for actors {env_ids}')
        return (obs, rewards, terminateds, truncateds, infos, {})

    @override(BaseEnv)
    @PublicAPI
    def send_actions(self, action_dict: MultiEnvDict) -> None:
        if False:
            for i in range(10):
                print('nop')
        for (env_id, actions) in action_dict.items():
            actor = self.actors[env_id]
            if not self.multiagent and self.make_env_creates_actors:
                obj_ref = actor.step.remote(actions[_DUMMY_AGENT_ID])
            else:
                obj_ref = actor.step.remote(actions)
            self.pending[obj_ref] = actor

    @override(BaseEnv)
    @PublicAPI
    def try_reset(self, env_id: Optional[EnvID]=None, *, seed: Optional[int]=None, options: Optional[dict]=None) -> Tuple[MultiEnvDict, MultiEnvDict]:
        if False:
            return 10
        actor = self.actors[env_id]
        obj_ref = actor.reset.remote(seed=seed, options=options)
        self.pending[obj_ref] = actor
        return (ASYNC_RESET_RETURN, ASYNC_RESET_RETURN)

    @override(BaseEnv)
    def try_restart(self, env_id: Optional[EnvID]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        try:
            self.actors[env_id].close.remote()
        except Exception as e:
            if log_once('close_sub_env'):
                logger.warning(f'Trying to close old and replaced sub-environment (at vector index={env_id}), but closing resulted in error:\n{e}')
        self.actors[env_id].__ray_terminate__.remote()
        self.actors[env_id] = self._make_sub_env(env_id)

    @override(BaseEnv)
    @PublicAPI
    def stop(self) -> None:
        if False:
            print('Hello World!')
        if self.actors is not None:
            for actor in self.actors:
                actor.__ray_terminate__.remote()

    @override(BaseEnv)
    @PublicAPI
    def get_sub_environments(self, as_dict: bool=False) -> List[EnvType]:
        if False:
            for i in range(10):
                print('nop')
        if as_dict:
            return {env_id: actor for (env_id, actor) in enumerate(self.actors)}
        return self.actors

    @property
    @override(BaseEnv)
    @PublicAPI
    def observation_space(self) -> gym.spaces.Dict:
        if False:
            print('Hello World!')
        return self._observation_space

    @property
    @override(BaseEnv)
    @PublicAPI
    def action_space(self) -> gym.Space:
        if False:
            for i in range(10):
                print('nop')
        return self._action_space

    def _make_sub_env(self, idx: Optional[int]=None):
        if False:
            while True:
                i = 10
        'Re-creates a sub-environment at the new index.'
        if self.make_env_creates_actors:
            sub_env = self.make_env(idx)
            if self.worker is not None:
                self.worker.callbacks.on_sub_environment_created(worker=self.worker, sub_environment=self.actors[idx], env_context=self.worker.env_context.copy_with_overrides(vector_index=idx))
        else:

            def make_remote_env(i):
                if False:
                    for i in range(10):
                        print('nop')
                logger.info('Launching env {} in remote actor'.format(i))
                if self.multiagent:
                    sub_env = _RemoteMultiAgentEnv.remote(self.make_env, i)
                else:
                    sub_env = _RemoteSingleAgentEnv.remote(self.make_env, i)
                if self.worker is not None:
                    self.worker.callbacks.on_sub_environment_created(worker=self.worker, sub_environment=sub_env, env_context=self.worker.env_context.copy_with_overrides(vector_index=i))
                return sub_env
            sub_env = make_remote_env(idx)
        return sub_env

    @override(BaseEnv)
    def get_agent_ids(self) -> Set[AgentID]:
        if False:
            print('Hello World!')
        if self.multiagent:
            return ray.get(self.actors[0].get_agent_ids.remote())
        else:
            return {_DUMMY_AGENT_ID}

@ray.remote(num_cpus=0)
class _RemoteMultiAgentEnv:
    """Wrapper class for making a multi-agent env a remote actor."""

    def __init__(self, make_env, i):
        if False:
            return 10
        self.env = make_env(i)
        self.agent_ids = set()

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        if False:
            for i in range(10):
                print('nop')
        (obs, info) = self.env.reset(seed=seed, options=options)
        rew = {}
        for agent_id in obs.keys():
            self.agent_ids.add(agent_id)
            rew[agent_id] = 0.0
        terminated = {'__all__': False}
        truncated = {'__all__': False}
        return (obs, rew, terminated, truncated, info)

    def step(self, action_dict):
        if False:
            i = 10
            return i + 15
        return self.env.step(action_dict)

    def observation_space(self):
        if False:
            while True:
                i = 10
        return self.env.observation_space

    def action_space(self):
        if False:
            while True:
                i = 10
        return self.env.action_space

    def get_agent_ids(self) -> Set[AgentID]:
        if False:
            while True:
                i = 10
        return self.agent_ids

@ray.remote(num_cpus=0)
class _RemoteSingleAgentEnv:
    """Wrapper class for making a gym env a remote actor."""

    def __init__(self, make_env, i):
        if False:
            i = 10
            return i + 15
        self.env = make_env(i)

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        if False:
            return 10
        obs_and_info = self.env.reset(seed=seed, options=options)
        obs = {_DUMMY_AGENT_ID: obs_and_info[0]}
        info = {_DUMMY_AGENT_ID: obs_and_info[1]}
        rew = {_DUMMY_AGENT_ID: 0.0}
        terminated = {'__all__': False}
        truncated = {'__all__': False}
        return (obs, rew, terminated, truncated, info)

    def step(self, action):
        if False:
            for i in range(10):
                print('nop')
        results = self.env.step(action[_DUMMY_AGENT_ID])
        (obs, rew, terminated, truncated, info) = [{_DUMMY_AGENT_ID: x} for x in results]
        terminated['__all__'] = terminated[_DUMMY_AGENT_ID]
        truncated['__all__'] = truncated[_DUMMY_AGENT_ID]
        return (obs, rew, terminated, truncated, info)

    def observation_space(self):
        if False:
            return 10
        return self.env.observation_space

    def action_space(self):
        if False:
            return 10
        return self.env.action_space