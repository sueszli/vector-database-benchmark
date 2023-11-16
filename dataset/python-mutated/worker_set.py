import functools
import gymnasium as gym
import logging
import importlib.util
import os
from typing import Callable, Container, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, TypeVar, Union
import ray
from ray.actor import ActorHandle
from ray.exceptions import RayActorError
from ray.rllib.core.learner import LearnerGroup
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.utils.actor_manager import RemoteCallResults
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.offline import get_dataset_and_shards
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated, deprecation_warning, DEPRECATED_VALUE
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.typing import AgentID, EnvCreator, EnvType, EpisodeID, PartialAlgorithmConfigDict, PolicyID, SampleBatchType, TensorType
if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
(tf1, tf, tfv) = try_import_tf()
logger = logging.getLogger(__name__)
T = TypeVar('T')

def handle_remote_call_result_errors(results: RemoteCallResults, ignore_worker_failures: bool) -> None:
    if False:
        print('Hello World!')
    'Checks given results for application errors and raises them if necessary.\n\n    Args:\n        results: The results to check.\n    '
    for r in results.ignore_ray_errors():
        if r.ok:
            continue
        if ignore_worker_failures:
            logger.exception(r.get())
        else:
            raise r.get()

@DeveloperAPI
class WorkerSet:
    """Set of EnvRunners with n @ray.remote workers and zero or one local worker.

    Where: n >= 0.
    """

    def __init__(self, *, env_creator: Optional[EnvCreator]=None, validate_env: Optional[Callable[[EnvType], None]]=None, default_policy_class: Optional[Type[Policy]]=None, config: Optional['AlgorithmConfig']=None, num_workers: int=0, local_worker: bool=True, logdir: Optional[str]=None, _setup: bool=True):
        if False:
            print('Hello World!')
        "Initializes a WorkerSet instance.\n\n        Args:\n            env_creator: Function that returns env given env config.\n            validate_env: Optional callable to validate the generated\n                environment (only on worker=0). This callable should raise\n                an exception if the environment is invalid.\n            default_policy_class: An optional default Policy class to use inside\n                the (multi-agent) `policies` dict. In case the PolicySpecs in there\n                have no class defined, use this `default_policy_class`.\n                If None, PolicySpecs will be using the Algorithm's default Policy\n                class.\n            config: Optional AlgorithmConfig (or config dict).\n            num_workers: Number of remote rollout workers to create.\n            local_worker: Whether to create a local (non @ray.remote) worker\n                in the returned set as well (default: True). If `num_workers`\n                is 0, always create a local worker.\n            logdir: Optional logging directory for workers.\n            _setup: Whether to actually set up workers. This is only for testing.\n        "
        from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
        if not config:
            config = AlgorithmConfig()
        elif isinstance(config, dict):
            config = AlgorithmConfig.from_dict(config)
        self._env_creator = env_creator
        self._policy_class = default_policy_class
        self._remote_config = config
        self._remote_args = {'num_cpus': self._remote_config.num_cpus_per_worker, 'num_gpus': self._remote_config.num_gpus_per_worker, 'resources': self._remote_config.custom_resources_per_worker, 'max_restarts': config.max_num_worker_restarts}
        self.env_runner_cls = RolloutWorker if config.env_runner_cls is None else config.env_runner_cls
        self._cls = ray.remote(**self._remote_args)(self.env_runner_cls).remote
        self._logdir = logdir
        self._ignore_worker_failures = config['ignore_worker_failures']
        self.__worker_manager = FaultTolerantActorManager(max_remote_requests_in_flight_per_actor=config['max_requests_in_flight_per_sampler_worker'], init_id=1)
        if _setup:
            try:
                self._setup(validate_env=validate_env, config=config, num_workers=num_workers, local_worker=local_worker)
            except RayActorError as e:
                if e.actor_init_failed:
                    raise e.args[0].args[2]
                else:
                    raise e

    def _setup(self, *, validate_env: Optional[Callable[[EnvType], None]]=None, config: Optional['AlgorithmConfig']=None, num_workers: int=0, local_worker: bool=True):
        if False:
            while True:
                i = 10
        'Initializes a WorkerSet instance.\n        Args:\n            validate_env: Optional callable to validate the generated\n                environment (only on worker=0).\n            config: Optional dict that extends the common config of\n                the Algorithm class.\n            num_workers: Number of remote rollout workers to create.\n            local_worker: Whether to create a local (non @ray.remote) worker\n                in the returned set as well (default: True). If `num_workers`\n                is 0, always create a local worker.\n        '
        self._local_worker = None
        if num_workers == 0:
            local_worker = True
        local_tf_session_args = config.tf_session_args.copy()
        local_tf_session_args.update(config.local_tf_session_args)
        self._local_config = config.copy(copy_frozen=False).framework(tf_session_args=local_tf_session_args)
        if config.input_ == 'dataset':
            (self._ds, self._ds_shards) = get_dataset_and_shards(config, num_workers)
        else:
            self._ds = None
            self._ds_shards = None
        self.add_workers(num_workers, validate=config.validate_workers_after_construction)
        if local_worker and self.__worker_manager.num_actors() > 0 and (not config.create_env_on_local_worker) and (not config.observation_space or not config.action_space):
            spaces = self._get_spaces_from_remote_worker()
        else:
            spaces = None
        if local_worker:
            self._local_worker = self._make_worker(cls=self.env_runner_cls, env_creator=self._env_creator, validate_env=validate_env, worker_index=0, num_workers=num_workers, config=self._local_config, spaces=spaces)

    def _get_spaces_from_remote_worker(self):
        if False:
            i = 10
            return i + 15
        'Infer observation and action spaces from a remote worker.\n\n        Returns:\n            A dict mapping from policy ids to spaces.\n        '
        worker_id = self.__worker_manager.actor_ids()[0]
        if issubclass(self.env_runner_cls, RolloutWorker):
            remote_spaces = self.foreach_worker(lambda worker: worker.foreach_policy(lambda p, pid: (pid, p.observation_space, p.action_space)), remote_worker_ids=[worker_id], local_worker=False)
        else:
            remote_spaces = self.foreach_worker(lambda worker: worker.marl_module.foreach_module(lambda mid, m: (mid, m.config.observation_space, m.config.action_space)) if hasattr(worker, 'marl_module') else [(DEFAULT_POLICY_ID, worker.module.config.observation_space, worker.module.config.action_space)])
        if not remote_spaces:
            raise ValueError('Could not get observation and action spaces from remote worker. Maybe specify them manually in the config?')
        spaces = {e[0]: (getattr(e[1], 'original_space', e[1]), e[2]) for e in remote_spaces[0]}
        if issubclass(self.env_runner_cls, RolloutWorker):
            env_spaces = self.foreach_worker(lambda worker: worker.foreach_env(lambda env: (env.observation_space, env.action_space)), remote_worker_ids=[worker_id], local_worker=False)
            if env_spaces:
                spaces['__env__'] = env_spaces[0][0]
        logger.info(f'Inferred observation/action spaces from remote worker (local worker has no env): {spaces}')
        return spaces

    @DeveloperAPI
    def local_worker(self) -> EnvRunner:
        if False:
            return 10
        'Returns the local rollout worker.'
        return self._local_worker

    @DeveloperAPI
    def healthy_worker_ids(self) -> List[int]:
        if False:
            while True:
                i = 10
        'Returns the list of remote worker IDs.'
        return self.__worker_manager.healthy_actor_ids()

    @DeveloperAPI
    def num_remote_workers(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Returns the number of remote rollout workers.'
        return self.__worker_manager.num_actors()

    @DeveloperAPI
    def num_healthy_remote_workers(self) -> int:
        if False:
            print('Hello World!')
        'Returns the number of healthy remote workers.'
        return self.__worker_manager.num_healthy_actors()

    @DeveloperAPI
    def num_healthy_workers(self) -> int:
        if False:
            return 10
        'Returns the number of all healthy workers, including the local worker.'
        return int(bool(self._local_worker)) + self.num_healthy_remote_workers()

    @DeveloperAPI
    def num_in_flight_async_reqs(self) -> int:
        if False:
            i = 10
            return i + 15
        'Returns the number of in-flight async requests.'
        return self.__worker_manager.num_outstanding_async_reqs()

    @DeveloperAPI
    def num_remote_worker_restarts(self) -> int:
        if False:
            while True:
                i = 10
        'Total number of times managed remote workers have been restarted.'
        return self.__worker_manager.total_num_restarts()

    @DeveloperAPI
    def sync_weights(self, policies: Optional[List[PolicyID]]=None, from_worker_or_learner_group: Optional[Union[EnvRunner, LearnerGroup]]=None, to_worker_indices: Optional[List[int]]=None, global_vars: Optional[Dict[str, TensorType]]=None, timeout_seconds: Optional[int]=0) -> None:
        if False:
            print('Hello World!')
        "Syncs model weights from the given weight source to all remote workers.\n\n        Weight source can be either a (local) rollout worker or a learner_group. It\n        should just implement a `get_weights` method.\n\n        Args:\n            policies: Optional list of PolicyIDs to sync weights for.\n                If None (default), sync weights to/from all policies.\n            from_worker_or_learner_group: Optional (local) EnvRunner instance or\n                LearnerGroup instance to sync from. If None (default),\n                sync from this WorkerSet's local worker.\n            to_worker_indices: Optional list of worker indices to sync the\n                weights to. If None (default), sync to all remote workers.\n            global_vars: An optional global vars dict to set this\n                worker to. If None, do not update the global_vars.\n            timeout_seconds: Timeout in seconds to wait for the sync weights\n                calls to complete. Default is 0 (sync-and-forget, do not wait\n                for any sync calls to finish). This significantly improves\n                algorithm performance.\n        "
        if self.local_worker() is None and from_worker_or_learner_group is None:
            raise TypeError('No `local_worker` in WorkerSet, must provide `from_worker_or_learner_group` arg in `sync_weights()`!')
        weights = None
        if self.num_remote_workers() or from_worker_or_learner_group is not None:
            weights_src = from_worker_or_learner_group or self.local_worker()
            if weights_src is None:
                raise ValueError('`from_worker_or_trainer` is None. In this case, workerset should have local_worker. But local_worker is also None.')
            weights = weights_src.get_weights(policies)

            def set_weight(w):
                if False:
                    return 10
                w.set_weights(weights, global_vars)
            self.foreach_worker(func=set_weight, local_worker=False, remote_worker_ids=to_worker_indices, healthy_only=True, timeout_seconds=timeout_seconds)
        if self.local_worker() is not None:
            if from_worker_or_learner_group is not None:
                self.local_worker().set_weights(weights, global_vars=global_vars)
            elif global_vars is not None:
                self.local_worker().set_global_vars(global_vars)

    @DeveloperAPI
    def add_policy(self, policy_id: PolicyID, policy_cls: Optional[Type[Policy]]=None, policy: Optional[Policy]=None, *, observation_space: Optional[gym.spaces.Space]=None, action_space: Optional[gym.spaces.Space]=None, config: Optional[Union['AlgorithmConfig', PartialAlgorithmConfigDict]]=None, policy_state: Optional[PolicyState]=None, policy_mapping_fn: Optional[Callable[[AgentID, EpisodeID], PolicyID]]=None, policies_to_train: Optional[Union[Container[PolicyID], Callable[[PolicyID, Optional[SampleBatchType]], bool]]]=None, module_spec: Optional[SingleAgentRLModuleSpec]=None, workers: Optional[List[Union[EnvRunner, ActorHandle]]]=DEPRECATED_VALUE) -> None:
        if False:
            print('Hello World!')
        "Adds a policy to this WorkerSet's workers or a specific list of workers.\n\n        Args:\n            policy_id: ID of the policy to add.\n            policy_cls: The Policy class to use for constructing the new Policy.\n                Note: Only one of `policy_cls` or `policy` must be provided.\n            policy: The Policy instance to add to this WorkerSet. If not None, the\n                given Policy object will be directly inserted into the\n                local worker and clones of that Policy will be created on all remote\n                workers.\n                Note: Only one of `policy_cls` or `policy` must be provided.\n            observation_space: The observation space of the policy to add.\n                If None, try to infer this space from the environment.\n            action_space: The action space of the policy to add.\n                If None, try to infer this space from the environment.\n            config: The config object or overrides for the policy to add.\n            policy_state: Optional state dict to apply to the new\n                policy instance, right after its construction.\n            policy_mapping_fn: An optional (updated) policy mapping function\n                to use from here on. Note that already ongoing episodes will\n                not change their mapping but will use the old mapping till\n                the end of the episode.\n            policies_to_train: An optional list of policy IDs to be trained\n                or a callable taking PolicyID and SampleBatchType and\n                returning a bool (trainable or not?).\n                If None, will keep the existing setup in place. Policies,\n                whose IDs are not in the list (or for which the callable\n                returns False) will not be updated.\n            module_spec: In the new RLModule API we need to pass in the module_spec for\n                the new module that is supposed to be added. Knowing the policy spec is\n                not sufficient.\n            workers: A list of EnvRunner/ActorHandles (remote\n                EnvRunners) to add this policy to. If defined, will only\n                add the given policy to these workers.\n\n        Raises:\n            KeyError: If the given `policy_id` already exists in this WorkerSet.\n        "
        if self.local_worker() and policy_id in self.local_worker().policy_map:
            raise KeyError(f"Policy ID '{policy_id}' already exists in policy map! Make sure you use a Policy ID that has not been taken yet. Policy IDs that are already in your policy map: {list(self.local_worker().policy_map.keys())}")
        if workers is not DEPRECATED_VALUE:
            deprecation_warning(old='WorkerSet.add_policy(.., workers=..)', help='The `workers` argument to `WorkerSet.add_policy()` is deprecated! Please do not use it anymore.', error=True)
        if (policy_cls is None) == (policy is None):
            raise ValueError('Only one of `policy_cls` or `policy` must be provided to staticmethod: `WorkerSet.add_policy()`!')
        validate_policy_id(policy_id, error=False)
        if policy_cls is not None:
            new_policy_instance_kwargs = dict(policy_id=policy_id, policy_cls=policy_cls, observation_space=observation_space, action_space=action_space, config=config, policy_state=policy_state, policy_mapping_fn=policy_mapping_fn, policies_to_train=list(policies_to_train) if policies_to_train else None, module_spec=module_spec)
        else:
            new_policy_instance_kwargs = dict(policy_id=policy_id, policy_cls=type(policy), observation_space=policy.observation_space, action_space=policy.action_space, config=policy.config, policy_state=policy.get_state(), policy_mapping_fn=policy_mapping_fn, policies_to_train=list(policies_to_train) if policies_to_train else None, module_spec=module_spec)

        def _create_new_policy_fn(worker):
            if False:
                print('Hello World!')
            worker.add_policy(**new_policy_instance_kwargs)
        if self.local_worker() is not None:
            if policy is not None:
                self.local_worker().add_policy(policy_id=policy_id, policy=policy, policy_mapping_fn=policy_mapping_fn, policies_to_train=policies_to_train, module_spec=module_spec)
            else:
                self.local_worker().add_policy(**new_policy_instance_kwargs)
        self.foreach_worker(_create_new_policy_fn, local_worker=False)

    @DeveloperAPI
    def add_workers(self, num_workers: int, validate: bool=False) -> None:
        if False:
            return 10
        'Creates and adds a number of remote workers to this worker set.\n\n        Can be called several times on the same WorkerSet to add more\n        EnvRunners to the set.\n\n        Args:\n            num_workers: The number of remote Workers to add to this\n                WorkerSet.\n            validate: Whether to validate remote workers after their construction\n                process.\n\n        Raises:\n            RayError: If any of the constructed remote workers is not up and running\n            properly.\n        '
        old_num_workers = self.__worker_manager.num_actors()
        new_workers = [self._make_worker(cls=self._cls, env_creator=self._env_creator, validate_env=None, worker_index=old_num_workers + i + 1, num_workers=old_num_workers + num_workers, config=self._remote_config) for i in range(num_workers)]
        self.__worker_manager.add_actors(new_workers)
        if validate:
            for result in self.__worker_manager.foreach_actor(lambda w: w.assert_healthy()):
                if not result.ok:
                    raise result.get()

    @DeveloperAPI
    def reset(self, new_remote_workers: List[ActorHandle]) -> None:
        if False:
            while True:
                i = 10
        'Hard overrides the remote workers in this set with the given one.\n\n        Args:\n            new_remote_workers: A list of new EnvRunners\n                (as `ActorHandles`) to use as remote workers.\n        '
        self.__worker_manager.clear()
        self.__worker_manager.add_actors(new_remote_workers)

    @DeveloperAPI
    def stop(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Calls `stop` on all rollout workers (including the local one).'
        try:
            self.foreach_worker(lambda w: w.stop(), healthy_only=False, local_worker=True)
        except Exception:
            logger.exception('Failed to stop workers!')
        finally:
            self.__worker_manager.clear()

    @DeveloperAPI
    def is_policy_to_train(self, policy_id: PolicyID, batch: Optional[SampleBatchType]=None) -> bool:
        if False:
            return 10
        'Whether given PolicyID (optionally inside some batch) is trainable.'
        local_worker = self.local_worker()
        if local_worker:
            if local_worker.is_policy_to_train is None:
                return True
            return local_worker.is_policy_to_train(policy_id, batch)
        else:
            raise NotImplementedError

    @DeveloperAPI
    def foreach_worker(self, func: Callable[[EnvRunner], T], *, local_worker: bool=True, healthy_only: bool=False, remote_worker_ids: List[int]=None, timeout_seconds: Optional[int]=None, return_obj_refs: bool=False, mark_healthy: bool=False) -> List[T]:
        if False:
            print('Hello World!')
        'Calls the given function with each worker instance as the argument.\n\n        Args:\n            func: The function to call for each worker (as only arg).\n            local_worker: Whether apply `func` on local worker too. Default is True.\n            healthy_only: Apply `func` on known-to-be healthy workers only.\n            remote_worker_ids: Apply `func` on a selected set of remote workers.\n            timeout_seconds: Time to wait for results. Default is None.\n            return_obj_refs: whether to return ObjectRef instead of actual results.\n                Note, for fault tolerance reasons, these returned ObjectRefs should\n                never be resolved with ray.get() outside of this WorkerSet.\n            mark_healthy: Whether to mark the worker as healthy based on call results.\n\n        Returns:\n             The list of return values of all calls to `func([worker])`.\n        '
        assert not return_obj_refs or not local_worker, 'Can not return ObjectRef from local worker.'
        local_result = []
        if local_worker and self.local_worker() is not None:
            local_result = [func(self.local_worker())]
        remote_results = self.__worker_manager.foreach_actor(func, healthy_only=healthy_only, remote_actor_ids=remote_worker_ids, timeout_seconds=timeout_seconds, return_obj_refs=return_obj_refs, mark_healthy=mark_healthy)
        handle_remote_call_result_errors(remote_results, self._ignore_worker_failures)
        remote_results = [r.get() for r in remote_results.ignore_errors()]
        return local_result + remote_results

    @DeveloperAPI
    def foreach_worker_with_id(self, func: Callable[[int, EnvRunner], T], *, local_worker: bool=True, healthy_only: bool=False, remote_worker_ids: List[int]=None, timeout_seconds: Optional[int]=None) -> List[T]:
        if False:
            while True:
                i = 10
        'Similar to foreach_worker(), but calls the function with id of the worker too.\n\n        Args:\n            func: The function to call for each worker (as only arg).\n            local_worker: Whether apply `func` on local worker too. Default is True.\n            healthy_only: Apply `func` on known-to-be healthy workers only.\n            remote_worker_ids: Apply `func` on a selected set of remote workers.\n            timeout_seconds: Time to wait for results. Default is None.\n\n        Returns:\n             The list of return values of all calls to `func([worker, id])`.\n        '
        local_result = []
        if local_worker and self.local_worker() is not None:
            local_result = [func(0, self.local_worker())]
        if not remote_worker_ids:
            remote_worker_ids = self.__worker_manager.actor_ids()
        funcs = [functools.partial(func, i) for i in remote_worker_ids]
        remote_results = self.__worker_manager.foreach_actor(funcs, healthy_only=healthy_only, remote_actor_ids=remote_worker_ids, timeout_seconds=timeout_seconds)
        handle_remote_call_result_errors(remote_results, self._ignore_worker_failures)
        remote_results = [r.get() for r in remote_results.ignore_errors()]
        return local_result + remote_results

    @DeveloperAPI
    def foreach_worker_async(self, func: Callable[[EnvRunner], T], *, healthy_only: bool=False, remote_worker_ids: List[int]=None) -> int:
        if False:
            i = 10
            return i + 15
        'Calls the given function asynchronously with each worker as the argument.\n\n        foreach_worker_async() does not return results directly. Instead,\n        fetch_ready_async_reqs() can be used to pull results in an async manner\n        whenever they are available.\n\n        Args:\n            func: The function to call for each worker (as only arg).\n            healthy_only: Apply `func` on known-to-be healthy workers only.\n            remote_worker_ids: Apply `func` on a selected set of remote workers.\n\n        Returns:\n             The number of async requests that are currently in-flight.\n        '
        return self.__worker_manager.foreach_actor_async(func, healthy_only=healthy_only, remote_actor_ids=remote_worker_ids)

    @DeveloperAPI
    def fetch_ready_async_reqs(self, *, timeout_seconds: Optional[int]=0, return_obj_refs: bool=False, mark_healthy: bool=False) -> List[Tuple[int, T]]:
        if False:
            return 10
        'Get esults from outstanding asynchronous requests that are ready.\n\n        Args:\n            timeout_seconds: Time to wait for results. Default is 0, meaning\n                those requests that are already ready.\n            return_obj_refs: Whether to return ObjectRef instead of actual results.\n            mark_healthy: Whether to mark the worker as healthy based on call results.\n\n        Returns:\n            A list of results successfully returned from outstanding remote calls,\n            paired with the indices of the callee workers.\n        '
        remote_results = self.__worker_manager.fetch_ready_async_reqs(timeout_seconds=timeout_seconds, return_obj_refs=return_obj_refs, mark_healthy=mark_healthy)
        handle_remote_call_result_errors(remote_results, self._ignore_worker_failures)
        return [(r.actor_id, r.get()) for r in remote_results.ignore_errors()]

    @DeveloperAPI
    def foreach_policy(self, func: Callable[[Policy, PolicyID], T]) -> List[T]:
        if False:
            while True:
                i = 10
        "Calls `func` with each worker's (policy, PolicyID) tuple.\n\n        Note that in the multi-agent case, each worker may have more than one\n        policy.\n\n        Args:\n            func: A function - taking a Policy and its ID - that is\n                called on all workers' Policies.\n\n        Returns:\n            The list of return values of func over all workers' policies. The\n                length of this list is:\n                (num_workers + 1 (local-worker)) *\n                [num policies in the multi-agent config dict].\n                The local workers' results are first, followed by all remote\n                workers' results\n        "
        results = []
        for r in self.foreach_worker(lambda w: w.foreach_policy(func), local_worker=True):
            results.extend(r)
        return results

    @DeveloperAPI
    def foreach_policy_to_train(self, func: Callable[[Policy, PolicyID], T]) -> List[T]:
        if False:
            print('Hello World!')
        "Apply `func` to all workers' Policies iff in `policies_to_train`.\n\n        Args:\n            func: A function - taking a Policy and its ID - that is\n                called on all workers' Policies, for which\n                `worker.is_policy_to_train()` returns True.\n\n        Returns:\n            List[any]: The list of n return values of all\n                `func([trainable policy], [ID])`-calls.\n        "
        results = []
        for r in self.foreach_worker(lambda w: w.foreach_policy_to_train(func), local_worker=True):
            results.extend(r)
        return results

    @DeveloperAPI
    def foreach_env(self, func: Callable[[EnvType], List[T]]) -> List[List[T]]:
        if False:
            for i in range(10):
                print('nop')
        'Calls `func` with all workers\' sub-environments as args.\n\n        An "underlying sub environment" is a single clone of an env within\n        a vectorized environment.\n        `func` takes a single underlying sub environment as arg, e.g. a\n        gym.Env object.\n\n        Args:\n            func: A function - taking an EnvType (normally a gym.Env object)\n                as arg and returning a list of lists of return values, one\n                value per underlying sub-environment per each worker.\n\n        Returns:\n            The list (workers) of lists (sub environments) of results.\n        '
        return list(self.foreach_worker(lambda w: w.foreach_env(func), local_worker=True))

    @DeveloperAPI
    def foreach_env_with_context(self, func: Callable[[BaseEnv, EnvContext], List[T]]) -> List[List[T]]:
        if False:
            return 10
        'Calls `func` with all workers\' sub-environments and env_ctx as args.\n\n        An "underlying sub environment" is a single clone of an env within\n        a vectorized environment.\n        `func` takes a single underlying sub environment and the env_context\n        as args.\n\n        Args:\n            func: A function - taking a BaseEnv object and an EnvContext as\n                arg - and returning a list of lists of return values over envs\n                of the worker.\n\n        Returns:\n            The list (1 item per workers) of lists (1 item per sub-environment)\n                of results.\n        '
        return list(self.foreach_worker(lambda w: w.foreach_env_with_context(func), local_worker=True))

    @DeveloperAPI
    def probe_unhealthy_workers(self) -> List[int]:
        if False:
            return 10
        'Checks for unhealthy workers and tries restoring their states.\n\n        Returns:\n            List of IDs of the workers that were restored.\n        '
        return self.__worker_manager.probe_unhealthy_actors(timeout_seconds=self._remote_config.worker_health_probe_timeout_s)

    @staticmethod
    def _from_existing(local_worker: EnvRunner, remote_workers: List[ActorHandle]=None):
        if False:
            print('Hello World!')
        workers = WorkerSet(env_creator=None, default_policy_class=None, config=None, _setup=False)
        workers.reset(remote_workers or [])
        workers._local_worker = local_worker
        return workers

    def _make_worker(self, *, cls: Callable, env_creator: EnvCreator, validate_env: Optional[Callable[[EnvType], None]], worker_index: int, num_workers: int, recreated_worker: bool=False, config: 'AlgorithmConfig', spaces: Optional[Dict[PolicyID, Tuple[gym.spaces.Space, gym.spaces.Space]]]=None) -> Union[EnvRunner, ActorHandle]:
        if False:
            i = 10
            return i + 15
        worker = cls(env_creator=env_creator, validate_env=validate_env, default_policy_class=self._policy_class, config=config, worker_index=worker_index, num_workers=num_workers, recreated_worker=recreated_worker, log_dir=self._logdir, spaces=spaces, dataset_shards=self._ds_shards)
        return worker

    @classmethod
    def _valid_module(cls, class_path):
        if False:
            while True:
                i = 10
        del cls
        if isinstance(class_path, str) and (not os.path.isfile(class_path)) and ('.' in class_path):
            (module_path, class_name) = class_path.rsplit('.', 1)
            try:
                spec = importlib.util.find_spec(module_path)
                if spec is not None:
                    return True
            except (ModuleNotFoundError, ValueError):
                print(f'module {module_path} not found while trying to get input {class_path}')
        return False

    @Deprecated(new='WorkerSet.foreach_policy_to_train', error=True)
    def foreach_trainable_policy(self, func):
        if False:
            for i in range(10):
                print('nop')
        pass

    @property
    @Deprecated(old='_remote_workers', new='Use either the `foreach_worker()`, `foreach_worker_with_id()`, or `foreach_worker_async()` APIs of `WorkerSet`, which all handle fault tolerance.', error=False)
    def _remote_workers(self) -> List[ActorHandle]:
        if False:
            return 10
        return list(self.__worker_manager.actors().values())

    @Deprecated(old='remote_workers()', new='Use either the `foreach_worker()`, `foreach_worker_with_id()`, or `foreach_worker_async()` APIs of `WorkerSet`, which all handle fault tolerance.', error=False)
    def remote_workers(self) -> List[ActorHandle]:
        if False:
            i = 10
            return i + 15
        return list(self.__worker_manager.actors().values())