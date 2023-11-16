import copy
import importlib.util
import logging
import os
import platform
import threading
from collections import defaultdict
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Container, Dict, List, Optional, Set, Tuple, Type, Union
import numpy as np
import tree
from gymnasium.spaces import Discrete, MultiDiscrete, Space
import ray
from ray import ObjectRef
from ray import cloudpickle as pickle
from ray.rllib.connectors.util import create_connectors_for_policy, maybe_get_filters_for_syncing
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.base_env import BaseEnv, convert_to_base_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.evaluation.sampler import AsyncSampler, SyncSampler
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.offline import D4RLReader, DatasetReader, DatasetWriter, InputReader, IOContext, JsonReader, JsonWriter, MixedInput, NoopOutput, OutputWriter, ShuffledInput
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, MultiAgentBatch, concat_samples, convert_ma_batch_to_sample_batch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils import check_env, force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.debug import summarize, update_global_seed_if_necessary
from ray.rllib.utils.deprecation import DEPRECATED_VALUE, deprecation_warning
from ray.rllib.utils.error import ERR_MSG_NO_GPUS, HOWTO_CHANGE_CONFIG
from ray.rllib.utils.filter import Filter, NoFilter, get_filter
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.policy import create_policy_for_framework, validate_policy_id
from ray.rllib.utils.sgd import do_minibatch_sgd
from ray.rllib.utils.tf_run_builder import _TFRunBuilder
from ray.rllib.utils.tf_utils import get_gpu_devices as get_tf_gpu_devices
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.utils.typing import AgentID, EnvCreator, EnvType, ModelGradients, ModelWeights, MultiAgentPolicyConfigDict, PartialAlgorithmConfigDict, PolicyID, PolicyState, SampleBatchType, T
from ray.tune.registry import registry_contains_input, registry_get_input
from ray.util.annotations import PublicAPI
from ray.util.debug import disable_log_once_globally, enable_periodic_logging, log_once
from ray.util.iter import ParallelIteratorWorker
if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    from ray.rllib.evaluation.episode import Episode
(tf1, tf, tfv) = try_import_tf()
(torch, _) = try_import_torch()
logger = logging.getLogger(__name__)
_global_worker: Optional['RolloutWorker'] = None

@DeveloperAPI
def get_global_worker() -> 'RolloutWorker':
    if False:
        return 10
    'Returns a handle to the active rollout worker in this process.'
    global _global_worker
    return _global_worker

def _update_env_seed_if_necessary(env: EnvType, seed: int, worker_idx: int, vector_idx: int):
    if False:
        return 10
    'Set a deterministic random seed on environment.\n\n    NOTE: this may not work with remote environments (issue #18154).\n    '
    if seed is None:
        return
    max_num_envs_per_workers: int = 1000
    assert worker_idx < max_num_envs_per_workers, 'Too many envs per worker. Random seeds may collide.'
    computed_seed: int = worker_idx * max_num_envs_per_workers + vector_idx + seed
    if not hasattr(env, 'reset'):
        if log_once('env_has_no_reset_method'):
            logger.info(f"Env {env} doesn't have a `reset()` method. Cannot seed.")
    else:
        try:
            env.reset(seed=computed_seed)
        except Exception:
            logger.info(f"Env {env} doesn't support setting a seed via its `reset()` method! Implement this method as `reset(self, *, seed=None, options=None)` for it to abide to the correct API. Cannot seed.")

@DeveloperAPI
class RolloutWorker(ParallelIteratorWorker, EnvRunner):
    """Common experience collection class.

    This class wraps a policy instance and an environment class to
    collect experiences from the environment. You can create many replicas of
    this class as Ray actors to scale RL training.

    This class supports vectorized and multi-agent policy evaluation (e.g.,
    VectorEnv, MultiAgentEnv, etc.)

    .. testcode::
        :skipif: True

        # Create a rollout worker and using it to collect experiences.
        import gymnasium as gym
        from ray.rllib.evaluation.rollout_worker import RolloutWorker
        from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy
        worker = RolloutWorker(
          env_creator=lambda _: gym.make("CartPole-v1"),
          default_policy_class=PPOTF1Policy)
        print(worker.sample())

        # Creating a multi-agent rollout worker
        from gymnasium.spaces import Discrete, Box
        import random
        MultiAgentTrafficGrid = ...
        worker = RolloutWorker(
          env_creator=lambda _: MultiAgentTrafficGrid(num_cars=25),
          config=AlgorithmConfig().multi_agent(
            policies={
              # Use an ensemble of two policies for car agents
              "car_policy1":
                (PGTFPolicy, Box(...), Discrete(...),
                 AlgorithmConfig.overrides(gamma=0.99)),
              "car_policy2":
                (PGTFPolicy, Box(...), Discrete(...),
                 AlgorithmConfig.overrides(gamma=0.95)),
              # Use a single shared policy for all traffic lights
              "traffic_light_policy":
                (PGTFPolicy, Box(...), Discrete(...), {}),
            },
            policy_mapping_fn=(
              lambda agent_id, episode, **kwargs:
              random.choice(["car_policy1", "car_policy2"])
              if agent_id.startswith("car_") else "traffic_light_policy"),
            ),
        )
        print(worker.sample())

    .. testoutput::

        SampleBatch({
            "obs": [[...]], "actions": [[...]], "rewards": [[...]],
            "terminateds": [[...]], "truncateds": [[...]], "new_obs": [[...]]}
        )

        MultiAgentBatch({
            "car_policy1": SampleBatch(...),
            "car_policy2": SampleBatch(...),
            "traffic_light_policy": SampleBatch(...)}
        )

    """

    def __init__(self, *, env_creator: EnvCreator, validate_env: Optional[Callable[[EnvType, EnvContext], None]]=None, config: Optional['AlgorithmConfig']=None, worker_index: int=0, num_workers: Optional[int]=None, recreated_worker: bool=False, log_dir: Optional[str]=None, spaces: Optional[Dict[PolicyID, Tuple[Space, Space]]]=None, default_policy_class: Optional[Type[Policy]]=None, dataset_shards: Optional[List[ray.data.Dataset]]=None, tf_session_creator=DEPRECATED_VALUE):
        if False:
            print('Hello World!')
        "Initializes a RolloutWorker instance.\n\n        Args:\n            env_creator: Function that returns a gym.Env given an EnvContext\n                wrapped configuration.\n            validate_env: Optional callable to validate the generated\n                environment (only on worker=0).\n            worker_index: For remote workers, this should be set to a\n                non-zero and unique value. This index is passed to created envs\n                through EnvContext so that envs can be configured per worker.\n            recreated_worker: Whether this worker is a recreated one. Workers are\n                recreated by an Algorithm (via WorkerSet) in case\n                `recreate_failed_workers=True` and one of the original workers (or an\n                already recreated one) has failed. They don't differ from original\n                workers other than the value of this flag (`self.recreated_worker`).\n            log_dir: Directory where logs can be placed.\n            spaces: An optional space dict mapping policy IDs\n                to (obs_space, action_space)-tuples. This is used in case no\n                Env is created on this RolloutWorker.\n        "
        if tf_session_creator != DEPRECATED_VALUE:
            deprecation_warning(old='RolloutWorker(.., tf_session_creator=.., ..)', new='config.framework(tf_session_args={..}); RolloutWorker(config=config, ..)', error=True)
        self._original_kwargs: dict = locals().copy()
        del self._original_kwargs['self']
        global _global_worker
        _global_worker = self
        from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
        if config is None or isinstance(config, dict):
            config = AlgorithmConfig().update_from_dict(config or {})
        config.freeze()
        if config.extra_python_environs_for_driver and worker_index == 0:
            for (key, value) in config.extra_python_environs_for_driver.items():
                os.environ[key] = str(value)
        elif config.extra_python_environs_for_worker and worker_index > 0:
            for (key, value) in config.extra_python_environs_for_worker.items():
                os.environ[key] = str(value)

        def gen_rollouts():
            if False:
                return 10
            while True:
                yield self.sample()
        ParallelIteratorWorker.__init__(self, gen_rollouts, False)
        EnvRunner.__init__(self, config=config)
        self.num_workers = num_workers if num_workers is not None else self.config.num_rollout_workers
        self._ds_shards = dataset_shards
        self.worker_index: int = worker_index
        self._lock = threading.Lock()
        if tf1 and (config.framework_str == 'tf2' or config.enable_tf1_exec_eagerly) and (not tf1.executing_eagerly()):
            tf1.enable_eager_execution()
        if self.config.log_level:
            logging.getLogger('ray.rllib').setLevel(self.config.log_level)
        if self.worker_index > 1:
            disable_log_once_globally()
        elif self.config.log_level == 'DEBUG':
            enable_periodic_logging()
        env_context = EnvContext(self.config.env_config, worker_index=self.worker_index, vector_index=0, num_workers=self.num_workers, remote=self.config.remote_worker_envs, recreated_worker=recreated_worker)
        self.env_context = env_context
        self.config: AlgorithmConfig = config
        self.callbacks: DefaultCallbacks = self.config.callbacks_class()
        self.recreated_worker: bool = recreated_worker
        self.policy_mapping_fn = lambda agent_id, episode, worker, **kw: DEFAULT_POLICY_ID
        self.set_policy_mapping_fn(self.config.policy_mapping_fn)
        self.env_creator: EnvCreator = env_creator
        configured_rollout_fragment_length = self.config.get_rollout_fragment_length(worker_index=self.worker_index)
        self.total_rollout_fragment_length: int = configured_rollout_fragment_length * self.config.num_envs_per_worker
        self.preprocessing_enabled: bool = not config._disable_preprocessor_api
        self.last_batch: Optional[SampleBatchType] = None
        self.global_vars: dict = {'timestep': 0, 'num_grad_updates_per_policy': defaultdict(int)}
        self.seed = None if self.config.seed is None else self.config.seed + self.worker_index + self.config.in_evaluation * 10000
        if self.worker_index > 0:
            update_global_seed_if_necessary(self.config.framework_str, self.seed)
        self.env = self.make_sub_env_fn = None
        if not (self.worker_index == 0 and self.num_workers > 0 and (not self.config.create_env_on_local_worker)):
            self.env = env_creator(copy.deepcopy(self.env_context))
        clip_rewards = self.config.clip_rewards
        if self.env is not None:
            if not self.config.disable_env_checking:
                check_env(self.env, self.config)
            if validate_env is not None:
                validate_env(self.env, self.env_context)
            if isinstance(self.env, (BaseEnv, ray.actor.ActorHandle)):

                def wrap(env):
                    if False:
                        i = 10
                        return i + 15
                    return env
            elif is_atari(self.env) and self.config.preprocessor_pref == 'deepmind':
                self.preprocessing_enabled = False
                if self.config.clip_rewards is None:
                    clip_rewards = True
                use_framestack = self.config.model.get('framestack') is True

                def wrap(env):
                    if False:
                        return 10
                    env = wrap_deepmind(env, dim=self.config.model.get('dim'), framestack=use_framestack, noframeskip=self.config.env_config.get('frameskip', 0) == 1)
                    return env
            elif self.config.preprocessor_pref is None:
                self.preprocessing_enabled = False

                def wrap(env):
                    if False:
                        for i in range(10):
                            print('nop')
                    return env
            else:

                def wrap(env):
                    if False:
                        return 10
                    return env
            self.env: EnvType = wrap(self.env)
            _update_env_seed_if_necessary(self.env, self.seed, self.worker_index, 0)
            self.callbacks.on_sub_environment_created(worker=self, sub_environment=self.env, env_context=self.env_context)
            self.make_sub_env_fn = self._get_make_sub_env_fn(env_creator, env_context, validate_env, wrap, self.seed)
        self.spaces = spaces
        self.default_policy_class = default_policy_class
        (self.policy_dict, self.is_policy_to_train) = self.config.get_multi_agent_setup(env=self.env, spaces=self.spaces, default_policy_class=self.default_policy_class)
        self.policy_map: Optional[PolicyMap] = None
        self.preprocessors: Dict[PolicyID, Preprocessor] = None
        num_gpus = self.config.num_gpus if self.worker_index == 0 else self.config.num_gpus_per_worker
        if not self.config._enable_new_api_stack:
            if ray.is_initialized() and ray._private.worker._mode() != ray._private.worker.LOCAL_MODE and (not config._fake_gpus):
                devices = []
                if self.config.framework_str in ['tf2', 'tf']:
                    devices = get_tf_gpu_devices()
                elif self.config.framework_str == 'torch':
                    devices = list(range(torch.cuda.device_count()))
                if len(devices) < num_gpus:
                    raise RuntimeError(ERR_MSG_NO_GPUS.format(len(devices), devices) + HOWTO_CHANGE_CONFIG)
            elif ray.is_initialized() and ray._private.worker._mode() == ray._private.worker.LOCAL_MODE and (num_gpus > 0) and (not self.config._fake_gpus):
                logger.warning(f'You are running ray with `local_mode=True`, but have configured {num_gpus} GPUs to be used! In local mode, Policies are placed on the CPU and the `num_gpus` setting is ignored.')
        self.filters: Dict[PolicyID, Filter] = defaultdict(NoFilter)
        self.marl_module_spec = None
        self._update_policy_map(policy_dict=self.policy_dict)
        for pol in self.policy_map.values():
            if not pol._model_init_state_automatically_added and (not pol.config.get('_enable_new_api_stack', False)):
                pol._update_model_view_requirements_from_init_state()
        self.multiagent: bool = set(self.policy_map.keys()) != {DEFAULT_POLICY_ID}
        if self.multiagent and self.env is not None:
            if not isinstance(self.env, (BaseEnv, ExternalMultiAgentEnv, MultiAgentEnv, ray.actor.ActorHandle)):
                raise ValueError(f'Have multiple policies {self.policy_map}, but the env {self.env} is not a subclass of BaseEnv, MultiAgentEnv, ActorHandle, or ExternalMultiAgentEnv!')
        if self.worker_index == 0:
            logger.info('Built filter map: {}'.format(self.filters))
        if self.env is None:
            self.async_env = None
        elif 'custom_vector_env' in self.config:
            self.async_env = self.config.custom_vector_env(self.env)
        else:
            self.async_env: BaseEnv = convert_to_base_env(self.env, make_env=self.make_sub_env_fn, num_envs=self.config.num_envs_per_worker, remote_envs=self.config.remote_worker_envs, remote_env_batch_wait_ms=self.config.remote_env_batch_wait_ms, worker=self, restart_failed_sub_environments=self.config.restart_failed_sub_environments)
        rollout_fragment_length_for_sampler = configured_rollout_fragment_length
        if self.config.batch_mode == 'truncate_episodes':
            pack = True
        else:
            assert self.config.batch_mode == 'complete_episodes'
            rollout_fragment_length_for_sampler = float('inf')
            pack = False
        self.io_context: IOContext = IOContext(log_dir, self.config, self.worker_index, self)
        render = False
        if self.config.render_env is True and (self.num_workers == 0 or self.worker_index == 1):
            render = True
        if self.env is None:
            self.sampler = None
        elif self.config.sample_async:
            self.sampler = AsyncSampler(worker=self, env=self.async_env, clip_rewards=clip_rewards, rollout_fragment_length=rollout_fragment_length_for_sampler, count_steps_by=self.config.count_steps_by, callbacks=self.callbacks, multiple_episodes_in_batch=pack, normalize_actions=self.config.normalize_actions, clip_actions=self.config.clip_actions, observation_fn=self.config.observation_fn, sample_collector_class=self.config.sample_collector, render=render)
            self.sampler.start()
        else:
            self.sampler = SyncSampler(worker=self, env=self.async_env, clip_rewards=clip_rewards, rollout_fragment_length=rollout_fragment_length_for_sampler, count_steps_by=self.config.count_steps_by, callbacks=self.callbacks, multiple_episodes_in_batch=pack, normalize_actions=self.config.normalize_actions, clip_actions=self.config.clip_actions, observation_fn=self.config.observation_fn, sample_collector_class=self.config.sample_collector, render=render)
        self.input_reader: InputReader = self._get_input_creator_from_config()(self.io_context)
        self.output_writer: OutputWriter = self._get_output_creator_from_config()(self.io_context)
        self.weights_seq_no: Optional[int] = None
        logger.debug('Created rollout worker with env {} ({}), policies {}'.format(self.async_env, self.env, self.policy_map))

    @override(EnvRunner)
    def assert_healthy(self):
        if False:
            for i in range(10):
                print('nop')
        is_healthy = self.policy_map and self.input_reader and self.output_writer
        assert is_healthy, f'RolloutWorker {self} (idx={self.worker_index}; num_workers={self.num_workers}) not healthy!'

    @override(EnvRunner)
    def sample(self, **kwargs) -> SampleBatchType:
        if False:
            return 10
        'Returns a batch of experience sampled from this worker.\n\n        This method must be implemented by subclasses.\n\n        Returns:\n            A columnar batch of experiences (e.g., tensors) or a MultiAgentBatch.\n\n        .. testcode::\n            :skipif: True\n\n            import gymnasium as gym\n            from ray.rllib.evaluation.rollout_worker import RolloutWorker\n            from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy\n            worker = RolloutWorker(\n              env_creator=lambda _: gym.make("CartPole-v1"),\n              default_policy_class=PPOTF1Policy,\n              config=AlgorithmConfig(),\n            )\n            print(worker.sample())\n\n        .. testoutput::\n\n            SampleBatch({"obs": [...], "action": [...], ...})\n        '
        if self.config.fake_sampler and self.last_batch is not None:
            return self.last_batch
        elif self.input_reader is None:
            raise ValueError('RolloutWorker has no `input_reader` object! Cannot call `sample()`. You can try setting `create_env_on_driver` to True.')
        if log_once('sample_start'):
            logger.info('Generating sample batch of size {}'.format(self.total_rollout_fragment_length))
        batches = [self.input_reader.next()]
        steps_so_far = batches[0].count if self.config.count_steps_by == 'env_steps' else batches[0].agent_steps()
        if self.config.batch_mode == 'truncate_episodes' and (not self.config.offline_sampling):
            max_batches = self.config.num_envs_per_worker
        else:
            max_batches = float('inf')
        while steps_so_far < self.total_rollout_fragment_length and len(batches) < max_batches:
            batch = self.input_reader.next()
            steps_so_far += batch.count if self.config.count_steps_by == 'env_steps' else batch.agent_steps()
            batches.append(batch)
        batch = concat_samples(batches)
        self.callbacks.on_sample_end(worker=self, samples=batch)
        self.output_writer.write(batch)
        if log_once('sample_end'):
            logger.info('Completed sample batch:\n\n{}\n'.format(summarize(batch)))
        if self.config.compress_observations:
            batch.compress(bulk=self.config.compress_observations == 'bulk')
        if self.config.fake_sampler:
            self.last_batch = batch
        return batch

    @ray.method(num_returns=2)
    def sample_with_count(self) -> Tuple[SampleBatchType, int]:
        if False:
            while True:
                i = 10
        'Same as sample() but returns the count as a separate value.\n\n        Returns:\n            A columnar batch of experiences (e.g., tensors) and the\n                size of the collected batch.\n\n        .. testcode::\n            :skipif: True\n\n            import gymnasium as gym\n            from ray.rllib.evaluation.rollout_worker import RolloutWorker\n            from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy\n            worker = RolloutWorker(\n              env_creator=lambda _: gym.make("CartPole-v1"),\n              default_policy_class=PPOTFPolicy)\n            print(worker.sample_with_count())\n\n        .. testoutput::\n\n            (SampleBatch({"obs": [...], "action": [...], ...}), 3)\n        '
        batch = self.sample()
        return (batch, batch.count)

    def learn_on_batch(self, samples: SampleBatchType) -> Dict:
        if False:
            print('Hello World!')
        'Update policies based on the given batch.\n\n        This is the equivalent to apply_gradients(compute_gradients(samples)),\n        but can be optimized to avoid pulling gradients into CPU memory.\n\n        Args:\n            samples: The SampleBatch or MultiAgentBatch to learn on.\n\n        Returns:\n            Dictionary of extra metadata from compute_gradients().\n\n        .. testcode::\n            :skipif: True\n\n            import gymnasium as gym\n            from ray.rllib.evaluation.rollout_worker import RolloutWorker\n            from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy\n            worker = RolloutWorker(\n              env_creator=lambda _: gym.make("CartPole-v1"),\n              default_policy_class=PPOTF1Policy)\n            batch = worker.sample()\n            info = worker.learn_on_batch(samples)\n        '
        if log_once('learn_on_batch'):
            logger.info('Training on concatenated sample batches:\n\n{}\n'.format(summarize(samples)))
        info_out = {}
        if isinstance(samples, MultiAgentBatch):
            builders = {}
            to_fetch = {}
            for (pid, batch) in samples.policy_batches.items():
                if self.is_policy_to_train is not None and (not self.is_policy_to_train(pid, samples)):
                    continue
                batch.decompress_if_needed()
                policy = self.policy_map[pid]
                tf_session = policy.get_session()
                if tf_session and hasattr(policy, '_build_learn_on_batch'):
                    builders[pid] = _TFRunBuilder(tf_session, 'learn_on_batch')
                    to_fetch[pid] = policy._build_learn_on_batch(builders[pid], batch)
                else:
                    info_out[pid] = policy.learn_on_batch(batch)
            info_out.update({pid: builders[pid].get(v) for (pid, v) in to_fetch.items()})
        elif self.is_policy_to_train is None or self.is_policy_to_train(DEFAULT_POLICY_ID, samples):
            info_out.update({DEFAULT_POLICY_ID: self.policy_map[DEFAULT_POLICY_ID].learn_on_batch(samples)})
        if log_once('learn_out'):
            logger.debug('Training out:\n\n{}\n'.format(summarize(info_out)))
        return info_out

    def sample_and_learn(self, expected_batch_size: int, num_sgd_iter: int, sgd_minibatch_size: str, standardize_fields: List[str]) -> Tuple[dict, int]:
        if False:
            for i in range(10):
                print('nop')
        "Sample and batch and learn on it.\n\n        This is typically used in combination with distributed allreduce.\n\n        Args:\n            expected_batch_size: Expected number of samples to learn on.\n            num_sgd_iter: Number of SGD iterations.\n            sgd_minibatch_size: SGD minibatch size.\n            standardize_fields: List of sample fields to normalize.\n\n        Returns:\n            A tuple consisting of a dictionary of extra metadata returned from\n                the policies' `learn_on_batch()` and the number of samples\n                learned on.\n        "
        batch = self.sample()
        assert batch.count == expected_batch_size, ('Batch size possibly out of sync between workers, expected:', expected_batch_size, 'got:', batch.count)
        logger.info('Executing distributed minibatch SGD with epoch size {}, minibatch size {}'.format(batch.count, sgd_minibatch_size))
        info = do_minibatch_sgd(batch, self.policy_map, self, num_sgd_iter, sgd_minibatch_size, standardize_fields)
        return (info, batch.count)

    def compute_gradients(self, samples: SampleBatchType, single_agent: bool=None) -> Tuple[ModelGradients, dict]:
        if False:
            i = 10
            return i + 15
        'Returns a gradient computed w.r.t the specified samples.\n\n        Uses the Policy\'s/ies\' compute_gradients method(s) to perform the\n        calculations. Skips policies that are not trainable as per\n        `self.is_policy_to_train()`.\n\n        Args:\n            samples: The SampleBatch or MultiAgentBatch to compute gradients\n                for using this worker\'s trainable policies.\n\n        Returns:\n            In the single-agent case, a tuple consisting of ModelGradients and\n            info dict of the worker\'s policy.\n            In the multi-agent case, a tuple consisting of a dict mapping\n            PolicyID to ModelGradients and a dict mapping PolicyID to extra\n            metadata info.\n            Note that the first return value (grads) can be applied as is to a\n            compatible worker using the worker\'s `apply_gradients()` method.\n\n        .. testcode::\n            :skipif: True\n\n            import gymnasium as gym\n            from ray.rllib.evaluation.rollout_worker import RolloutWorker\n            from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy\n            worker = RolloutWorker(\n              env_creator=lambda _: gym.make("CartPole-v1"),\n              default_policy_class=PPOTF1Policy)\n            batch = worker.sample()\n            grads, info = worker.compute_gradients(samples)\n        '
        if log_once('compute_gradients'):
            logger.info('Compute gradients on:\n\n{}\n'.format(summarize(samples)))
        if single_agent is True:
            samples = convert_ma_batch_to_sample_batch(samples)
            (grad_out, info_out) = self.policy_map[DEFAULT_POLICY_ID].compute_gradients(samples)
            info_out['batch_count'] = samples.count
            return (grad_out, info_out)
        samples = samples.as_multi_agent()
        (grad_out, info_out) = ({}, {})
        if self.config.framework_str == 'tf':
            for (pid, batch) in samples.policy_batches.items():
                if self.is_policy_to_train is not None and (not self.is_policy_to_train(pid, samples)):
                    continue
                policy = self.policy_map[pid]
                builder = _TFRunBuilder(policy.get_session(), 'compute_gradients')
                (grad_out[pid], info_out[pid]) = policy._build_compute_gradients(builder, batch)
            grad_out = {k: builder.get(v) for (k, v) in grad_out.items()}
            info_out = {k: builder.get(v) for (k, v) in info_out.items()}
        else:
            for (pid, batch) in samples.policy_batches.items():
                if self.is_policy_to_train is not None and (not self.is_policy_to_train(pid, samples)):
                    continue
                (grad_out[pid], info_out[pid]) = self.policy_map[pid].compute_gradients(batch)
        info_out['batch_count'] = samples.count
        if log_once('grad_out'):
            logger.info('Compute grad info:\n\n{}\n'.format(summarize(info_out)))
        return (grad_out, info_out)

    def apply_gradients(self, grads: Union[ModelGradients, Dict[PolicyID, ModelGradients]]) -> None:
        if False:
            print('Hello World!')
        'Applies the given gradients to this worker\'s models.\n\n        Uses the Policy\'s/ies\' apply_gradients method(s) to perform the\n        operations.\n\n        Args:\n            grads: Single ModelGradients (single-agent case) or a dict\n                mapping PolicyIDs to the respective model gradients\n                structs.\n\n        .. testcode::\n            :skipif: True\n\n            import gymnasium as gym\n            from ray.rllib.evaluation.rollout_worker import RolloutWorker\n            from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy\n            worker = RolloutWorker(\n              env_creator=lambda _: gym.make("CartPole-v1"),\n              default_policy_class=PPOTF1Policy)\n            samples = worker.sample()\n            grads, info = worker.compute_gradients(samples)\n            worker.apply_gradients(grads)\n        '
        if log_once('apply_gradients'):
            logger.info('Apply gradients:\n\n{}\n'.format(summarize(grads)))
        if isinstance(grads, dict):
            for (pid, g) in grads.items():
                if self.is_policy_to_train is None or self.is_policy_to_train(pid, None):
                    self.policy_map[pid].apply_gradients(g)
        elif self.is_policy_to_train is None or self.is_policy_to_train(DEFAULT_POLICY_ID, None):
            self.policy_map[DEFAULT_POLICY_ID].apply_gradients(grads)

    def get_metrics(self) -> List[RolloutMetrics]:
        if False:
            i = 10
            return i + 15
        "Returns the thus-far collected metrics from this worker's rollouts.\n\n        Returns:\n             List of RolloutMetrics collected thus-far.\n        "
        if self.sampler is not None:
            out = self.sampler.get_metrics()
        else:
            out = []
        return out

    def foreach_env(self, func: Callable[[EnvType], T]) -> List[T]:
        if False:
            for i in range(10):
                print('nop')
        'Calls the given function with each sub-environment as arg.\n\n        Args:\n            func: The function to call for each underlying\n                sub-environment (as only arg).\n\n        Returns:\n             The list of return values of all calls to `func([env])`.\n        '
        if self.async_env is None:
            return []
        envs = self.async_env.get_sub_environments()
        if not envs:
            return [func(self.async_env)]
        else:
            return [func(e) for e in envs]

    def foreach_env_with_context(self, func: Callable[[EnvType, EnvContext], T]) -> List[T]:
        if False:
            while True:
                i = 10
        'Calls given function with each sub-env plus env_ctx as args.\n\n        Args:\n            func: The function to call for each underlying\n                sub-environment and its EnvContext (as the args).\n\n        Returns:\n             The list of return values of all calls to `func([env, ctx])`.\n        '
        if self.async_env is None:
            return []
        envs = self.async_env.get_sub_environments()
        if not envs:
            return [func(self.async_env, self.env_context)]
        else:
            ret = []
            for (i, e) in enumerate(envs):
                ctx = self.env_context.copy_with_overrides(vector_index=i)
                ret.append(func(e, ctx))
            return ret

    def get_policy(self, policy_id: PolicyID=DEFAULT_POLICY_ID) -> Optional[Policy]:
        if False:
            while True:
                i = 10
        'Return policy for the specified id, or None.\n\n        Args:\n            policy_id: ID of the policy to return. None for DEFAULT_POLICY_ID\n                (in the single agent case).\n\n        Returns:\n            The policy under the given ID (or None if not found).\n        '
        return self.policy_map.get(policy_id)

    def add_policy(self, policy_id: PolicyID, policy_cls: Optional[Type[Policy]]=None, policy: Optional[Policy]=None, *, observation_space: Optional[Space]=None, action_space: Optional[Space]=None, config: Optional[PartialAlgorithmConfigDict]=None, policy_state: Optional[PolicyState]=None, policy_mapping_fn: Optional[Callable[[AgentID, 'Episode'], PolicyID]]=None, policies_to_train: Optional[Union[Container[PolicyID], Callable[[PolicyID, SampleBatchType], bool]]]=None, module_spec: Optional[SingleAgentRLModuleSpec]=None) -> Policy:
        if False:
            for i in range(10):
                print('nop')
        "Adds a new policy to this RolloutWorker.\n\n        Args:\n            policy_id: ID of the policy to add.\n            policy_cls: The Policy class to use for constructing the new Policy.\n                Note: Only one of `policy_cls` or `policy` must be provided.\n            policy: The Policy instance to add to this algorithm.\n                Note: Only one of `policy_cls` or `policy` must be provided.\n            observation_space: The observation space of the policy to add.\n            action_space: The action space of the policy to add.\n            config: The config overrides for the policy to add.\n            policy_state: Optional state dict to apply to the new\n                policy instance, right after its construction.\n            policy_mapping_fn: An optional (updated) policy mapping function\n                to use from here on. Note that already ongoing episodes will\n                not change their mapping but will use the old mapping till\n                the end of the episode.\n            policies_to_train: An optional container of policy IDs to be\n                trained or a callable taking PolicyID and - optionally -\n                SampleBatchType and returning a bool (trainable or not?).\n                If None, will keep the existing setup in place.\n                Policies, whose IDs are not in the list (or for which the\n                callable returns False) will not be updated.\n            module_spec: In the new RLModule API we need to pass in the module_spec for\n                the new module that is supposed to be added. Knowing the policy spec is\n                not sufficient.\n\n        Returns:\n            The newly added policy.\n\n        Raises:\n            ValueError: If both `policy_cls` AND `policy` are provided.\n            KeyError: If the given `policy_id` already exists in this worker's\n                PolicyMap.\n        "
        validate_policy_id(policy_id, error=False)
        if module_spec is not None and (not self.config._enable_new_api_stack):
            raise ValueError('If you pass in module_spec to the policy, the RLModule API needs to be enabled.')
        if policy_id in self.policy_map:
            raise KeyError(f"Policy ID '{policy_id}' already exists in policy map! Make sure you use a Policy ID that has not been taken yet. Policy IDs that are already in your policy map: {list(self.policy_map.keys())}")
        if (policy_cls is None) == (policy is None):
            raise ValueError('Only one of `policy_cls` or `policy` must be provided to RolloutWorker.add_policy()!')
        if policy is None:
            (policy_dict_to_add, _) = self.config.get_multi_agent_setup(policies={policy_id: PolicySpec(policy_cls, observation_space, action_space, config)}, env=self.env, spaces=self.spaces, default_policy_class=self.default_policy_class)
        else:
            policy_dict_to_add = {policy_id: PolicySpec(type(policy), policy.observation_space, policy.action_space, policy.config)}
        self.policy_dict.update(policy_dict_to_add)
        self._update_policy_map(policy_dict=policy_dict_to_add, policy=policy, policy_states={policy_id: policy_state}, single_agent_rl_module_spec=module_spec)
        self.set_policy_mapping_fn(policy_mapping_fn)
        if policies_to_train is not None:
            self.set_is_policy_to_train(policies_to_train)
        return self.policy_map[policy_id]

    def remove_policy(self, *, policy_id: PolicyID=DEFAULT_POLICY_ID, policy_mapping_fn: Optional[Callable[[AgentID], PolicyID]]=None, policies_to_train: Optional[Union[Container[PolicyID], Callable[[PolicyID, SampleBatchType], bool]]]=None) -> None:
        if False:
            return 10
        'Removes a policy from this RolloutWorker.\n\n        Args:\n            policy_id: ID of the policy to be removed. None for\n                DEFAULT_POLICY_ID.\n            policy_mapping_fn: An optional (updated) policy mapping function\n                to use from here on. Note that already ongoing episodes will\n                not change their mapping but will use the old mapping till\n                the end of the episode.\n            policies_to_train: An optional container of policy IDs to be\n                trained or a callable taking PolicyID and - optionally -\n                SampleBatchType and returning a bool (trainable or not?).\n                If None, will keep the existing setup in place.\n                Policies, whose IDs are not in the list (or for which the\n                callable returns False) will not be updated.\n        '
        if policy_id not in self.policy_map:
            raise ValueError(f"Policy ID '{policy_id}' not in policy map!")
        del self.policy_map[policy_id]
        del self.preprocessors[policy_id]
        self.set_policy_mapping_fn(policy_mapping_fn)
        if policies_to_train is not None:
            self.set_is_policy_to_train(policies_to_train)

    def set_policy_mapping_fn(self, policy_mapping_fn: Optional[Callable[[AgentID, 'Episode'], PolicyID]]=None) -> None:
        if False:
            print('Hello World!')
        'Sets `self.policy_mapping_fn` to a new callable (if provided).\n\n        Args:\n            policy_mapping_fn: The new mapping function to use. If None,\n                will keep the existing mapping function in place.\n        '
        if policy_mapping_fn is not None:
            self.policy_mapping_fn = policy_mapping_fn
            if not callable(self.policy_mapping_fn):
                raise ValueError('`policy_mapping_fn` must be a callable!')

    def set_is_policy_to_train(self, is_policy_to_train: Union[Container[PolicyID], Callable[[PolicyID, Optional[SampleBatchType]], bool]]) -> None:
        if False:
            print('Hello World!')
        'Sets `self.is_policy_to_train()` to a new callable.\n\n        Args:\n            is_policy_to_train: A container of policy IDs to be\n                trained or a callable taking PolicyID and - optionally -\n                SampleBatchType and returning a bool (trainable or not?).\n                If None, will keep the existing setup in place.\n                Policies, whose IDs are not in the list (or for which the\n                callable returns False) will not be updated.\n        '
        if not callable(is_policy_to_train):
            assert isinstance(is_policy_to_train, (list, set, tuple)), 'ERROR: `is_policy_to_train`must be a [list|set|tuple] or a callable taking PolicyID and SampleBatch and returning True|False (trainable or not?).'
            pols = set(is_policy_to_train)

            def is_policy_to_train(pid, batch=None):
                if False:
                    for i in range(10):
                        print('nop')
                return pid in pols
        self.is_policy_to_train = is_policy_to_train

    @PublicAPI(stability='alpha')
    def get_policies_to_train(self, batch: Optional[SampleBatchType]=None) -> Set[PolicyID]:
        if False:
            while True:
                i = 10
        'Returns all policies-to-train, given an optional batch.\n\n        Loops through all policies currently in `self.policy_map` and checks\n        the return value of `self.is_policy_to_train(pid, batch)`.\n\n        Args:\n            batch: An optional SampleBatchType for the\n                `self.is_policy_to_train(pid, [batch]?)` check.\n\n        Returns:\n            The set of currently trainable policy IDs, given the optional\n            `batch`.\n        '
        return {pid for pid in self.policy_map.keys() if self.is_policy_to_train is None or self.is_policy_to_train(pid, batch)}

    def for_policy(self, func: Callable[[Policy, Optional[Any]], T], policy_id: Optional[PolicyID]=DEFAULT_POLICY_ID, **kwargs) -> T:
        if False:
            i = 10
            return i + 15
        'Calls the given function with the specified policy as first arg.\n\n        Args:\n            func: The function to call with the policy as first arg.\n            policy_id: The PolicyID of the policy to call the function with.\n\n        Keyword Args:\n            kwargs: Additional kwargs to be passed to the call.\n\n        Returns:\n            The return value of the function call.\n        '
        return func(self.policy_map[policy_id], **kwargs)

    def foreach_policy(self, func: Callable[[Policy, PolicyID, Optional[Any]], T], **kwargs) -> List[T]:
        if False:
            i = 10
            return i + 15
        'Calls the given function with each (policy, policy_id) tuple.\n\n        Args:\n            func: The function to call with each (policy, policy ID) tuple.\n\n        Keyword Args:\n            kwargs: Additional kwargs to be passed to the call.\n\n        Returns:\n             The list of return values of all calls to\n                `func([policy, pid, **kwargs])`.\n        '
        return [func(policy, pid, **kwargs) for (pid, policy) in self.policy_map.items()]

    def foreach_policy_to_train(self, func: Callable[[Policy, PolicyID, Optional[Any]], T], **kwargs) -> List[T]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Calls the given function with each (policy, policy_id) tuple.\n\n        Only those policies/IDs will be called on, for which\n        `self.is_policy_to_train()` returns True.\n\n        Args:\n            func: The function to call with each (policy, policy ID) tuple,\n                for only those policies that `self.is_policy_to_train`\n                returns True.\n\n        Keyword Args:\n            kwargs: Additional kwargs to be passed to the call.\n\n        Returns:\n            The list of return values of all calls to\n            `func([policy, pid, **kwargs])`.\n        '
        return [func(self.policy_map[pid], pid, **kwargs) for pid in self.policy_map.keys() if self.is_policy_to_train is None or self.is_policy_to_train(pid, None)]

    def sync_filters(self, new_filters: dict) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Changes self's filter to given and rebases any accumulated delta.\n\n        Args:\n            new_filters: Filters with new state to update local copy.\n        "
        assert all((k in new_filters for k in self.filters))
        for k in self.filters:
            self.filters[k].sync(new_filters[k])

    def get_filters(self, flush_after: bool=False) -> Dict:
        if False:
            print('Hello World!')
        'Returns a snapshot of filters.\n\n        Args:\n            flush_after: Clears the filter buffer state.\n\n        Returns:\n            Dict for serializable filters\n        '
        return_filters = {}
        for (k, f) in self.filters.items():
            return_filters[k] = f.as_serializable()
            if flush_after:
                f.reset_buffer()
        return return_filters

    @override(EnvRunner)
    def get_state(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        filters = self.get_filters(flush_after=True)
        policy_states = {}
        for pid in self.policy_map.keys():
            if not self.config.checkpoint_trainable_policies_only or self.is_policy_to_train is None or self.is_policy_to_train(pid):
                policy_states[pid] = self.policy_map[pid].get_state()
        return {'policy_ids': list(self.policy_map.keys()), 'policy_states': policy_states, 'policy_mapping_fn': self.policy_mapping_fn, 'is_policy_to_train': self.is_policy_to_train, 'filters': filters}

    @override(EnvRunner)
    def set_state(self, state: dict) -> None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(state, bytes):
            state = pickle.loads(state)
        self.sync_filters(state['filters'])
        connector_enabled = self.config.enable_connectors
        policy_states = state['policy_states'] if 'policy_states' in state else state['state']
        for (pid, policy_state) in policy_states.items():
            validate_policy_id(pid, error=False)
            if pid not in self.policy_map:
                spec = policy_state.get('policy_spec', None)
                if spec is None:
                    logger.warning(f"PolicyID '{pid}' was probably added on-the-fly (not part of the static `multagent.policies` config) and no PolicySpec objects found in the pickled policy state. Will not add `{pid}`, but ignore it for now.")
                else:
                    policy_spec = PolicySpec.deserialize(spec) if connector_enabled or isinstance(spec, dict) else spec
                    self.add_policy(policy_id=pid, policy_cls=policy_spec.policy_class, observation_space=policy_spec.observation_space, action_space=policy_spec.action_space, config=policy_spec.config)
            if pid in self.policy_map:
                self.policy_map[pid].set_state(policy_state)
        if 'policy_mapping_fn' in state:
            self.set_policy_mapping_fn(state['policy_mapping_fn'])
        if state.get('is_policy_to_train') is not None:
            self.set_is_policy_to_train(state['is_policy_to_train'])

    def get_weights(self, policies: Optional[Container[PolicyID]]=None) -> Dict[PolicyID, ModelWeights]:
        if False:
            print('Hello World!')
        'Returns each policies\' model weights of this worker.\n\n        Args:\n            policies: List of PolicyIDs to get the weights from.\n                Use None for all policies.\n\n        Returns:\n            Dict mapping PolicyIDs to ModelWeights.\n\n        .. testcode::\n            :skipif: True\n\n            from ray.rllib.evaluation.rollout_worker import RolloutWorker\n            # Create a RolloutWorker.\n            worker = ...\n            weights = worker.get_weights()\n            print(weights)\n\n        .. testoutput::\n\n            {"default_policy": {"layer1": array(...), "layer2": ...}}\n        '
        if policies is None:
            policies = list(self.policy_map.keys())
        policies = force_list(policies)
        return {pid: self.policy_map[pid].get_weights() for pid in self.policy_map.keys() if pid in policies}

    def set_weights(self, weights: Dict[PolicyID, ModelWeights], global_vars: Optional[Dict]=None, weights_seq_no: Optional[int]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Sets each policies\' model weights of this worker.\n\n        Args:\n            weights: Dict mapping PolicyIDs to the new weights to be used.\n            global_vars: An optional global vars dict to set this\n                worker to. If None, do not update the global_vars.\n            weights_seq_no: If needed, a sequence number for the weights version\n                can be passed into this method. If not None, will store this seq no\n                (in self.weights_seq_no) and in future calls - if the seq no did not\n                change wrt. the last call - will ignore the call to save on performance.\n\n        .. testcode::\n            :skipif: True\n\n            from ray.rllib.evaluation.rollout_worker import RolloutWorker\n            # Create a RolloutWorker.\n            worker = ...\n            weights = worker.get_weights()\n            # Set `global_vars` (timestep) as well.\n            worker.set_weights(weights, {"timestep": 42})\n        '
        if weights_seq_no is None or weights_seq_no != self.weights_seq_no:
            if weights and isinstance(next(iter(weights.values())), ObjectRef):
                actual_weights = ray.get(list(weights.values()))
                weights = {pid: actual_weights[i] for (i, pid) in enumerate(weights.keys())}
            for (pid, w) in weights.items():
                self.policy_map[pid].set_weights(w)
        self.weights_seq_no = weights_seq_no
        if global_vars:
            self.set_global_vars(global_vars)

    def get_global_vars(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        'Returns the current `self.global_vars` dict of this RolloutWorker.\n\n        Returns:\n            The current `self.global_vars` dict of this RolloutWorker.\n\n        .. testcode::\n            :skipif: True\n\n            from ray.rllib.evaluation.rollout_worker import RolloutWorker\n            # Create a RolloutWorker.\n            worker = ...\n            global_vars = worker.get_global_vars()\n            print(global_vars)\n\n        .. testoutput::\n\n            {"timestep": 424242}\n        '
        return self.global_vars

    def set_global_vars(self, global_vars: dict, policy_ids: Optional[List[PolicyID]]=None) -> None:
        if False:
            print('Hello World!')
        'Updates this worker\'s and all its policies\' global vars.\n\n        Updates are done using the dict\'s update method.\n\n        Args:\n            global_vars: The global_vars dict to update the `self.global_vars` dict\n                from.\n            policy_ids: Optional list of Policy IDs to update. If None, will update all\n                policies on the to-be-updated workers.\n\n        .. testcode::\n            :skipif: True\n\n            worker = ...\n            global_vars = worker.set_global_vars(\n            ...     {"timestep": 4242})\n        '
        global_vars_copy = global_vars.copy()
        gradient_updates_per_policy = global_vars_copy.pop('num_grad_updates_per_policy', {})
        self.global_vars['num_grad_updates_per_policy'].update(gradient_updates_per_policy)
        for pid in policy_ids if policy_ids is not None else self.policy_map.keys():
            if self.is_policy_to_train is None or self.is_policy_to_train(pid, None):
                self.policy_map[pid].on_global_var_update(dict(global_vars_copy, **{'num_grad_updates': gradient_updates_per_policy.get(pid)}))
        self.global_vars.update(global_vars_copy)

    @override(EnvRunner)
    def stop(self) -> None:
        if False:
            return 10
        'Releases all resources used by this RolloutWorker.'
        if self.env is not None:
            self.async_env.stop()
        if hasattr(self, 'sampler') and isinstance(self.sampler, AsyncSampler):
            self.sampler.shutdown = True
        for policy in self.policy_map.cache.values():
            sess = policy.get_session()
            if sess is not None:
                sess.close()

    def lock(self) -> None:
        if False:
            return 10
        'Locks this RolloutWorker via its own threading.Lock.'
        self._lock.acquire()

    def unlock(self) -> None:
        if False:
            return 10
        'Unlocks this RolloutWorker via its own threading.Lock.'
        self._lock.release()

    def setup_torch_data_parallel(self, url: str, world_rank: int, world_size: int, backend: str) -> None:
        if False:
            i = 10
            return i + 15
        'Join a torch process group for distributed SGD.'
        logger.info('Joining process group, url={}, world_rank={}, world_size={}, backend={}'.format(url, world_rank, world_size, backend))
        torch.distributed.init_process_group(backend=backend, init_method=url, rank=world_rank, world_size=world_size)
        for (pid, policy) in self.policy_map.items():
            if not isinstance(policy, (TorchPolicy, TorchPolicyV2)):
                raise ValueError('This policy does not support torch distributed', policy)
            policy.distributed_world_size = world_size

    def creation_args(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        'Returns the kwargs dict used to create this worker.'
        return self._original_kwargs

    @DeveloperAPI
    def get_host(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns the hostname of the process running this evaluator.'
        return platform.node()

    @DeveloperAPI
    def get_node_ip(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns the IP address of the node that this worker runs on.'
        return ray.util.get_node_ip_address()

    @DeveloperAPI
    def find_free_port(self) -> int:
        if False:
            while True:
                i = 10
        'Finds a free port on the node that this worker runs on.'
        from ray.air._internal.util import find_free_port
        return find_free_port()

    def _update_policy_map(self, *, policy_dict: MultiAgentPolicyConfigDict, policy: Optional[Policy]=None, policy_states: Optional[Dict[PolicyID, PolicyState]]=None, single_agent_rl_module_spec: Optional[SingleAgentRLModuleSpec]=None) -> None:
        if False:
            i = 10
            return i + 15
        "Updates the policy map (and other stuff) on this worker.\n\n        It performs the following:\n            1. It updates the observation preprocessors and updates the policy_specs\n                with the postprocessed observation_spaces.\n            2. It updates the policy_specs with the complete algorithm_config (merged\n                with the policy_spec's config).\n            3. If needed it will update the self.marl_module_spec on this worker\n            3. It updates the policy map with the new policies\n            4. It updates the filter dict\n            5. It calls the on_create_policy() hook of the callbacks on the newly added\n                policies.\n\n        Args:\n            policy_dict: The policy dict to update the policy map with.\n            policy: The policy to update the policy map with.\n            policy_states: The policy states to update the policy map with.\n            single_agent_rl_module_spec: The SingleAgentRLModuleSpec to add to the\n                MultiAgentRLModuleSpec. If None, the config's\n                `get_default_rl_module_spec` method's output will be used to create\n                the policy with.\n        "
        updated_policy_dict = self._get_complete_policy_specs_dict(policy_dict)
        if self.config._enable_new_api_stack:
            spec = self.config.get_marl_module_spec(policy_dict=updated_policy_dict, single_agent_rl_module_spec=single_agent_rl_module_spec)
            if self.marl_module_spec is None:
                self.marl_module_spec = spec
            else:
                self.marl_module_spec.add_modules(spec.module_specs)
            updated_policy_dict = self._update_policy_dict_with_marl_module(updated_policy_dict)
        self._build_policy_map(policy_dict=updated_policy_dict, policy=policy, policy_states=policy_states)
        self._update_filter_dict(updated_policy_dict)
        if policy is None:
            self._call_callbacks_on_create_policy()
        if self.worker_index == 0:
            logger.info(f'Built policy map: {self.policy_map}')
            logger.info(f'Built preprocessor map: {self.preprocessors}')

    def _get_complete_policy_specs_dict(self, policy_dict: MultiAgentPolicyConfigDict) -> MultiAgentPolicyConfigDict:
        if False:
            return 10
        'Processes the policy dict and creates a new copy with the processed attrs.\n\n        This processes the observation_space and prepares them for passing to rl module\n        construction. It also merges the policy configs with the algorithm config.\n        During this processing, we will also construct the preprocessors dict.\n        '
        from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
        updated_policy_dict = copy.deepcopy(policy_dict)
        self.preprocessors = self.preprocessors or {}
        for (name, policy_spec) in sorted(updated_policy_dict.items()):
            logger.debug('Creating policy for {}'.format(name))
            if isinstance(policy_spec.config, AlgorithmConfig):
                merged_conf = policy_spec.config
            else:
                merged_conf: 'AlgorithmConfig' = self.config.copy(copy_frozen=False)
                merged_conf.update_from_dict(policy_spec.config or {})
            merged_conf.worker_index = self.worker_index
            obs_space = policy_spec.observation_space
            self.preprocessors[name] = None
            if self.preprocessing_enabled:
                preprocessor = ModelCatalog.get_preprocessor_for_space(obs_space, merged_conf.model, include_multi_binary=self.config.get('_enable_new_api_stack', False))
                if preprocessor is not None:
                    obs_space = preprocessor.observation_space
                if not merged_conf.enable_connectors:
                    self.preprocessors[name] = preprocessor
            policy_spec.config = merged_conf
            policy_spec.observation_space = obs_space
        return updated_policy_dict

    def _update_policy_dict_with_marl_module(self, policy_dict: MultiAgentPolicyConfigDict) -> MultiAgentPolicyConfigDict:
        if False:
            for i in range(10):
                print('nop')
        for (name, policy_spec) in policy_dict.items():
            policy_spec.config['__marl_module_spec'] = self.marl_module_spec
        return policy_dict

    def _build_policy_map(self, *, policy_dict: MultiAgentPolicyConfigDict, policy: Optional[Policy]=None, policy_states: Optional[Dict[PolicyID, PolicyState]]=None) -> None:
        if False:
            print('Hello World!')
        "Adds the given policy_dict to `self.policy_map`.\n\n        Args:\n            policy_dict: The MultiAgentPolicyConfigDict to be added to this\n                worker's PolicyMap.\n            policy: If the policy to add already exists, user can provide it here.\n            policy_states: Optional dict from PolicyIDs to PolicyStates to\n                restore the states of the policies being built.\n        "
        self.policy_map = self.policy_map or PolicyMap(capacity=self.config.policy_map_capacity, policy_states_are_swappable=self.config.policy_states_are_swappable)
        for (name, policy_spec) in sorted(policy_dict.items()):
            if policy is None:
                new_policy = create_policy_for_framework(policy_id=name, policy_class=get_tf_eager_cls_if_necessary(policy_spec.policy_class, policy_spec.config), merged_config=policy_spec.config, observation_space=policy_spec.observation_space, action_space=policy_spec.action_space, worker_index=self.worker_index, seed=self.seed)
            else:
                new_policy = policy
            if self.config.get('_enable_new_api_stack', False) and self.config.get('torch_compile_worker'):
                if self.config.framework_str != 'torch':
                    raise ValueError('Attempting to compile a non-torch RLModule.')
                rl_module = getattr(new_policy, 'model', None)
                if rl_module is not None:
                    compile_config = self.config.get_torch_compile_worker_config()
                    rl_module.compile(compile_config)
            self.policy_map[name] = new_policy
            restore_states = (policy_states or {}).get(name, None)
            if restore_states:
                new_policy.set_state(restore_states)

    def _update_filter_dict(self, policy_dict: MultiAgentPolicyConfigDict) -> None:
        if False:
            print('Hello World!')
        'Updates the filter dict for the given policy_dict.'
        for (name, policy_spec) in sorted(policy_dict.items()):
            new_policy = self.policy_map[name]
            if policy_spec.config.enable_connectors:
                if new_policy.agent_connectors is None or new_policy.action_connectors is None:
                    create_connectors_for_policy(new_policy, policy_spec.config)
                maybe_get_filters_for_syncing(self, name)
            else:
                filter_shape = tree.map_structure(lambda s: None if isinstance(s, (Discrete, MultiDiscrete)) else np.array(s.shape), new_policy.observation_space_struct)
                self.filters[name] = get_filter(policy_spec.config.observation_filter, filter_shape)

    def _call_callbacks_on_create_policy(self):
        if False:
            print('Hello World!')
        'Calls the on_create_policy callback for each policy in the policy map.'
        for (name, policy) in self.policy_map.items():
            self.callbacks.on_create_policy(policy_id=name, policy=policy)

    def _get_input_creator_from_config(self):
        if False:
            return 10

        def valid_module(class_path):
            if False:
                print('Hello World!')
            if isinstance(class_path, str) and (not os.path.isfile(class_path)) and ('.' in class_path):
                (module_path, class_name) = class_path.rsplit('.', 1)
                try:
                    spec = importlib.util.find_spec(module_path)
                    if spec is not None:
                        return True
                except (ModuleNotFoundError, ValueError):
                    print(f'module {module_path} not found while trying to get input {class_path}')
            return False
        if isinstance(self.config.input_, FunctionType):
            return self.config.input_
        elif self.config.input_ == 'sampler':
            return lambda ioctx: ioctx.default_sampler_input()
        elif self.config.input_ == 'dataset':
            assert self._ds_shards is not None
            return lambda ioctx: DatasetReader(self._ds_shards[self.worker_index], ioctx)
        elif isinstance(self.config.input_, dict):
            return lambda ioctx: ShuffledInput(MixedInput(self.config.input_, ioctx), self.config.shuffle_buffer_size)
        elif isinstance(self.config.input_, str) and registry_contains_input(self.config.input_):
            return registry_get_input(self.config.input_)
        elif 'd4rl' in self.config.input_:
            env_name = self.config.input_.split('.')[-1]
            return lambda ioctx: D4RLReader(env_name, ioctx)
        elif valid_module(self.config.input_):
            return lambda ioctx: ShuffledInput(from_config(self.config.input_, ioctx=ioctx))
        else:
            return lambda ioctx: ShuffledInput(JsonReader(self.config.input_, ioctx), self.config.shuffle_buffer_size)

    def _get_output_creator_from_config(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.config.output, FunctionType):
            return self.config.output
        elif self.config.output is None:
            return lambda ioctx: NoopOutput()
        elif self.config.output == 'dataset':
            return lambda ioctx: DatasetWriter(ioctx, compress_columns=self.config.output_compress_columns)
        elif self.config.output == 'logdir':
            return lambda ioctx: JsonWriter(ioctx.log_dir, ioctx, max_file_size=self.config.output_max_file_size, compress_columns=self.config.output_compress_columns)
        else:
            return lambda ioctx: JsonWriter(self.config.output, ioctx, max_file_size=self.config.output_max_file_size, compress_columns=self.config.output_compress_columns)

    def _get_make_sub_env_fn(self, env_creator, env_context, validate_env, env_wrapper, seed):
        if False:
            print('Hello World!')
        config = self.config

        def _make_sub_env_local(vector_index):
            if False:
                print('Hello World!')
            env_ctx = env_context.copy_with_overrides(vector_index=vector_index)
            env = env_creator(env_ctx)
            if not config.disable_env_checking:
                try:
                    check_env(env, config)
                except Exception as e:
                    logger.warning("We've added a module for checking environments that are used in experiments. Your env may not be set upcorrectly. You can disable env checking for now by setting `disable_env_checking` to True in your experiment config dictionary. You can run the environment checking module standalone by calling ray.rllib.utils.check_env(env).")
                    raise e
            if validate_env is not None:
                validate_env(env, env_ctx)
            env = env_wrapper(env)
            _update_env_seed_if_necessary(env, seed, env_context.worker_index, vector_index)
            return env
        if not env_context.remote:

            def _make_sub_env_remote(vector_index):
                if False:
                    print('Hello World!')
                sub_env = _make_sub_env_local(vector_index)
                self.callbacks.on_sub_environment_created(worker=self, sub_environment=sub_env, env_context=env_context.copy_with_overrides(worker_index=env_context.worker_index, vector_index=vector_index, remote=False))
                return sub_env
            return _make_sub_env_remote
        else:
            return _make_sub_env_local