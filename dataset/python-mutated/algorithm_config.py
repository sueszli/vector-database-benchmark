import copy
import logging
import math
import os
import sys
from typing import TYPE_CHECKING, Any, Callable, Container, Dict, Mapping, Optional, Tuple, Type, Union
from packaging import version
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.learner.learner import LearnerHyperparameters
from ray.rllib.core.learner.learner_group_config import LearnerGroupConfig, ModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import ModuleID, SingleAgentRLModuleSpec
from ray.rllib.core.learner.learner import TorchCompileWhatToCompile
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.atari_wrappers import is_atari
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils import deep_update, merge_dicts
from ray.rllib.utils.annotations import ExperimentalAPI, OverrideToImplementCustomLogic_CallToSuperRecommended
from ray.rllib.utils.deprecation import DEPRECATED_VALUE, Deprecated, deprecation_warning
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import NotProvided, from_config
from ray.rllib.utils.gym import convert_old_gym_space_to_gymnasium_space, try_import_gymnasium_and_gym
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.serialization import NOT_SERIALIZABLE, deserialize_type, serialize_type
from ray.rllib.utils.torch_utils import TORCH_COMPILE_REQUIRED_VERSION
from ray.rllib.utils.typing import AgentID, AlgorithmConfigDict, EnvConfigDict, EnvType, LearningRateOrSchedule, MultiAgentPolicyConfigDict, PartialAlgorithmConfigDict, PolicyID, ResultDict, SampleBatchType
from ray.tune.logger import Logger
from ray.tune.registry import get_trainable_cls
from ray.tune.result import TRIAL_INFO
from ray.tune.tune import _Config
(gym, old_gym) = try_import_gymnasium_and_gym()
Space = gym.Space
'TODO(jungong, sven): in "offline_data" we can potentially unify all input types\nunder input and input_config keys. E.g.\ninput: sample\ninput_config {\nenv: CartPole-v1\n}\nor:\ninput: json_reader\ninput_config {\npath: /tmp/\n}\nor:\ninput: dataset\ninput_config {\nformat: parquet\npath: /tmp/\n}\n'
if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.core.learner import Learner
    from ray.rllib.evaluation.episode import Episode as OldEpisode
logger = logging.getLogger(__name__)

def _check_rl_module_spec(module_spec: ModuleSpec) -> None:
    if False:
        return 10
    if not isinstance(module_spec, (SingleAgentRLModuleSpec, MultiAgentRLModuleSpec)):
        raise ValueError(f'rl_module_spec must be an instance of SingleAgentRLModuleSpec or MultiAgentRLModuleSpec.Got {type(module_spec)} instead.')

class AlgorithmConfig(_Config):
    """A RLlib AlgorithmConfig builds an RLlib Algorithm from a given configuration.

    .. testcode::

        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.algorithms.callbacks import MemoryTrackingCallbacks
        # Construct a generic config object, specifying values within different
        # sub-categories, e.g. "training".
        config = (PPOConfig().training(gamma=0.9, lr=0.01)
                .environment(env="CartPole-v1")
                .resources(num_gpus=0)
                .rollouts(num_rollout_workers=0)
                .callbacks(MemoryTrackingCallbacks)
            )
        # A config object can be used to construct the respective Algorithm.
        rllib_algo = config.build()

    .. testcode::

        from ray.rllib.algorithms.ppo import PPOConfig
        from ray import tune
        # In combination with a tune.grid_search:
        config = PPOConfig()
        config.training(lr=tune.grid_search([0.01, 0.001]))
        # Use `to_dict()` method to get the legacy plain python config dict
        # for usage with `tune.Tuner().fit()`.
        tune.Tuner("PPO", param_space=config.to_dict())
    """

    @staticmethod
    def DEFAULT_POLICY_MAPPING_FN(aid, episode, worker, **kwargs):
        if False:
            print('Hello World!')
        return DEFAULT_POLICY_ID

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AlgorithmConfig':
        if False:
            return 10
        'Creates an AlgorithmConfig from a legacy python config dict.\n\n        .. testcode::\n\n            from ray.rllib.algorithms.ppo.ppo import PPOConfig\n            # pass a RLlib config dict\n            ppo_config = PPOConfig.from_dict({})\n            ppo = ppo_config.build(env="Pendulum-v1")\n\n        Args:\n            config_dict: The legacy formatted python config dict for some algorithm.\n\n        Returns:\n             A new AlgorithmConfig object that matches the given python config dict.\n        '
        config_obj = cls()
        config_dict.pop('_is_frozen', None)
        config_obj.update_from_dict(config_dict)
        return config_obj

    @classmethod
    def overrides(cls, **kwargs):
        if False:
            print('Hello World!')
        'Generates and validates a set of config key/value pairs (passed via kwargs).\n\n        Validation whether given config keys are valid is done immediately upon\n        construction (by comparing against the properties of a default AlgorithmConfig\n        object of this class).\n        Allows combination with a full AlgorithmConfig object to yield a new\n        AlgorithmConfig object.\n\n        Used anywhere, we would like to enable the user to only define a few config\n        settings that would change with respect to some main config, e.g. in multi-agent\n        setups and evaluation configs.\n\n        .. testcode::\n\n            from ray.rllib.algorithms.ppo import PPOConfig\n            from ray.rllib.policy.policy import PolicySpec\n            config = (\n                PPOConfig()\n                .multi_agent(\n                    policies={\n                        "pol0": PolicySpec(config=PPOConfig.overrides(lambda_=0.95))\n                    },\n                )\n            )\n\n\n        .. testcode::\n\n            from ray.rllib.algorithms.algorithm_config import AlgorithmConfig\n            from ray.rllib.algorithms.ppo import PPOConfig\n            config = (\n                PPOConfig()\n                .evaluation(\n                    evaluation_num_workers=1,\n                    evaluation_interval=1,\n                    evaluation_config=AlgorithmConfig.overrides(explore=False),\n                )\n            )\n\n        Returns:\n            A dict mapping valid config property-names to values.\n\n        Raises:\n            KeyError: In case a non-existing property name (kwargs key) is being\n            passed in. Valid property names are taken from a default AlgorithmConfig\n            object of `cls`.\n        '
        default_config = cls()
        config_overrides = {}
        for (key, value) in kwargs.items():
            if not hasattr(default_config, key):
                raise KeyError(f'Invalid property name {key} for config class {cls.__name__}!')
            key = cls._translate_special_keys(key, warn_deprecated=True)
            config_overrides[key] = value
        return config_overrides

    def __init__(self, algo_class=None):
        if False:
            for i in range(10):
                print('nop')
        self.algo_class = algo_class
        self.extra_python_environs_for_driver = {}
        self.extra_python_environs_for_worker = {}
        self.num_gpus = 0
        self.num_cpus_per_worker = 1
        self.num_gpus_per_worker = 0
        self._fake_gpus = False
        self.num_cpus_for_local_worker = 1
        self.num_learner_workers = 0
        self.num_gpus_per_learner_worker = 0
        self.num_cpus_per_learner_worker = 1
        self.local_gpu_idx = 0
        self.custom_resources_per_worker = {}
        self.placement_strategy = 'PACK'
        self.framework_str = 'torch'
        self.eager_tracing = True
        self.eager_max_retraces = 20
        self.tf_session_args = {'intra_op_parallelism_threads': 2, 'inter_op_parallelism_threads': 2, 'gpu_options': {'allow_growth': True}, 'log_device_placement': False, 'device_count': {'CPU': 1}, 'allow_soft_placement': True}
        self.local_tf_session_args = {'intra_op_parallelism_threads': 8, 'inter_op_parallelism_threads': 8}
        self.torch_compile_learner = False
        self.torch_compile_learner_what_to_compile = TorchCompileWhatToCompile.FORWARD_TRAIN
        self.torch_compile_learner_dynamo_backend = 'aot_eager' if sys.platform == 'darwin' else 'inductor'
        self.torch_compile_learner_dynamo_mode = None
        self.torch_compile_worker = False
        self.torch_compile_worker_dynamo_backend = 'aot_eager' if sys.platform == 'darwin' else 'onnxrt'
        self.torch_compile_worker_dynamo_mode = None
        self.env = None
        self.env_config = {}
        self.observation_space = None
        self.action_space = None
        self.env_task_fn = None
        self.render_env = False
        self.clip_rewards = None
        self.normalize_actions = True
        self.clip_actions = False
        self.disable_env_checking = False
        self.auto_wrap_old_gym_envs = True
        self.action_mask_key = 'action_mask'
        self._is_atari = None
        self.env_runner_cls = None
        self.num_rollout_workers = 0
        self.num_envs_per_worker = 1
        self.sample_collector = SimpleListCollector
        self.create_env_on_local_worker = False
        self.sample_async = False
        self.enable_connectors = True
        self.update_worker_filter_stats = True
        self.use_worker_filter_stats = True
        self.rollout_fragment_length = 200
        self.batch_mode = 'truncate_episodes'
        self.remote_worker_envs = False
        self.remote_env_batch_wait_ms = 0
        self.validate_workers_after_construction = True
        self.preprocessor_pref = 'deepmind'
        self.observation_filter = 'NoFilter'
        self.compress_observations = False
        self.enable_tf1_exec_eagerly = False
        self.sampler_perf_stats_ema_coef = None
        self.gamma = 0.99
        self.lr = 0.001
        self.grad_clip = None
        self.grad_clip_by = 'global_norm'
        self.train_batch_size = 32
        try:
            self.model = copy.deepcopy(MODEL_DEFAULTS)
        except AttributeError:
            pass
        self.optimizer = {}
        self.max_requests_in_flight_per_sampler_worker = 2
        self._learner_class = None
        self.callbacks_class = DefaultCallbacks
        self.explore = True
        self.exploration_config = {}
        self.policies = {DEFAULT_POLICY_ID: PolicySpec()}
        self.algorithm_config_overrides_per_module = {}
        self.policy_map_capacity = 100
        self.policy_mapping_fn = self.DEFAULT_POLICY_MAPPING_FN
        self.policies_to_train = None
        self.policy_states_are_swappable = False
        self.observation_fn = None
        self.count_steps_by = 'env_steps'
        self.input_ = 'sampler'
        self.input_config = {}
        self.actions_in_input_normalized = False
        self.postprocess_inputs = False
        self.shuffle_buffer_size = 0
        self.output = None
        self.output_config = {}
        self.output_compress_columns = ['obs', 'new_obs']
        self.output_max_file_size = 64 * 1024 * 1024
        self.offline_sampling = False
        self.evaluation_interval = None
        self.evaluation_duration = 10
        self.evaluation_duration_unit = 'episodes'
        self.evaluation_sample_timeout_s = 180.0
        self.evaluation_parallel_to_training = False
        self.evaluation_config = None
        self.off_policy_estimation_methods = {}
        self.ope_split_batch_by_episode = True
        self.evaluation_num_workers = 0
        self.custom_evaluation_function = None
        self.always_attach_evaluation_results = False
        self.enable_async_evaluation = False
        self.in_evaluation = False
        self.sync_filters_on_rollout_workers_timeout_s = 60.0
        self.keep_per_episode_custom_metrics = False
        self.metrics_episode_collection_timeout_s = 60.0
        self.metrics_num_episodes_for_smoothing = 100
        self.min_time_s_per_iteration = None
        self.min_train_timesteps_per_iteration = 0
        self.min_sample_timesteps_per_iteration = 0
        self.export_native_model_files = False
        self.checkpoint_trainable_policies_only = False
        self.logger_creator = None
        self.logger_config = None
        self.log_level = 'WARN'
        self.log_sys_usage = True
        self.fake_sampler = False
        self.seed = None
        self.ignore_worker_failures = False
        self.recreate_failed_workers = False
        self.max_num_worker_restarts = 1000
        self.delay_between_worker_restarts_s = 60.0
        self.restart_failed_sub_environments = False
        self.num_consecutive_worker_failures_tolerance = 100
        self.worker_health_probe_timeout_s = 60
        self.worker_restore_timeout_s = 1800
        self._rl_module_spec = None
        self.__prior_exploration_config = None
        self._enable_new_api_stack = False
        self._tf_policy_handles_more_than_one_loss = False
        self._disable_preprocessor_api = False
        self._disable_action_flattening = False
        self._disable_execution_plan_api = True
        self._disable_initialize_loss_from_dummy_batch = False
        self._is_frozen = False
        self.simple_optimizer = DEPRECATED_VALUE
        self.monitor = DEPRECATED_VALUE
        self.evaluation_num_episodes = DEPRECATED_VALUE
        self.metrics_smoothing_episodes = DEPRECATED_VALUE
        self.timesteps_per_iteration = DEPRECATED_VALUE
        self.min_iter_time_s = DEPRECATED_VALUE
        self.collect_metrics_timeout = DEPRECATED_VALUE
        self.min_time_s_per_reporting = DEPRECATED_VALUE
        self.min_train_timesteps_per_reporting = DEPRECATED_VALUE
        self.min_sample_timesteps_per_reporting = DEPRECATED_VALUE
        self.input_evaluation = DEPRECATED_VALUE
        self.policy_map_cache = DEPRECATED_VALUE
        self.worker_cls = DEPRECATED_VALUE
        self.synchronize_filters = DEPRECATED_VALUE
        self.buffer_size = DEPRECATED_VALUE
        self.prioritized_replay = DEPRECATED_VALUE
        self.learning_starts = DEPRECATED_VALUE
        self.replay_batch_size = DEPRECATED_VALUE
        self.replay_sequence_length = None
        self.replay_mode = DEPRECATED_VALUE
        self.prioritized_replay_alpha = DEPRECATED_VALUE
        self.prioritized_replay_beta = DEPRECATED_VALUE
        self.prioritized_replay_eps = DEPRECATED_VALUE
        self.min_time_s_per_reporting = DEPRECATED_VALUE
        self.min_train_timesteps_per_reporting = DEPRECATED_VALUE
        self.min_sample_timesteps_per_reporting = DEPRECATED_VALUE

    def to_dict(self) -> AlgorithmConfigDict:
        if False:
            i = 10
            return i + 15
        'Converts all settings into a legacy config dict for backward compatibility.\n\n        Returns:\n            A complete AlgorithmConfigDict, usable in backward-compatible Tune/RLlib\n            use cases, e.g. w/ `tune.Tuner().fit()`.\n        '
        config = copy.deepcopy(vars(self))
        config.pop('algo_class')
        config.pop('_is_frozen')
        if 'lambda_' in config:
            assert hasattr(self, 'lambda_')
            config['lambda'] = getattr(self, 'lambda_')
            config.pop('lambda_')
        if 'input_' in config:
            assert hasattr(self, 'input_')
            config['input'] = getattr(self, 'input_')
            config.pop('input_')
        if 'policies' in config and isinstance(config['policies'], dict):
            policies_dict = {}
            for (policy_id, policy_spec) in config.pop('policies').items():
                if isinstance(policy_spec, PolicySpec):
                    policies_dict[policy_id] = (policy_spec.policy_class, policy_spec.observation_space, policy_spec.action_space, policy_spec.config)
                else:
                    policies_dict[policy_id] = policy_spec
            config['policies'] = policies_dict
        config['callbacks'] = config.pop('callbacks_class', DefaultCallbacks)
        config['create_env_on_driver'] = config.pop('create_env_on_local_worker', 1)
        config['custom_eval_function'] = config.pop('custom_evaluation_function', None)
        config['framework'] = config.pop('framework_str', None)
        config['num_cpus_for_driver'] = config.pop('num_cpus_for_local_worker', 1)
        config['num_workers'] = config.pop('num_rollout_workers', 0)
        for dep_k in ['monitor', 'evaluation_num_episodes', 'metrics_smoothing_episodes', 'timesteps_per_iteration', 'min_iter_time_s', 'collect_metrics_timeout', 'buffer_size', 'prioritized_replay', 'learning_starts', 'replay_batch_size', 'replay_mode', 'prioritized_replay_alpha', 'prioritized_replay_beta', 'prioritized_replay_eps', 'min_time_s_per_reporting', 'min_train_timesteps_per_reporting', 'min_sample_timesteps_per_reporting', 'input_evaluation']:
            if config.get(dep_k) == DEPRECATED_VALUE:
                config.pop(dep_k, None)
        return config

    def update_from_dict(self, config_dict: PartialAlgorithmConfigDict) -> 'AlgorithmConfig':
        if False:
            i = 10
            return i + 15
        'Modifies this AlgorithmConfig via the provided python config dict.\n\n        Warns if `config_dict` contains deprecated keys.\n        Silently sets even properties of `self` that do NOT exist. This way, this method\n        may be used to configure custom Policies which do not have their own specific\n        AlgorithmConfig classes, e.g.\n        `ray.rllib.examples.policy.random_policy::RandomPolicy`.\n\n        Args:\n            config_dict: The old-style python config dict (PartialAlgorithmConfigDict)\n                to use for overriding some properties defined in there.\n\n        Returns:\n            This updated AlgorithmConfig object.\n        '
        eval_call = {}
        if '_enable_new_api_stack' in config_dict:
            self.experimental(_enable_new_api_stack=config_dict['_enable_new_api_stack'])
        for (key, value) in config_dict.items():
            key = self._translate_special_keys(key, warn_deprecated=False)
            if key == TRIAL_INFO:
                continue
            if key == '_enable_new_api_stack':
                continue
            elif key == 'multiagent':
                kwargs = {k: value[k] for k in ['policies', 'policy_map_capacity', 'policy_mapping_fn', 'policies_to_train', 'policy_states_are_swappable', 'observation_fn', 'count_steps_by'] if k in value}
                self.multi_agent(**kwargs)
            elif key == 'callbacks_class' and value != NOT_SERIALIZABLE:
                if isinstance(value, str):
                    value = deserialize_type(value, error=True)
                self.callbacks(callbacks_class=value)
            elif key == 'env_config':
                self.environment(env_config=value)
            elif key.startswith('evaluation_'):
                eval_call[key] = value
            elif key == 'exploration_config':
                if config_dict.get('_enable_new_api_stack', False):
                    self.exploration_config = value
                    continue
                if isinstance(value, dict) and 'type' in value:
                    value['type'] = deserialize_type(value['type'])
                self.exploration(exploration_config=value)
            elif key == 'model':
                if isinstance(value, dict) and value.get('custom_model'):
                    value['custom_model'] = deserialize_type(value['custom_model'])
                self.training(**{key: value})
            elif key == 'optimizer':
                self.training(**{key: value})
            elif key == 'replay_buffer_config':
                if isinstance(value, dict) and 'type' in value:
                    value['type'] = deserialize_type(value['type'])
                self.training(**{key: value})
            elif key == 'sample_collector':
                value = deserialize_type(value)
                self.rollouts(sample_collector=value)
            else:
                setattr(self, key, value)
        self.evaluation(**eval_call)
        return self

    def serialize(self) -> Mapping[str, Any]:
        if False:
            print('Hello World!')
        "Returns a mapping from str to JSON'able values representing this config.\n\n        The resulting values will not have any code in them.\n        Classes (such as `callbacks_class`) will be converted to their full\n        classpath, e.g. `ray.rllib.algorithms.callbacks.DefaultCallbacks`.\n        Actual code such as lambda functions will be written as their source\n        code (str) plus any closure information for properly restoring the\n        code inside the AlgorithmConfig object made from the returned dict data.\n        Dataclass objects get converted to dicts.\n\n        Returns:\n            A mapping from str to JSON'able values.\n        "
        config = self.to_dict()
        return self._serialize_dict(config)

    def copy(self, copy_frozen: Optional[bool]=None) -> 'AlgorithmConfig':
        if False:
            return 10
        'Creates a deep copy of this config and (un)freezes if necessary.\n\n        Args:\n            copy_frozen: Whether the created deep copy will be frozen or not. If None,\n                keep the same frozen status that `self` currently has.\n\n        Returns:\n            A deep copy of `self` that is (un)frozen.\n        '
        cp = copy.deepcopy(self)
        if copy_frozen is True:
            cp.freeze()
        elif copy_frozen is False:
            cp._is_frozen = False
            if isinstance(cp.evaluation_config, AlgorithmConfig):
                cp.evaluation_config._is_frozen = False
        return cp

    def freeze(self) -> None:
        if False:
            i = 10
            return i + 15
        'Freezes this config object, such that no attributes can be set anymore.\n\n        Algorithms should use this method to make sure that their config objects\n        remain read-only after this.\n        '
        if self._is_frozen:
            return
        self._is_frozen = True
        if isinstance(self.evaluation_config, AlgorithmConfig):
            self.evaluation_config.freeze()

    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def validate(self) -> None:
        if False:
            return 10
        'Validates all values in this config.'
        if self.evaluation_interval and self.env_runner_cls is not None and (not issubclass(self.env_runner_cls, RolloutWorker)) and (not self.enable_async_evaluation):
            raise ValueError(f"When using an EnvRunner class that's not a subclass of `RolloutWorker`(yours is {self.env_runner_cls.__name__}), `config.enable_async_evaluation` must be set to True! Call `config.evaluation(enable_async_evaluation=True) on your config object to fix this problem.")
        if not (isinstance(self.rollout_fragment_length, int) and self.rollout_fragment_length > 0 or self.rollout_fragment_length == 'auto'):
            raise ValueError("`rollout_fragment_length` must be int >0 or 'auto'!")
        if self.batch_mode not in ['truncate_episodes', 'complete_episodes']:
            raise ValueError('`config.batch_mode` must be one of [truncate_episodes|complete_episodes]! Got {}'.format(self.batch_mode))
        if self.preprocessor_pref not in ['rllib', 'deepmind', None]:
            raise ValueError("`config.preprocessor_pref` must be either 'rllib', 'deepmind' or None!")
        if self.num_envs_per_worker <= 0:
            raise ValueError(f'`num_envs_per_worker` ({self.num_envs_per_worker}) must be larger than 0!')
        (_tf1, _tf, _tfv) = (None, None, None)
        _torch = None
        if self.framework_str not in {'tf', 'tf2'} and self.framework_str != 'torch':
            return
        elif self.framework_str in {'tf', 'tf2'}:
            (_tf1, _tf, _tfv) = try_import_tf()
        else:
            (_torch, _) = try_import_torch()
        if self.framework_str == 'tf' and self._enable_new_api_stack:
            raise ValueError("Cannot use `framework=tf` with the new API stack! Either switch to tf2 via `config.framework('tf2')` OR disable the new API stack via `config.experimental(_enable_new_api_stack=False)`.")
        if _torch is not None and self.framework_str == 'torch' and (version.parse(_torch.__version__) < TORCH_COMPILE_REQUIRED_VERSION) and (self.torch_compile_learner or self.torch_compile_worker):
            raise ValueError('torch.compile is only supported from torch 2.0.0')
        self._check_if_correct_nn_framework_installed(_tf1, _tf, _torch)
        self._resolve_tf_settings(_tf1, _tfv)
        if isinstance(self.policies_to_train, (list, set, tuple)):
            for pid in self.policies_to_train:
                if pid not in self.policies:
                    raise ValueError(f'`config.multi_agent(policies_to_train=..)` contains policy ID ({pid}) that was not defined in `config.multi_agent(policies=..)`!')
        if self.enable_async_evaluation and self.custom_evaluation_function:
            raise ValueError('`config.custom_evaluation_function` not supported in combination with `enable_async_evaluation=True` config setting!')
        if self.evaluation_num_workers > 0 and (not self.evaluation_interval):
            logger.warning(f'You have specified {self.evaluation_num_workers} evaluation workers, but your `evaluation_interval` is None! Therefore, evaluation will not occur automatically with each call to `Algorithm.train()`. Instead, you will have to call `Algorithm.evaluate()` manually in order to trigger an evaluation run.')
        elif self.evaluation_num_workers == 0 and self.evaluation_parallel_to_training:
            raise ValueError('`evaluation_parallel_to_training` can only be done if `evaluation_num_workers` > 0! Try setting `config.evaluation_parallel_to_training` to False.')
        if self.evaluation_duration == 'auto':
            if not self.evaluation_parallel_to_training:
                raise ValueError('`evaluation_duration=auto` not supported for `evaluation_parallel_to_training=False`!')
        elif not isinstance(self.evaluation_duration, int) or self.evaluation_duration <= 0:
            raise ValueError(f'`evaluation_duration` ({self.evaluation_duration}) must be an int and >0!')
        if self._disable_preprocessor_api is True:
            self.model['_disable_preprocessor_api'] = True
        if self._disable_action_flattening is True:
            self.model['_disable_action_flattening'] = True
        if self.model.get('custom_preprocessor'):
            deprecation_warning(old="AlgorithmConfig.training(model={'custom_preprocessor': ...})", help='Custom preprocessors are deprecated, since they sometimes conflict with the built-in preprocessors for handling complex observation spaces. Please use wrapper classes around your environment instead.', error=True)
        if not self.enable_connectors and self._enable_new_api_stack:
            raise ValueError('The new API stack (RLModule and Learner APIs) only works with connectors! Please enable connectors via `config.rollouts(enable_connectors=True)`.')
        if self._rl_module_spec is not None and (not self._enable_new_api_stack):
            logger.warning('You have setup a RLModuleSpec (via calling `config.rl_module(...)`), but have not enabled the new API stack. To enable it, call `config.experimental(_enable_new_api_stack=True)`.')
        if self._enable_new_api_stack:
            Scheduler.validate(fixed_value_or_schedule=self.lr, setting_name='lr', description='learning rate')
        if self._learner_class is not None and (not self._enable_new_api_stack):
            logger.warning(f'You specified a custom Learner class (via `AlgorithmConfig.training(learner_class={self._learner_class})`, but have the new API stack disabled. You need to enable it via `AlgorithmConfig.experimental(_enable_new_api_stack=True)`.')
        if self.num_cpus_per_learner_worker > 1 and self.num_gpus_per_learner_worker > 0:
            raise ValueError('Cannot set both `num_cpus_per_learner_worker` and  `num_gpus_per_learner_worker` > 0! Users must set one or the other due to issues with placement group fragmentation. See https://github.com/ray-project/ray/issues/35409 for more details.')
        if bool(os.environ.get('RLLIB_ENABLE_RL_MODULE', False)):
            self.experimental(_enable_new_api_stack=True)
            self.enable_connectors = True
        if self.grad_clip_by not in ['value', 'norm', 'global_norm']:
            raise ValueError(f"`grad_clip_by` ({self.grad_clip_by}) must be one of: 'value', 'norm', or 'global_norm'!")
        if self.simple_optimizer is True:
            pass
        elif not self._enable_new_api_stack and self.num_gpus > 1:
            if self.framework_str == 'tf2' and type(self).__name__ != 'AlphaStar':
                raise ValueError(f'`num_gpus` > 1 not supported yet for framework={self.framework_str}!')
            elif self.simple_optimizer is True:
                raise ValueError('Cannot use `simple_optimizer` if `num_gpus` > 1! Consider not setting `simple_optimizer` in your config.')
            self.simple_optimizer = False
        elif self.simple_optimizer == DEPRECATED_VALUE:
            if self.framework_str not in ['tf', 'torch']:
                self.simple_optimizer = True
            elif self.is_multi_agent():
                from ray.rllib.policy.dynamic_tf_policy import DynamicTFPolicy
                from ray.rllib.policy.torch_policy import TorchPolicy
                default_policy_cls = None
                if self.algo_class:
                    default_policy_cls = self.algo_class.get_default_policy_class(self)
                policies = self.policies
                policy_specs = [PolicySpec(*spec) if isinstance(spec, (tuple, list)) else spec for spec in policies.values()] if isinstance(policies, dict) else [PolicySpec() for _ in policies]
                if any(((spec.policy_class or default_policy_cls) is None or not issubclass(spec.policy_class or default_policy_cls, (DynamicTFPolicy, TorchPolicy)) for spec in policy_specs)):
                    self.simple_optimizer = True
                else:
                    self.simple_optimizer = False
            else:
                self.simple_optimizer = False
        elif self.simple_optimizer is False:
            if self.framework_str == 'tf2':
                raise ValueError(f'`simple_optimizer=False` not supported for config.framework({self.framework_str})!')
        if self.input_ == 'sampler' and self.off_policy_estimation_methods:
            raise ValueError('Off-policy estimation methods can only be used if the input is a dataset. We currently do not support applying off_policy_esitmation method on a sampler input.')
        if self.input_ == 'dataset':
            self.input_config['num_cpus_per_read_task'] = self.num_cpus_per_worker
            if self.in_evaluation:
                self.input_config['parallelism'] = self.evaluation_num_workers or 1
            else:
                self.input_config['parallelism'] = self.num_rollout_workers or 1
        if self._enable_new_api_stack:
            not_compatible_w_rlm_msg = 'Cannot use `{}` option with the new API stack (RLModule and Learner APIs)! `{}` is part of the ModelV2 API and Policy API, which are not compatible with the new API stack. You can either deactivate the new stack via `config.experimental( _enable_new_api_stack=False)`,or use the new stack (incl. RLModule API) and implement your custom model as an RLModule.'
            if self.model['custom_model'] is not None:
                raise ValueError(not_compatible_w_rlm_msg.format('custom_model', 'custom_model'))
            if self.model['custom_model_config'] != {}:
                raise ValueError(not_compatible_w_rlm_msg.format('custom_model_config', 'custom_model_config'))
            if self.exploration_config:
                raise ValueError('When RLModule API are enabled, exploration_config can not be set. If you want to implement custom exploration behaviour, please modify the `forward_exploration` method of the RLModule at hand. On configs that have a default exploration config, this must be done with `config.exploration_config={}`.')
        if self.num_learner_workers == 0 and self.num_gpus_per_worker > 1:
            raise ValueError('num_gpus_per_worker must be 0 (cpu) or 1 (gpu) when using local mode (i.e. num_learner_workers = 0)')

    def build(self, env: Optional[Union[str, EnvType]]=None, logger_creator: Optional[Callable[[], Logger]]=None, use_copy: bool=True) -> 'Algorithm':
        if False:
            for i in range(10):
                print('nop')
        'Builds an Algorithm from this AlgorithmConfig (or a copy thereof).\n\n        Args:\n            env: Name of the environment to use (e.g. a gym-registered str),\n                a full class path (e.g.\n                "ray.rllib.examples.env.random_env.RandomEnv"), or an Env\n                class directly. Note that this arg can also be specified via\n                the "env" key in `config`.\n            logger_creator: Callable that creates a ray.tune.Logger\n                object. If unspecified, a default logger is created.\n            use_copy: Whether to deepcopy `self` and pass the copy to the Algorithm\n                (instead of `self`) as config. This is useful in case you would like to\n                recycle the same AlgorithmConfig over and over, e.g. in a test case, in\n                which we loop over different DL-frameworks.\n\n        Returns:\n            A ray.rllib.algorithms.algorithm.Algorithm object.\n        '
        if env is not None:
            self.env = env
            if self.evaluation_config is not None:
                self.evaluation_config['env'] = env
        if logger_creator is not None:
            self.logger_creator = logger_creator
        algo_class = self.algo_class
        if isinstance(self.algo_class, str):
            algo_class = get_trainable_cls(self.algo_class)
        return algo_class(config=self if not use_copy else copy.deepcopy(self), logger_creator=self.logger_creator)

    def python_environment(self, *, extra_python_environs_for_driver: Optional[dict]=NotProvided, extra_python_environs_for_worker: Optional[dict]=NotProvided) -> 'AlgorithmConfig':
        if False:
            for i in range(10):
                print('nop')
        'Sets the config\'s python environment settings.\n\n        Args:\n            extra_python_environs_for_driver: Any extra python env vars to set in the\n                algorithm\'s process, e.g., {"OMP_NUM_THREADS": "16"}.\n            extra_python_environs_for_worker: The extra python environments need to set\n                for worker processes.\n\n        Returns:\n            This updated AlgorithmConfig object.\n        '
        if extra_python_environs_for_driver is not NotProvided:
            self.extra_python_environs_for_driver = extra_python_environs_for_driver
        if extra_python_environs_for_worker is not NotProvided:
            self.extra_python_environs_for_worker = extra_python_environs_for_worker
        return self

    def resources(self, *, num_gpus: Optional[Union[float, int]]=NotProvided, _fake_gpus: Optional[bool]=NotProvided, num_cpus_per_worker: Optional[Union[float, int]]=NotProvided, num_gpus_per_worker: Optional[Union[float, int]]=NotProvided, num_cpus_for_local_worker: Optional[int]=NotProvided, num_learner_workers: Optional[int]=NotProvided, num_cpus_per_learner_worker: Optional[Union[float, int]]=NotProvided, num_gpus_per_learner_worker: Optional[Union[float, int]]=NotProvided, local_gpu_idx: Optional[int]=NotProvided, custom_resources_per_worker: Optional[dict]=NotProvided, placement_strategy: Optional[str]=NotProvided) -> 'AlgorithmConfig':
        if False:
            i = 10
            return i + 15
        'Specifies resources allocated for an Algorithm and its ray actors/workers.\n\n        Args:\n            num_gpus: Number of GPUs to allocate to the algorithm process.\n                Note that not all algorithms can take advantage of GPUs.\n                Support for multi-GPU is currently only available for\n                tf-[PPO/IMPALA/DQN/PG]. This can be fractional (e.g., 0.3 GPUs).\n            _fake_gpus: Set to True for debugging (multi-)?GPU funcitonality on a\n                CPU machine. GPU towers will be simulated by graphs located on\n                CPUs in this case. Use `num_gpus` to test for different numbers of\n                fake GPUs.\n            num_cpus_per_worker: Number of CPUs to allocate per worker.\n            num_gpus_per_worker: Number of GPUs to allocate per worker. This can be\n                fractional. This is usually needed only if your env itself requires a\n                GPU (i.e., it is a GPU-intensive video game), or model inference is\n                unusually expensive.\n            num_learner_workers: Number of workers used for training. A value of 0\n                means training will take place on a local worker on head node CPUs or 1\n                GPU (determined by `num_gpus_per_learner_worker`). For multi-gpu\n                training, set number of workers greater than 1 and set\n                `num_gpus_per_learner_worker` accordingly (e.g. 4 GPUs total, and model\n                needs 2 GPUs: `num_learner_workers = 2` and\n                `num_gpus_per_learner_worker = 2`)\n            num_cpus_per_learner_worker: Number of CPUs allocated per Learner worker.\n                Only necessary for custom processing pipeline inside each Learner\n                requiring multiple CPU cores. Ignored if `num_learner_workers = 0`.\n            num_gpus_per_learner_worker: Number of GPUs allocated per worker. If\n                `num_learner_workers = 0`, any value greater than 0 will run the\n                training on a single GPU on the head node, while a value of 0 will run\n                the training on head node CPU cores. If num_gpus_per_learner_worker is\n                set, then num_cpus_per_learner_worker cannot be set.\n            local_gpu_idx: if num_gpus_per_worker > 0, and num_workers<2, then this gpu\n                index will be used for training. This is an index into the available\n                cuda devices. For example if os.environ["CUDA_VISIBLE_DEVICES"] = "1"\n                then a local_gpu_idx of 0 will use the gpu with id 1 on the node.\n            custom_resources_per_worker: Any custom Ray resources to allocate per\n                worker.\n            num_cpus_for_local_worker: Number of CPUs to allocate for the algorithm.\n                Note: this only takes effect when running in Tune. Otherwise,\n                the algorithm runs in the main program (driver).\n            custom_resources_per_worker: Any custom Ray resources to allocate per\n                worker.\n            placement_strategy: The strategy for the placement group factory returned by\n                `Algorithm.default_resource_request()`. A PlacementGroup defines, which\n                devices (resources) should always be co-located on the same node.\n                For example, an Algorithm with 2 rollout workers, running with\n                num_gpus=1 will request a placement group with the bundles:\n                [{"gpu": 1, "cpu": 1}, {"cpu": 1}, {"cpu": 1}], where the first bundle\n                is for the driver and the other 2 bundles are for the two workers.\n                These bundles can now be "placed" on the same or different\n                nodes depending on the value of `placement_strategy`:\n                "PACK": Packs bundles into as few nodes as possible.\n                "SPREAD": Places bundles across distinct nodes as even as possible.\n                "STRICT_PACK": Packs bundles into one node. The group is not allowed\n                to span multiple nodes.\n                "STRICT_SPREAD": Packs bundles across distinct nodes.\n\n        Returns:\n            This updated AlgorithmConfig object.\n        '
        if num_gpus is not NotProvided:
            self.num_gpus = num_gpus
        if _fake_gpus is not NotProvided:
            self._fake_gpus = _fake_gpus
        if num_cpus_per_worker is not NotProvided:
            self.num_cpus_per_worker = num_cpus_per_worker
        if num_gpus_per_worker is not NotProvided:
            self.num_gpus_per_worker = num_gpus_per_worker
        if num_cpus_for_local_worker is not NotProvided:
            self.num_cpus_for_local_worker = num_cpus_for_local_worker
        if custom_resources_per_worker is not NotProvided:
            self.custom_resources_per_worker = custom_resources_per_worker
        if placement_strategy is not NotProvided:
            self.placement_strategy = placement_strategy
        if num_learner_workers is not NotProvided:
            self.num_learner_workers = num_learner_workers
        if num_cpus_per_learner_worker is not NotProvided:
            self.num_cpus_per_learner_worker = num_cpus_per_learner_worker
        if num_gpus_per_learner_worker is not NotProvided:
            self.num_gpus_per_learner_worker = num_gpus_per_learner_worker
        if local_gpu_idx is not NotProvided:
            self.local_gpu_idx = local_gpu_idx
        return self

    def framework(self, framework: Optional[str]=NotProvided, *, eager_tracing: Optional[bool]=NotProvided, eager_max_retraces: Optional[int]=NotProvided, tf_session_args: Optional[Dict[str, Any]]=NotProvided, local_tf_session_args: Optional[Dict[str, Any]]=NotProvided, torch_compile_learner: Optional[bool]=NotProvided, torch_compile_learner_what_to_compile: Optional[str]=NotProvided, torch_compile_learner_dynamo_mode: Optional[str]=NotProvided, torch_compile_learner_dynamo_backend: Optional[str]=NotProvided, torch_compile_worker: Optional[bool]=NotProvided, torch_compile_worker_dynamo_backend: Optional[str]=NotProvided, torch_compile_worker_dynamo_mode: Optional[str]=NotProvided) -> 'AlgorithmConfig':
        if False:
            i = 10
            return i + 15
        "Sets the config's DL framework settings.\n\n        Args:\n            framework: torch: PyTorch; tf2: TensorFlow 2.x (eager execution or traced\n                if eager_tracing=True); tf: TensorFlow (static-graph);\n            eager_tracing: Enable tracing in eager mode. This greatly improves\n                performance (speedup ~2x), but makes it slightly harder to debug\n                since Python code won't be evaluated after the initial eager pass.\n                Only possible if framework=tf2.\n            eager_max_retraces: Maximum number of tf.function re-traces before a\n                runtime error is raised. This is to prevent unnoticed retraces of\n                methods inside the `..._eager_traced` Policy, which could slow down\n                execution by a factor of 4, without the user noticing what the root\n                cause for this slowdown could be.\n                Only necessary for framework=tf2.\n                Set to None to ignore the re-trace count and never throw an error.\n            tf_session_args: Configures TF for single-process operation by default.\n            local_tf_session_args: Override the following tf session args on the local\n                worker\n            torch_compile_learner: If True, forward_train methods on TorchRLModules\n                on the learner are compiled. If not specified, the default is to compile\n                forward train on the learner.\n            torch_compile_learner_what_to_compile: A TorchCompileWhatToCompile\n                mode specifying what to compile on the learner side if\n                torch_compile_learner is True. See TorchCompileWhatToCompile for\n                details and advice on its usage.\n            torch_compile_learner_dynamo_backend: The torch dynamo backend to use on\n                the learner.\n            torch_compile_learner_dynamo_mode: The torch dynamo mode to use on the\n                learner.\n            torch_compile_worker: If True, forward exploration and inference methods on\n                TorchRLModules on the workers are compiled. If not specified,\n                the default is to not compile forward methods on the workers because\n                retracing can be expensive.\n            torch_compile_worker_dynamo_backend: The torch dynamo backend to use on\n                the workers.\n            torch_compile_worker_dynamo_mode: The torch dynamo mode to use on the\n                workers.\n\n        Returns:\n            This updated AlgorithmConfig object.\n        "
        if framework is not NotProvided:
            if framework == 'tfe':
                deprecation_warning(old="AlgorithmConfig.framework('tfe')", new="AlgorithmConfig.framework('tf2')", error=True)
            self.framework_str = framework
        if eager_tracing is not NotProvided:
            self.eager_tracing = eager_tracing
        if eager_max_retraces is not NotProvided:
            self.eager_max_retraces = eager_max_retraces
        if tf_session_args is not NotProvided:
            self.tf_session_args = tf_session_args
        if local_tf_session_args is not NotProvided:
            self.local_tf_session_args = local_tf_session_args
        if torch_compile_learner is not NotProvided:
            self.torch_compile_learner = torch_compile_learner
        if torch_compile_learner_dynamo_backend is not NotProvided:
            self.torch_compile_learner_dynamo_backend = torch_compile_learner_dynamo_backend
        if torch_compile_learner_dynamo_mode is not NotProvided:
            self.torch_compile_learner_dynamo_mode = torch_compile_learner_dynamo_mode
        if torch_compile_learner_what_to_compile is not NotProvided:
            self.torch_compile_learner_what_to_compile = torch_compile_learner_what_to_compile
        if torch_compile_worker is not NotProvided:
            self.torch_compile_worker = torch_compile_worker
        if torch_compile_worker_dynamo_backend is not NotProvided:
            self.torch_compile_worker_dynamo_backend = torch_compile_worker_dynamo_backend
        if torch_compile_worker_dynamo_mode is not NotProvided:
            self.torch_compile_worker_dynamo_mode = torch_compile_worker_dynamo_mode
        return self

    def environment(self, env: Optional[Union[str, EnvType]]=NotProvided, *, env_config: Optional[EnvConfigDict]=NotProvided, observation_space: Optional[gym.spaces.Space]=NotProvided, action_space: Optional[gym.spaces.Space]=NotProvided, env_task_fn: Optional[Callable[[ResultDict, EnvType, EnvContext], Any]]=NotProvided, render_env: Optional[bool]=NotProvided, clip_rewards: Optional[Union[bool, float]]=NotProvided, normalize_actions: Optional[bool]=NotProvided, clip_actions: Optional[bool]=NotProvided, disable_env_checking: Optional[bool]=NotProvided, is_atari: Optional[bool]=NotProvided, auto_wrap_old_gym_envs: Optional[bool]=NotProvided, action_mask_key: Optional[str]=NotProvided) -> 'AlgorithmConfig':
        if False:
            print('Hello World!')
        'Sets the config\'s RL-environment settings.\n\n        Args:\n            env: The environment specifier. This can either be a tune-registered env,\n                via `tune.register_env([name], lambda env_ctx: [env object])`,\n                or a string specifier of an RLlib supported type. In the latter case,\n                RLlib will try to interpret the specifier as either an Farama-Foundation\n                gymnasium env, a PyBullet env, a ViZDoomGym env, or a fully qualified\n                classpath to an Env class, e.g.\n                "ray.rllib.examples.env.random_env.RandomEnv".\n            env_config: Arguments dict passed to the env creator as an EnvContext\n                object (which is a dict plus the properties: num_rollout_workers,\n                worker_index, vector_index, and remote).\n            observation_space: The observation space for the Policies of this Algorithm.\n            action_space: The action space for the Policies of this Algorithm.\n            env_task_fn: A callable taking the last train results, the base env and the\n                env context as args and returning a new task to set the env to.\n                The env must be a `TaskSettableEnv` sub-class for this to work.\n                See `examples/curriculum_learning.py` for an example.\n            render_env: If True, try to render the environment on the local worker or on\n                worker 1 (if num_rollout_workers > 0). For vectorized envs, this usually\n                means that only the first sub-environment will be rendered.\n                In order for this to work, your env will have to implement the\n                `render()` method which either:\n                a) handles window generation and rendering itself (returning True) or\n                b) returns a numpy uint8 image of shape [height x width x 3 (RGB)].\n            clip_rewards: Whether to clip rewards during Policy\'s postprocessing.\n                None (default): Clip for Atari only (r=sign(r)).\n                True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0.\n                False: Never clip.\n                [float value]: Clip at -value and + value.\n                Tuple[value1, value2]: Clip at value1 and value2.\n            normalize_actions: If True, RLlib will learn entirely inside a normalized\n                action space (0.0 centered with small stddev; only affecting Box\n                components). We will unsquash actions (and clip, just in case) to the\n                bounds of the env\'s action space before sending actions back to the env.\n            clip_actions: If True, RLlib will clip actions according to the env\'s bounds\n                before sending them back to the env.\n                TODO: (sven) This option should be deprecated and always be False.\n            disable_env_checking: If True, disable the environment pre-checking module.\n            is_atari: This config can be used to explicitly specify whether the env is\n                an Atari env or not. If not specified, RLlib will try to auto-detect\n                this.\n            auto_wrap_old_gym_envs: Whether to auto-wrap old gym environments (using\n                the pre 0.24 gym APIs, e.g. reset() returning single obs and no info\n                dict). If True, RLlib will automatically wrap the given gym env class\n                with the gym-provided compatibility wrapper\n                (gym.wrappers.EnvCompatibility). If False, RLlib will produce a\n                descriptive error on which steps to perform to upgrade to gymnasium\n                (or to switch this flag to True).\n             action_mask_key: If observation is a dictionary, expect the value by\n                the key `action_mask_key` to contain a valid actions mask (`numpy.int8`\n                array of zeros and ones). Defaults to "action_mask".\n\n        Returns:\n            This updated AlgorithmConfig object.\n        '
        if env is not NotProvided:
            self.env = env
        if env_config is not NotProvided:
            deep_update(self.env_config, env_config, True)
        if observation_space is not NotProvided:
            self.observation_space = observation_space
        if action_space is not NotProvided:
            self.action_space = action_space
        if env_task_fn is not NotProvided:
            self.env_task_fn = env_task_fn
        if render_env is not NotProvided:
            self.render_env = render_env
        if clip_rewards is not NotProvided:
            self.clip_rewards = clip_rewards
        if normalize_actions is not NotProvided:
            self.normalize_actions = normalize_actions
        if clip_actions is not NotProvided:
            self.clip_actions = clip_actions
        if disable_env_checking is not NotProvided:
            self.disable_env_checking = disable_env_checking
        if is_atari is not NotProvided:
            self._is_atari = is_atari
        if auto_wrap_old_gym_envs is not NotProvided:
            self.auto_wrap_old_gym_envs = auto_wrap_old_gym_envs
        if action_mask_key is not NotProvided:
            self.action_mask_key = action_mask_key
        return self

    def rollouts(self, *, env_runner_cls: Optional[type]=NotProvided, num_rollout_workers: Optional[int]=NotProvided, num_envs_per_worker: Optional[int]=NotProvided, create_env_on_local_worker: Optional[bool]=NotProvided, sample_collector: Optional[Type[SampleCollector]]=NotProvided, sample_async: Optional[bool]=NotProvided, enable_connectors: Optional[bool]=NotProvided, use_worker_filter_stats: Optional[bool]=NotProvided, update_worker_filter_stats: Optional[bool]=NotProvided, rollout_fragment_length: Optional[Union[int, str]]=NotProvided, batch_mode: Optional[str]=NotProvided, remote_worker_envs: Optional[bool]=NotProvided, remote_env_batch_wait_ms: Optional[float]=NotProvided, validate_workers_after_construction: Optional[bool]=NotProvided, preprocessor_pref: Optional[str]=NotProvided, observation_filter: Optional[str]=NotProvided, compress_observations: Optional[bool]=NotProvided, enable_tf1_exec_eagerly: Optional[bool]=NotProvided, sampler_perf_stats_ema_coef: Optional[float]=NotProvided, ignore_worker_failures=DEPRECATED_VALUE, recreate_failed_workers=DEPRECATED_VALUE, restart_failed_sub_environments=DEPRECATED_VALUE, num_consecutive_worker_failures_tolerance=DEPRECATED_VALUE, worker_health_probe_timeout_s=DEPRECATED_VALUE, worker_restore_timeout_s=DEPRECATED_VALUE, synchronize_filter=DEPRECATED_VALUE) -> 'AlgorithmConfig':
        if False:
            print('Hello World!')
        'Sets the rollout worker configuration.\n\n        Args:\n            env_runner_cls: The EnvRunner class to use for environment rollouts (data\n                collection).\n            num_rollout_workers: Number of rollout worker actors to create for\n                parallel sampling. Setting this to 0 will force rollouts to be done in\n                the local worker (driver process or the Algorithm\'s actor when using\n                Tune).\n            num_envs_per_worker: Number of environments to evaluate vector-wise per\n                worker. This enables model inference batching, which can improve\n                performance for inference bottlenecked workloads.\n            sample_collector: The SampleCollector class to be used to collect and\n                retrieve environment-, model-, and sampler data. Override the\n                SampleCollector base class to implement your own\n                collection/buffering/retrieval logic.\n            create_env_on_local_worker: When `num_rollout_workers` > 0, the driver\n                (local_worker; worker-idx=0) does not need an environment. This is\n                because it doesn\'t have to sample (done by remote_workers;\n                worker_indices > 0) nor evaluate (done by evaluation workers;\n                see below).\n            sample_async: Use a background thread for sampling (slightly off-policy,\n                usually not advisable to turn on unless your env specifically requires\n                it).\n            enable_connectors: Use connector based environment runner, so that all\n                preprocessing of obs and postprocessing of actions are done in agent\n                and action connectors.\n            use_worker_filter_stats: Whether to use the workers in the WorkerSet to\n                update the central filters (held by the local worker). If False, stats\n                from the workers will not be used and discarded.\n            update_worker_filter_stats: Whether to push filter updates from the central\n                filters (held by the local worker) to the remote workers\' filters.\n                Setting this to True might be useful within the evaluation config in\n                order to disable the usage of evaluation trajectories for synching\n                the central filter (used for training).\n            rollout_fragment_length: Divide episodes into fragments of this many steps\n                each during rollouts. Trajectories of this size are collected from\n                rollout workers and combined into a larger batch of `train_batch_size`\n                for learning.\n                For example, given rollout_fragment_length=100 and\n                train_batch_size=1000:\n                1. RLlib collects 10 fragments of 100 steps each from rollout workers.\n                2. These fragments are concatenated and we perform an epoch of SGD.\n                When using multiple envs per worker, the fragment size is multiplied by\n                `num_envs_per_worker`. This is since we are collecting steps from\n                multiple envs in parallel. For example, if num_envs_per_worker=5, then\n                rollout workers will return experiences in chunks of 5*100 = 500 steps.\n                The dataflow here can vary per algorithm. For example, PPO further\n                divides the train batch into minibatches for multi-epoch SGD.\n                Set to "auto" to have RLlib compute an exact `rollout_fragment_length`\n                to match the given batch size.\n            batch_mode: How to build individual batches with the EnvRunner(s). Batches\n                coming from distributed EnvRunners are usually concat\'d to form the\n                train batch. Note that "steps" below can mean different things (either\n                env- or agent-steps) and depends on the `count_steps_by` setting,\n                adjustable via `AlgorithmConfig.multi_agent(count_steps_by=..)`:\n                1) "truncate_episodes": Each call to `EnvRunner.sample()` will return a\n                batch of at most `rollout_fragment_length * num_envs_per_worker` in\n                size. The batch will be exactly `rollout_fragment_length * num_envs`\n                in size if postprocessing does not change batch sizes. Episodes\n                may be truncated in order to meet this size requirement.\n                This mode guarantees evenly sized batches, but increases\n                variance as the future return must now be estimated at truncation\n                boundaries.\n                2) "complete_episodes": Each call to `EnvRunner.sample()` will return a\n                batch of at least `rollout_fragment_length * num_envs_per_worker` in\n                size. Episodes will not be truncated, but multiple episodes\n                may be packed within one batch to meet the (minimum) batch size.\n                Note that when `num_envs_per_worker > 1`, episode steps will be buffered\n                until the episode completes, and hence batches may contain\n                significant amounts of off-policy data.\n            remote_worker_envs: If using num_envs_per_worker > 1, whether to create\n                those new envs in remote processes instead of in the same worker.\n                This adds overheads, but can make sense if your envs can take much\n                time to step / reset (e.g., for StarCraft). Use this cautiously;\n                overheads are significant.\n            remote_env_batch_wait_ms: Timeout that remote workers are waiting when\n                polling environments. 0 (continue when at least one env is ready) is\n                a reasonable default, but optimal value could be obtained by measuring\n                your environment step / reset and model inference perf.\n            validate_workers_after_construction: Whether to validate that each created\n                remote worker is healthy after its construction process.\n            preprocessor_pref: Whether to use "rllib" or "deepmind" preprocessors by\n                default. Set to None for using no preprocessor. In this case, the\n                model will have to handle possibly complex observations from the\n                environment.\n            observation_filter: Element-wise observation filter, either "NoFilter"\n                or "MeanStdFilter".\n            compress_observations: Whether to LZ4 compress individual observations\n                in the SampleBatches collected during rollouts.\n            enable_tf1_exec_eagerly: Explicitly tells the rollout worker to enable\n                TF eager execution. This is useful for example when framework is\n                "torch", but a TF2 policy needs to be restored for evaluation or\n                league-based purposes.\n            sampler_perf_stats_ema_coef: If specified, perf stats are in EMAs. This\n                is the coeff of how much new data points contribute to the averages.\n                Default is None, which uses simple global average instead.\n                The EMA update rule is: updated = (1 - ema_coef) * old + ema_coef * new\n\n        Returns:\n            This updated AlgorithmConfig object.\n        '
        if env_runner_cls is not NotProvided:
            self.env_runner_cls = env_runner_cls
        if num_rollout_workers is not NotProvided:
            self.num_rollout_workers = num_rollout_workers
        if num_envs_per_worker is not NotProvided:
            self.num_envs_per_worker = num_envs_per_worker
        if sample_collector is not NotProvided:
            self.sample_collector = sample_collector
        if create_env_on_local_worker is not NotProvided:
            self.create_env_on_local_worker = create_env_on_local_worker
        if sample_async is not NotProvided:
            self.sample_async = sample_async
        if enable_connectors is not NotProvided:
            self.enable_connectors = enable_connectors
        if use_worker_filter_stats is not NotProvided:
            self.use_worker_filter_stats = use_worker_filter_stats
        if update_worker_filter_stats is not NotProvided:
            self.update_worker_filter_stats = update_worker_filter_stats
        if rollout_fragment_length is not NotProvided:
            self.rollout_fragment_length = rollout_fragment_length
        if batch_mode is not NotProvided:
            self.batch_mode = batch_mode
        if remote_worker_envs is not NotProvided:
            self.remote_worker_envs = remote_worker_envs
        if remote_env_batch_wait_ms is not NotProvided:
            self.remote_env_batch_wait_ms = remote_env_batch_wait_ms
        if validate_workers_after_construction is not NotProvided:
            self.validate_workers_after_construction = validate_workers_after_construction
        if preprocessor_pref is not NotProvided:
            self.preprocessor_pref = preprocessor_pref
        if observation_filter is not NotProvided:
            self.observation_filter = observation_filter
        if synchronize_filter is not NotProvided:
            self.synchronize_filters = synchronize_filter
        if compress_observations is not NotProvided:
            self.compress_observations = compress_observations
        if enable_tf1_exec_eagerly is not NotProvided:
            self.enable_tf1_exec_eagerly = enable_tf1_exec_eagerly
        if sampler_perf_stats_ema_coef is not NotProvided:
            self.sampler_perf_stats_ema_coef = sampler_perf_stats_ema_coef
        if synchronize_filter != DEPRECATED_VALUE:
            deprecation_warning(old='AlgorithmConfig.rollouts(synchronize_filter=..)', new='AlgorithmConfig.rollouts(update_worker_filter_stats=..)', error=False)
            self.update_worker_filter_stats = synchronize_filter
        if ignore_worker_failures != DEPRECATED_VALUE:
            deprecation_warning(old='ignore_worker_failures is deprecated, and will soon be a no-op', error=False)
            self.ignore_worker_failures = ignore_worker_failures
        if recreate_failed_workers != DEPRECATED_VALUE:
            deprecation_warning(old='AlgorithmConfig.rollouts(recreate_failed_workers=..)', new='AlgorithmConfig.fault_tolerance(recreate_failed_workers=..)', error=False)
            self.recreate_failed_workers = recreate_failed_workers
        if restart_failed_sub_environments != DEPRECATED_VALUE:
            deprecation_warning(old='AlgorithmConfig.rollouts(restart_failed_sub_environments=..)', new='AlgorithmConfig.fault_tolerance(restart_failed_sub_environments=..)', error=False)
            self.restart_failed_sub_environments = restart_failed_sub_environments
        if num_consecutive_worker_failures_tolerance != DEPRECATED_VALUE:
            deprecation_warning(old='AlgorithmConfig.rollouts(num_consecutive_worker_failures_tolerance=..)', new='AlgorithmConfig.fault_tolerance(num_consecutive_worker_failures_tolerance=..)', error=False)
            self.num_consecutive_worker_failures_tolerance = num_consecutive_worker_failures_tolerance
        if worker_health_probe_timeout_s != DEPRECATED_VALUE:
            deprecation_warning(old='AlgorithmConfig.rollouts(worker_health_probe_timeout_s=..)', new='AlgorithmConfig.fault_tolerance(worker_health_probe_timeout_s=..)', error=False)
            self.worker_health_probe_timeout_s = worker_health_probe_timeout_s
        if worker_restore_timeout_s != DEPRECATED_VALUE:
            deprecation_warning(old='AlgorithmConfig.rollouts(worker_restore_timeout_s=..)', new='AlgorithmConfig.fault_tolerance(worker_restore_timeout_s=..)', error=False)
            self.worker_restore_timeout_s = worker_restore_timeout_s
        return self

    def training(self, *, gamma: Optional[float]=NotProvided, lr: Optional[LearningRateOrSchedule]=NotProvided, grad_clip: Optional[float]=NotProvided, grad_clip_by: Optional[str]=NotProvided, train_batch_size: Optional[int]=NotProvided, model: Optional[dict]=NotProvided, optimizer: Optional[dict]=NotProvided, max_requests_in_flight_per_sampler_worker: Optional[int]=NotProvided, learner_class: Optional[Type['Learner']]=NotProvided, _enable_learner_api: Optional[bool]=NotProvided) -> 'AlgorithmConfig':
        if False:
            i = 10
            return i + 15
        "Sets the training related configuration.\n\n        Args:\n            gamma: Float specifying the discount factor of the Markov Decision process.\n            lr: The learning rate (float) or learning rate schedule in the format of\n                [[timestep, lr-value], [timestep, lr-value], ...]\n                In case of a schedule, intermediary timesteps will be assigned to\n                linearly interpolated learning rate values. A schedule config's first\n                entry must start with timestep 0, i.e.: [[0, initial_value], [...]].\n                Note: If you require a) more than one optimizer (per RLModule),\n                b) optimizer types that are not Adam, c) a learning rate schedule that\n                is not a linearly interpolated, piecewise schedule as described above,\n                or d) specifying c'tor arguments of the optimizer that are not the\n                learning rate (e.g. Adam's epsilon), then you must override your\n                Learner's `configure_optimizer_for_module()` method and handle\n                lr-scheduling yourself.\n            grad_clip: If None, no gradient clipping will be applied. Otherwise,\n                depending on the setting of `grad_clip_by`, the (float) value of\n                `grad_clip` will have the following effect:\n                If `grad_clip_by=value`: Will clip all computed gradients individually\n                inside the interval [-`grad_clip`, +`grad_clip`].\n                If `grad_clip_by=norm`, will compute the L2-norm of each weight/bias\n                gradient tensor individually and then clip all gradients such that these\n                L2-norms do not exceed `grad_clip`. The L2-norm of a tensor is computed\n                via: `sqrt(SUM(w0^2, w1^2, ..., wn^2))` where w[i] are the elements of\n                the tensor (no matter what the shape of this tensor is).\n                If `grad_clip_by=global_norm`, will compute the square of the L2-norm of\n                each weight/bias gradient tensor individually, sum up all these squared\n                L2-norms across all given gradient tensors (e.g. the entire module to\n                be updated), square root that overall sum, and then clip all gradients\n                such that this global L2-norm does not exceed the given value.\n                The global L2-norm over a list of tensors (e.g. W and V) is computed\n                via:\n                `sqrt[SUM(w0^2, w1^2, ..., wn^2) + SUM(v0^2, v1^2, ..., vm^2)]`, where\n                w[i] and v[j] are the elements of the tensors W and V (no matter what\n                the shapes of these tensors are).\n            grad_clip_by: See `grad_clip` for the effect of this setting on gradient\n                clipping. Allowed values are `value`, `norm`, and `global_norm`.\n            train_batch_size: Training batch size, if applicable.\n            model: Arguments passed into the policy model. See models/catalog.py for a\n                full list of the available model options.\n                TODO: Provide ModelConfig objects instead of dicts.\n            optimizer: Arguments to pass to the policy optimizer. This setting is not\n                used when `_enable_new_api_stack=True`.\n            max_requests_in_flight_per_sampler_worker: Max number of inflight requests\n                to each sampling worker. See the FaultTolerantActorManager class for\n                more details.\n                Tuning these values is important when running experimens with\n                large sample batches, where there is the risk that the object store may\n                fill up, causing spilling of objects to disk. This can cause any\n                asynchronous requests to become very slow, making your experiment run\n                slow as well. You can inspect the object store during your experiment\n                via a call to ray memory on your headnode, and by using the ray\n                dashboard. If you're seeing that the object store is filling up,\n                turn down the number of remote requests in flight, or enable compression\n                in your experiment of timesteps.\n\n        Returns:\n            This updated AlgorithmConfig object.\n        "
        if gamma is not NotProvided:
            self.gamma = gamma
        if lr is not NotProvided:
            self.lr = lr
        if grad_clip is not NotProvided:
            self.grad_clip = grad_clip
        if grad_clip_by is not NotProvided:
            self.grad_clip_by = grad_clip_by
        if train_batch_size is not NotProvided:
            self.train_batch_size = train_batch_size
        if model is not NotProvided:
            self.model.update(model)
            if model.get('_use_default_native_models', DEPRECATED_VALUE) != DEPRECATED_VALUE:
                deprecation_warning(old='AlgorithmConfig.training(_use_default_native_models=True)', help='_use_default_native_models is not supported anymore. To get rid of this error, set `config.experimental(_enable_new_api_stack=True)`. Native models will be better supported by the upcoming RLModule API.', error=model['_use_default_native_models'])
        if optimizer is not NotProvided:
            self.optimizer = merge_dicts(self.optimizer, optimizer)
        if max_requests_in_flight_per_sampler_worker is not NotProvided:
            self.max_requests_in_flight_per_sampler_worker = max_requests_in_flight_per_sampler_worker
        if _enable_learner_api is not NotProvided:
            deprecation_warning(old='AlgorithmConfig.training(_enable_learner_api=True|False)', new='AlgorithmConfig.experimental(_enable_new_api_stack=True|False)', error=True)
        if learner_class is not NotProvided:
            self._learner_class = learner_class
        return self

    def callbacks(self, callbacks_class) -> 'AlgorithmConfig':
        if False:
            return 10
        'Sets the callbacks configuration.\n\n        Args:\n            callbacks_class: Callbacks class, whose methods will be run during\n                various phases of training and environment sample collection.\n                See the `DefaultCallbacks` class and\n                `examples/custom_metrics_and_callbacks.py` for more usage information.\n\n        Returns:\n            This updated AlgorithmConfig object.\n        '
        if callbacks_class is None:
            callbacks_class = DefaultCallbacks
        if not callable(callbacks_class):
            raise ValueError(f'`config.callbacks_class` must be a callable method that returns a subclass of DefaultCallbacks, got {callbacks_class}!')
        self.callbacks_class = callbacks_class
        return self

    def exploration(self, *, explore: Optional[bool]=NotProvided, exploration_config: Optional[dict]=NotProvided) -> 'AlgorithmConfig':
        if False:
            return 10
        "Sets the config's exploration settings.\n\n        Args:\n            explore: Default exploration behavior, iff `explore=None` is passed into\n                compute_action(s). Set to False for no exploration behavior (e.g.,\n                for evaluation).\n            exploration_config: A dict specifying the Exploration object's config.\n\n        Returns:\n            This updated AlgorithmConfig object.\n        "
        if explore is not NotProvided:
            self.explore = explore
        if exploration_config is not NotProvided:
            new_exploration_config = deep_update({'exploration_config': self.exploration_config}, {'exploration_config': exploration_config}, False, ['exploration_config'], ['exploration_config'])
            self.exploration_config = new_exploration_config['exploration_config']
        return self

    def evaluation(self, *, evaluation_interval: Optional[int]=NotProvided, evaluation_duration: Optional[Union[int, str]]=NotProvided, evaluation_duration_unit: Optional[str]=NotProvided, evaluation_sample_timeout_s: Optional[float]=NotProvided, evaluation_parallel_to_training: Optional[bool]=NotProvided, evaluation_config: Optional[Union['AlgorithmConfig', PartialAlgorithmConfigDict]]=NotProvided, off_policy_estimation_methods: Optional[Dict]=NotProvided, ope_split_batch_by_episode: Optional[bool]=NotProvided, evaluation_num_workers: Optional[int]=NotProvided, custom_evaluation_function: Optional[Callable]=NotProvided, always_attach_evaluation_results: Optional[bool]=NotProvided, enable_async_evaluation: Optional[bool]=NotProvided, evaluation_num_episodes=DEPRECATED_VALUE) -> 'AlgorithmConfig':
        if False:
            print('Hello World!')
        'Sets the config\'s evaluation settings.\n\n        Args:\n            evaluation_interval: Evaluate with every `evaluation_interval` training\n                iterations. The evaluation stats will be reported under the "evaluation"\n                metric key. Note that for Ape-X metrics are already only reported for\n                the lowest epsilon workers (least random workers).\n                Set to None (or 0) for no evaluation.\n            evaluation_duration: Duration for which to run evaluation each\n                `evaluation_interval`. The unit for the duration can be set via\n                `evaluation_duration_unit` to either "episodes" (default) or\n                "timesteps". If using multiple evaluation workers\n                (evaluation_num_workers > 1), the load to run will be split amongst\n                these.\n                If the value is "auto":\n                - For `evaluation_parallel_to_training=True`: Will run as many\n                episodes/timesteps that fit into the (parallel) training step.\n                - For `evaluation_parallel_to_training=False`: Error.\n            evaluation_duration_unit: The unit, with which to count the evaluation\n                duration. Either "episodes" (default) or "timesteps".\n            evaluation_sample_timeout_s: The timeout (in seconds) for the ray.get call\n                to the remote evaluation worker(s) `sample()` method. After this time,\n                the user will receive a warning and instructions on how to fix the\n                issue. This could be either to make sure the episode ends, increasing\n                the timeout, or switching to `evaluation_duration_unit=timesteps`.\n            evaluation_parallel_to_training: Whether to run evaluation in parallel to\n                a Algorithm.train() call using threading. Default=False.\n                E.g. evaluation_interval=2 -> For every other training iteration,\n                the Algorithm.train() and Algorithm.evaluate() calls run in parallel.\n                Note: This is experimental. Possible pitfalls could be race conditions\n                for weight synching at the beginning of the evaluation loop.\n            evaluation_config: Typical usage is to pass extra args to evaluation env\n                creator and to disable exploration by computing deterministic actions.\n                IMPORTANT NOTE: Policy gradient algorithms are able to find the optimal\n                policy, even if this is a stochastic one. Setting "explore=False" here\n                will result in the evaluation workers not using this optimal policy!\n            off_policy_estimation_methods: Specify how to evaluate the current policy,\n                along with any optional config parameters. This only has an effect when\n                reading offline experiences ("input" is not "sampler").\n                Available keys:\n                {ope_method_name: {"type": ope_type, ...}} where `ope_method_name`\n                is a user-defined string to save the OPE results under, and\n                `ope_type` can be any subclass of OffPolicyEstimator, e.g.\n                ray.rllib.offline.estimators.is::ImportanceSampling\n                or your own custom subclass, or the full class path to the subclass.\n                You can also add additional config arguments to be passed to the\n                OffPolicyEstimator in the dict, e.g.\n                {"qreg_dr": {"type": DoublyRobust, "q_model_type": "qreg", "k": 5}}\n            ope_split_batch_by_episode: Whether to use SampleBatch.split_by_episode() to\n                split the input batch to episodes before estimating the ope metrics. In\n                case of bandits you should make this False to see improvements in ope\n                evaluation speed. In case of bandits, it is ok to not split by episode,\n                since each record is one timestep already. The default is True.\n            evaluation_num_workers: Number of parallel workers to use for evaluation.\n                Note that this is set to zero by default, which means evaluation will\n                be run in the algorithm process (only if evaluation_interval is not\n                None). If you increase this, it will increase the Ray resource usage of\n                the algorithm since evaluation workers are created separately from\n                rollout workers (used to sample data for training).\n            custom_evaluation_function: Customize the evaluation method. This must be a\n                function of signature (algo: Algorithm, eval_workers: WorkerSet) ->\n                metrics: dict. See the Algorithm.evaluate() method to see the default\n                implementation. The Algorithm guarantees all eval workers have the\n                latest policy state before this function is called.\n            always_attach_evaluation_results: Make sure the latest available evaluation\n                results are always attached to a step result dict. This may be useful\n                if Tune or some other meta controller needs access to evaluation metrics\n                all the time.\n            enable_async_evaluation: If True, use an AsyncRequestsManager for\n                the evaluation workers and use this manager to send `sample()` requests\n                to the evaluation workers. This way, the Algorithm becomes more robust\n                against long running episodes and/or failing (and restarting) workers.\n\n        Returns:\n            This updated AlgorithmConfig object.\n        '
        if evaluation_num_episodes != DEPRECATED_VALUE:
            deprecation_warning(old='AlgorithmConfig.evaluation(evaluation_num_episodes=..)', new="AlgorithmConfig.evaluation(evaluation_duration=.., evaluation_duration_unit='episodes')", error=False)
            evaluation_duration = evaluation_num_episodes
        if evaluation_interval is not NotProvided:
            self.evaluation_interval = evaluation_interval
        if evaluation_duration is not NotProvided:
            self.evaluation_duration = evaluation_duration
        if evaluation_duration_unit is not NotProvided:
            self.evaluation_duration_unit = evaluation_duration_unit
        if evaluation_sample_timeout_s is not NotProvided:
            self.evaluation_sample_timeout_s = evaluation_sample_timeout_s
        if evaluation_parallel_to_training is not NotProvided:
            self.evaluation_parallel_to_training = evaluation_parallel_to_training
        if evaluation_config is not NotProvided:
            if evaluation_config is None:
                self.evaluation_config = None
            else:
                from ray.rllib.algorithms.algorithm import Algorithm
                self.evaluation_config = deep_update(self.evaluation_config or {}, evaluation_config, True, Algorithm._allow_unknown_subkeys, Algorithm._override_all_subkeys_if_type_changes, Algorithm._override_all_key_list)
        if off_policy_estimation_methods is not NotProvided:
            self.off_policy_estimation_methods = off_policy_estimation_methods
        if evaluation_num_workers is not NotProvided:
            self.evaluation_num_workers = evaluation_num_workers
        if custom_evaluation_function is not NotProvided:
            self.custom_evaluation_function = custom_evaluation_function
        if always_attach_evaluation_results is not NotProvided:
            self.always_attach_evaluation_results = always_attach_evaluation_results
        if enable_async_evaluation is not NotProvided:
            self.enable_async_evaluation = enable_async_evaluation
        if ope_split_batch_by_episode is not NotProvided:
            self.ope_split_batch_by_episode = ope_split_batch_by_episode
        return self

    def offline_data(self, *, input_=NotProvided, input_config=NotProvided, actions_in_input_normalized=NotProvided, input_evaluation=NotProvided, postprocess_inputs=NotProvided, shuffle_buffer_size=NotProvided, output=NotProvided, output_config=NotProvided, output_compress_columns=NotProvided, output_max_file_size=NotProvided, offline_sampling=NotProvided) -> 'AlgorithmConfig':
        if False:
            while True:
                i = 10
        'Sets the config\'s offline data settings.\n\n        Args:\n            input_: Specify how to generate experiences:\n                - "sampler": Generate experiences via online (env) simulation (default).\n                - A local directory or file glob expression (e.g., "/tmp/*.json").\n                - A list of individual file paths/URIs (e.g., ["/tmp/1.json",\n                "s3://bucket/2.json"]).\n                - A dict with string keys and sampling probabilities as values (e.g.,\n                {"sampler": 0.4, "/tmp/*.json": 0.4, "s3://bucket/expert.json": 0.2}).\n                - A callable that takes an `IOContext` object as only arg and returns a\n                ray.rllib.offline.InputReader.\n                - A string key that indexes a callable with tune.registry.register_input\n            input_config: Arguments that describe the settings for reading the input.\n                If input is `sample`, this will be environment configuation, e.g.\n                `env_name` and `env_config`, etc. See `EnvContext` for more info.\n                If the input is `dataset`, this will be e.g. `format`, `path`.\n            actions_in_input_normalized: True, if the actions in a given offline "input"\n                are already normalized (between -1.0 and 1.0). This is usually the case\n                when the offline file has been generated by another RLlib algorithm\n                (e.g. PPO or SAC), while "normalize_actions" was set to True.\n            postprocess_inputs: Whether to run postprocess_trajectory() on the\n                trajectory fragments from offline inputs. Note that postprocessing will\n                be done using the *current* policy, not the *behavior* policy, which\n                is typically undesirable for on-policy algorithms.\n            shuffle_buffer_size: If positive, input batches will be shuffled via a\n                sliding window buffer of this number of batches. Use this if the input\n                data is not in random enough order. Input is delayed until the shuffle\n                buffer is filled.\n            output: Specify where experiences should be saved:\n                 - None: don\'t save any experiences\n                 - "logdir" to save to the agent log dir\n                 - a path/URI to save to a custom output directory (e.g., "s3://bckt/")\n                 - a function that returns a rllib.offline.OutputWriter\n            output_config: Arguments accessible from the IOContext for configuring\n                custom output.\n            output_compress_columns: What sample batch columns to LZ4 compress in the\n                output data.\n            output_max_file_size: Max output file size (in bytes) before rolling over\n                to a new file.\n            offline_sampling: Whether sampling for the Algorithm happens via\n                reading from offline data. If True, EnvRunners will NOT limit the\n                number of collected batches within the same `sample()` call based on\n                the number of sub-environments within the worker (no sub-environments\n                present).\n\n        Returns:\n            This updated AlgorithmConfig object.\n        '
        if input_ is not NotProvided:
            self.input_ = input_
        if input_config is not NotProvided:
            if not isinstance(input_config, dict):
                raise ValueError(f'input_config must be a dict, got {type(input_config)}.')
            msg = '{} should not be set in the input_config. RLlib will use {} instead.'
            if input_config.get('num_cpus_per_read_task') is not None:
                raise ValueError(msg.format('num_cpus_per_read_task', 'config.resources(num_cpus_per_worker=..)'))
            if input_config.get('parallelism') is not None:
                if self.in_evaluation:
                    raise ValueError(msg.format('parallelism', 'config.evaluation(evaluation_num_workers=..)'))
                else:
                    raise ValueError(msg.format('parallelism', 'config.rollouts(num_rollout_workers=..)'))
            self.input_config = input_config
        if actions_in_input_normalized is not NotProvided:
            self.actions_in_input_normalized = actions_in_input_normalized
        if input_evaluation is not NotProvided:
            deprecation_warning(old='offline_data(input_evaluation={})'.format(input_evaluation), new='evaluation(off_policy_estimation_methods={})'.format(input_evaluation), error=True, help='Running OPE during training is not recommended.')
        if postprocess_inputs is not NotProvided:
            self.postprocess_inputs = postprocess_inputs
        if shuffle_buffer_size is not NotProvided:
            self.shuffle_buffer_size = shuffle_buffer_size
        if output is not NotProvided:
            self.output = output
        if output_config is not NotProvided:
            self.output_config = output_config
        if output_compress_columns is not NotProvided:
            self.output_compress_columns = output_compress_columns
        if output_max_file_size is not NotProvided:
            self.output_max_file_size = output_max_file_size
        if offline_sampling is not NotProvided:
            self.offline_sampling = offline_sampling
        return self

    def multi_agent(self, *, policies=NotProvided, algorithm_config_overrides_per_module: Optional[Dict[ModuleID, PartialAlgorithmConfigDict]]=NotProvided, policy_map_capacity: Optional[int]=NotProvided, policy_mapping_fn: Optional[Callable[[AgentID, 'OldEpisode'], PolicyID]]=NotProvided, policies_to_train: Optional[Union[Container[PolicyID], Callable[[PolicyID, SampleBatchType], bool]]]=NotProvided, policy_states_are_swappable: Optional[bool]=NotProvided, observation_fn: Optional[Callable]=NotProvided, count_steps_by: Optional[str]=NotProvided, replay_mode=DEPRECATED_VALUE, policy_map_cache=DEPRECATED_VALUE) -> 'AlgorithmConfig':
        if False:
            while True:
                i = 10
        'Sets the config\'s multi-agent settings.\n\n        Validates the new multi-agent settings and translates everything into\n        a unified multi-agent setup format. For example a `policies` list or set\n        of IDs is properly converted into a dict mapping these IDs to PolicySpecs.\n\n        Args:\n            policies: Map of type MultiAgentPolicyConfigDict from policy ids to either\n                4-tuples of (policy_cls, obs_space, act_space, config) or PolicySpecs.\n                These tuples or PolicySpecs define the class of the policy, the\n                observation- and action spaces of the policies, and any extra config.\n            algorithm_config_overrides_per_module: Only used if\n                `_enable_new_api_stack=True`.\n                A mapping from ModuleIDs to per-module AlgorithmConfig override dicts,\n                which apply certain settings,\n                e.g. the learning rate, from the main AlgorithmConfig only to this\n                particular module (within a MultiAgentRLModule).\n                You can create override dicts by using the `AlgorithmConfig.overrides`\n                utility. For example, to override your learning rate and (PPO) lambda\n                setting just for a single RLModule with your MultiAgentRLModule, do:\n                config.multi_agent(algorithm_config_overrides_per_module={\n                "module_1": PPOConfig.overrides(lr=0.0002, lambda_=0.75),\n                })\n            policy_map_capacity: Keep this many policies in the "policy_map" (before\n                writing least-recently used ones to disk/S3).\n            policy_mapping_fn: Function mapping agent ids to policy ids. The signature\n                is: `(agent_id, episode, worker, **kwargs) -> PolicyID`.\n            policies_to_train: Determines those policies that should be updated.\n                Options are:\n                - None, for training all policies.\n                - An iterable of PolicyIDs that should be trained.\n                - A callable, taking a PolicyID and a SampleBatch or MultiAgentBatch\n                and returning a bool (indicating whether the given policy is trainable\n                or not, given the particular batch). This allows you to have a policy\n                trained only on certain data (e.g. when playing against a certain\n                opponent).\n            policy_states_are_swappable: Whether all Policy objects in this map can be\n                "swapped out" via a simple `state = A.get_state(); B.set_state(state)`,\n                where `A` and `B` are policy instances in this map. You should set\n                this to True for significantly speeding up the PolicyMap\'s cache lookup\n                times, iff your policies all share the same neural network\n                architecture and optimizer types. If True, the PolicyMap will not\n                have to garbage collect old, least recently used policies, but instead\n                keep them in memory and simply override their state with the state of\n                the most recently accessed one.\n                For example, in a league-based training setup, you might have 100s of\n                the same policies in your map (playing against each other in various\n                combinations), but all of them share the same state structure\n                (are "swappable").\n            observation_fn: Optional function that can be used to enhance the local\n                agent observations to include more state. See\n                rllib/evaluation/observation_function.py for more info.\n            count_steps_by: Which metric to use as the "batch size" when building a\n                MultiAgentBatch. The two supported values are:\n                "env_steps": Count each time the env is "stepped" (no matter how many\n                multi-agent actions are passed/how many multi-agent observations\n                have been returned in the previous step).\n                "agent_steps": Count each individual agent step as one step.\n\n        Returns:\n            This updated AlgorithmConfig object.\n        '
        if policies is not NotProvided:
            for pid in policies:
                validate_policy_id(pid, error=True)
            if isinstance(policies, dict):
                for (pid, spec) in policies.items():
                    if not isinstance(spec, PolicySpec):
                        if not isinstance(spec, (list, tuple)) or len(spec) != 4:
                            raise ValueError(f'Policy specs must be tuples/lists of (cls or None, obs_space, action_space, config), got {spec} for PolicyID={pid}')
                    elif not isinstance(spec.config, (AlgorithmConfig, dict)) and spec.config is not None:
                        raise ValueError(f'Multi-agent policy config for {pid} must be a dict or AlgorithmConfig object, but got {type(spec.config)}!')
            self.policies = policies
        if algorithm_config_overrides_per_module is not NotProvided:
            self.algorithm_config_overrides_per_module = algorithm_config_overrides_per_module
        if policy_map_capacity is not NotProvided:
            self.policy_map_capacity = policy_map_capacity
        if policy_mapping_fn is not NotProvided:
            if isinstance(policy_mapping_fn, dict):
                policy_mapping_fn = from_config(policy_mapping_fn)
            self.policy_mapping_fn = policy_mapping_fn
        if observation_fn is not NotProvided:
            self.observation_fn = observation_fn
        if policy_map_cache != DEPRECATED_VALUE:
            deprecation_warning(old='AlgorithmConfig.multi_agent(policy_map_cache=..)', error=True)
        if replay_mode != DEPRECATED_VALUE:
            deprecation_warning(old='AlgorithmConfig.multi_agent(replay_mode=..)', new="AlgorithmConfig.training(replay_buffer_config={'replay_mode': ..})", error=True)
        if count_steps_by is not NotProvided:
            if count_steps_by not in ['env_steps', 'agent_steps']:
                raise ValueError(f'config.multi_agent(count_steps_by=..) must be one of [env_steps|agent_steps], not {count_steps_by}!')
            self.count_steps_by = count_steps_by
        if policies_to_train is not NotProvided:
            assert isinstance(policies_to_train, (list, set, tuple)) or callable(policies_to_train) or policies_to_train is None, 'ERROR: `policies_to_train` must be a [list|set|tuple] or a callable taking PolicyID and SampleBatch and returning True|False (trainable or not?) or None (for always training all policies).'
            if isinstance(policies_to_train, (list, set, tuple)):
                if len(policies_to_train) == 0:
                    logger.warning('`config.multi_agent(policies_to_train=..)` is empty! Make sure - if you would like to learn at least one policy - to add its ID to that list.')
            self.policies_to_train = policies_to_train
        if policy_states_are_swappable is not NotProvided:
            self.policy_states_are_swappable = policy_states_are_swappable
        return self

    def is_multi_agent(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Returns whether this config specifies a multi-agent setup.\n\n        Returns:\n            True, if a) >1 policies defined OR b) 1 policy defined, but its ID is NOT\n            DEFAULT_POLICY_ID.\n        '
        return len(self.policies) > 1 or DEFAULT_POLICY_ID not in self.policies

    def reporting(self, *, keep_per_episode_custom_metrics: Optional[bool]=NotProvided, metrics_episode_collection_timeout_s: Optional[float]=NotProvided, metrics_num_episodes_for_smoothing: Optional[int]=NotProvided, min_time_s_per_iteration: Optional[int]=NotProvided, min_train_timesteps_per_iteration: Optional[int]=NotProvided, min_sample_timesteps_per_iteration: Optional[int]=NotProvided) -> 'AlgorithmConfig':
        if False:
            while True:
                i = 10
        'Sets the config\'s reporting settings.\n\n        Args:\n            keep_per_episode_custom_metrics: Store raw custom metrics without\n                calculating max, min, mean\n            metrics_episode_collection_timeout_s: Wait for metric batches for at most\n                this many seconds. Those that have not returned in time will be\n                collected in the next train iteration.\n            metrics_num_episodes_for_smoothing: Smooth rollout metrics over this many\n                episodes, if possible.\n                In case rollouts (sample collection) just started, there may be fewer\n                than this many episodes in the buffer and we\'ll compute metrics\n                over this smaller number of available episodes.\n                In case there are more than this many episodes collected in a single\n                training iteration, use all of these episodes for metrics computation,\n                meaning don\'t ever cut any "excess" episodes.\n                Set this to 1 to disable smoothing and to always report only the most\n                recently collected episode\'s return.\n            min_time_s_per_iteration: Minimum time to accumulate within a single\n                `train()` call. This value does not affect learning,\n                only the number of times `Algorithm.training_step()` is called by\n                `Algorithm.train()`. If - after one such step attempt, the time taken\n                has not reached `min_time_s_per_iteration`, will perform n more\n                `training_step()` calls until the minimum time has been\n                consumed. Set to 0 or None for no minimum time.\n            min_train_timesteps_per_iteration: Minimum training timesteps to accumulate\n                within a single `train()` call. This value does not affect learning,\n                only the number of times `Algorithm.training_step()` is called by\n                `Algorithm.train()`. If - after one such step attempt, the training\n                timestep count has not been reached, will perform n more\n                `training_step()` calls until the minimum timesteps have been\n                executed. Set to 0 or None for no minimum timesteps.\n            min_sample_timesteps_per_iteration: Minimum env sampling timesteps to\n                accumulate within a single `train()` call. This value does not affect\n                learning, only the number of times `Algorithm.training_step()` is\n                called by `Algorithm.train()`. If - after one such step attempt, the env\n                sampling timestep count has not been reached, will perform n more\n                `training_step()` calls until the minimum timesteps have been\n                executed. Set to 0 or None for no minimum timesteps.\n\n        Returns:\n            This updated AlgorithmConfig object.\n        '
        if keep_per_episode_custom_metrics is not NotProvided:
            self.keep_per_episode_custom_metrics = keep_per_episode_custom_metrics
        if metrics_episode_collection_timeout_s is not NotProvided:
            self.metrics_episode_collection_timeout_s = metrics_episode_collection_timeout_s
        if metrics_num_episodes_for_smoothing is not NotProvided:
            self.metrics_num_episodes_for_smoothing = metrics_num_episodes_for_smoothing
        if min_time_s_per_iteration is not NotProvided:
            self.min_time_s_per_iteration = min_time_s_per_iteration
        if min_train_timesteps_per_iteration is not NotProvided:
            self.min_train_timesteps_per_iteration = min_train_timesteps_per_iteration
        if min_sample_timesteps_per_iteration is not NotProvided:
            self.min_sample_timesteps_per_iteration = min_sample_timesteps_per_iteration
        return self

    def checkpointing(self, export_native_model_files: Optional[bool]=NotProvided, checkpoint_trainable_policies_only: Optional[bool]=NotProvided) -> 'AlgorithmConfig':
        if False:
            for i in range(10):
                print('nop')
        'Sets the config\'s checkpointing settings.\n\n        Args:\n            export_native_model_files: Whether an individual Policy-\n                or the Algorithm\'s checkpoints also contain (tf or torch) native\n                model files. These could be used to restore just the NN models\n                from these files w/o requiring RLlib. These files are generated\n                by calling the tf- or torch- built-in saving utility methods on\n                the actual models.\n            checkpoint_trainable_policies_only: Whether to only add Policies to the\n                Algorithm checkpoint (in sub-directory "policies/") that are trainable\n                according to the `is_trainable_policy` callable of the local worker.\n\n        Returns:\n            This updated AlgorithmConfig object.\n        '
        if export_native_model_files is not NotProvided:
            self.export_native_model_files = export_native_model_files
        if checkpoint_trainable_policies_only is not NotProvided:
            self.checkpoint_trainable_policies_only = checkpoint_trainable_policies_only
        return self

    def debugging(self, *, logger_creator: Optional[Callable[[], Logger]]=NotProvided, logger_config: Optional[dict]=NotProvided, log_level: Optional[str]=NotProvided, log_sys_usage: Optional[bool]=NotProvided, fake_sampler: Optional[bool]=NotProvided, seed: Optional[int]=NotProvided) -> 'AlgorithmConfig':
        if False:
            for i in range(10):
                print('nop')
        "Sets the config's debugging settings.\n\n        Args:\n            logger_creator: Callable that creates a ray.tune.Logger\n                object. If unspecified, a default logger is created.\n            logger_config: Define logger-specific configuration to be used inside Logger\n                Default value None allows overwriting with nested dicts.\n            log_level: Set the ray.rllib.* log level for the agent process and its\n                workers. Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level\n                will also periodically print out summaries of relevant internal dataflow\n                (this is also printed out once at startup at the INFO level). When using\n                the `rllib train` command, you can also use the `-v` and `-vv` flags as\n                shorthand for INFO and DEBUG.\n            log_sys_usage: Log system resource metrics to results. This requires\n                `psutil` to be installed for sys stats, and `gputil` for GPU metrics.\n            fake_sampler: Use fake (infinite speed) sampler. For testing only.\n            seed: This argument, in conjunction with worker_index, sets the random\n                seed of each worker, so that identically configured trials will have\n                identical results. This makes experiments reproducible.\n\n        Returns:\n            This updated AlgorithmConfig object.\n        "
        if logger_creator is not NotProvided:
            self.logger_creator = logger_creator
        if logger_config is not NotProvided:
            self.logger_config = logger_config
        if log_level is not NotProvided:
            self.log_level = log_level
        if log_sys_usage is not NotProvided:
            self.log_sys_usage = log_sys_usage
        if fake_sampler is not NotProvided:
            self.fake_sampler = fake_sampler
        if seed is not NotProvided:
            self.seed = seed
        return self

    def fault_tolerance(self, recreate_failed_workers: Optional[bool]=NotProvided, max_num_worker_restarts: Optional[int]=NotProvided, delay_between_worker_restarts_s: Optional[float]=NotProvided, restart_failed_sub_environments: Optional[bool]=NotProvided, num_consecutive_worker_failures_tolerance: Optional[int]=NotProvided, worker_health_probe_timeout_s: int=NotProvided, worker_restore_timeout_s: int=NotProvided):
        if False:
            for i in range(10):
                print('nop')
        "Sets the config's fault tolerance settings.\n\n        Args:\n            recreate_failed_workers: Whether - upon a worker failure - RLlib will try to\n                recreate the lost worker as an identical copy of the failed one. The new\n                worker will only differ from the failed one in its\n                `self.recreated_worker=True` property value. It will have the same\n                `worker_index` as the original one. If True, the\n                `ignore_worker_failures` setting will be ignored.\n            max_num_worker_restarts: The maximum number of times a worker is allowed to\n                be restarted (if `recreate_failed_workers` is True).\n            delay_between_worker_restarts_s: The delay (in seconds) between two\n                consecutive worker restarts (if `recreate_failed_workers` is True).\n            restart_failed_sub_environments: If True and any sub-environment (within\n                a vectorized env) throws any error during env stepping, the\n                Sampler will try to restart the faulty sub-environment. This is done\n                without disturbing the other (still intact) sub-environment and without\n                the EnvRunner crashing.\n            num_consecutive_worker_failures_tolerance: The number of consecutive times\n                a rollout worker (or evaluation worker) failure is tolerated before\n                finally crashing the Algorithm. Only useful if either\n                `ignore_worker_failures` or `recreate_failed_workers` is True.\n                Note that for `restart_failed_sub_environments` and sub-environment\n                failures, the worker itself is NOT affected and won't throw any errors\n                as the flawed sub-environment is silently restarted under the hood.\n            worker_health_probe_timeout_s: Max amount of time we should spend waiting\n                for health probe calls to finish. Health pings are very cheap, so the\n                default is 1 minute.\n            worker_restore_timeout_s: Max amount of time we should wait to restore\n                states on recovered worker actors. Default is 30 mins.\n\n        Returns:\n            This updated AlgorithmConfig object.\n        "
        if recreate_failed_workers is not NotProvided:
            self.recreate_failed_workers = recreate_failed_workers
        if max_num_worker_restarts is not NotProvided:
            self.max_num_worker_restarts = max_num_worker_restarts
        if delay_between_worker_restarts_s is not NotProvided:
            self.delay_between_worker_restarts_s = delay_between_worker_restarts_s
        if restart_failed_sub_environments is not NotProvided:
            self.restart_failed_sub_environments = restart_failed_sub_environments
        if num_consecutive_worker_failures_tolerance is not NotProvided:
            self.num_consecutive_worker_failures_tolerance = num_consecutive_worker_failures_tolerance
        if worker_health_probe_timeout_s is not NotProvided:
            self.worker_health_probe_timeout_s = worker_health_probe_timeout_s
        if worker_restore_timeout_s is not NotProvided:
            self.worker_restore_timeout_s = worker_restore_timeout_s
        return self

    @ExperimentalAPI
    def rl_module(self, *, rl_module_spec: Optional[ModuleSpec]=NotProvided, _enable_rl_module_api: Optional[bool]=NotProvided) -> 'AlgorithmConfig':
        if False:
            for i in range(10):
                print('nop')
        "Sets the config's RLModule settings.\n\n        Args:\n            rl_module_spec: The RLModule spec to use for this config. It can be either\n                a SingleAgentRLModuleSpec or a MultiAgentRLModuleSpec. If the\n                observation_space, action_space, catalog_class, or the model config is\n                not specified it will be inferred from the env and other parts of the\n                algorithm config object.\n\n        Returns:\n            This updated AlgorithmConfig object.\n        "
        if rl_module_spec is not NotProvided:
            self._rl_module_spec = rl_module_spec
        if _enable_rl_module_api is not NotProvided:
            deprecation_warning(old='AlgorithmConfig.rl_module(_enable_rl_module_api=True|False)', new='AlgorithmConfig.experimental(_enable_new_api_stack=True|False)', error=True)
        return self

    def experimental(self, *, _enable_new_api_stack: Optional[bool]=NotProvided, _tf_policy_handles_more_than_one_loss: Optional[bool]=NotProvided, _disable_preprocessor_api: Optional[bool]=NotProvided, _disable_action_flattening: Optional[bool]=NotProvided, _disable_execution_plan_api: Optional[bool]=NotProvided, _disable_initialize_loss_from_dummy_batch: Optional[bool]=NotProvided) -> 'AlgorithmConfig':
        if False:
            return 10
        "Sets the config's experimental settings.\n\n        Args:\n            _enable_new_api_stack: Enables the new API stack, which will use RLModule\n                (instead of ModelV2) as well as the multi-GPU capable Learner API\n                (instead of using Policy to compute loss and update the model).\n            _tf_policy_handles_more_than_one_loss: Experimental flag.\n                If True, TFPolicy will handle more than one loss/optimizer.\n                Set this to True, if you would like to return more than\n                one loss term from your `loss_fn` and an equal number of optimizers\n                from your `optimizer_fn`. In the future, the default for this will be\n                True.\n            _disable_preprocessor_api: Experimental flag.\n                If True, no (observation) preprocessor will be created and\n                observations will arrive in model as they are returned by the env.\n                In the future, the default for this will be True.\n            _disable_action_flattening: Experimental flag.\n                If True, RLlib will no longer flatten the policy-computed actions into\n                a single tensor (for storage in SampleCollectors/output files/etc..),\n                but leave (possibly nested) actions as-is. Disabling flattening affects:\n                - SampleCollectors: Have to store possibly nested action structs.\n                - Models that have the previous action(s) as part of their input.\n                - Algorithms reading from offline files (incl. action information).\n            _disable_execution_plan_api: Experimental flag.\n                If True, the execution plan API will not be used. Instead,\n                a Algorithm's `training_iteration` method will be called as-is each\n                training iteration.\n\n        Returns:\n            This updated AlgorithmConfig object.\n        "
        if _enable_new_api_stack is not NotProvided:
            self._enable_new_api_stack = _enable_new_api_stack
            if _enable_new_api_stack is True and self.exploration_config:
                self.__prior_exploration_config = self.exploration_config
                self.exploration_config = {}
            elif _enable_new_api_stack is False and (not self.exploration_config):
                if self.__prior_exploration_config is not None:
                    self.exploration_config = self.__prior_exploration_config
                    self.__prior_exploration_config = None
                else:
                    logger.warning('config._enable_new_api_stack was set to False, but no prior exploration config was found to be restored.')
        if _tf_policy_handles_more_than_one_loss is not NotProvided:
            self._tf_policy_handles_more_than_one_loss = _tf_policy_handles_more_than_one_loss
        if _disable_preprocessor_api is not NotProvided:
            self._disable_preprocessor_api = _disable_preprocessor_api
        if _disable_action_flattening is not NotProvided:
            self._disable_action_flattening = _disable_action_flattening
        if _disable_execution_plan_api is not NotProvided:
            self._disable_execution_plan_api = _disable_execution_plan_api
        if _disable_initialize_loss_from_dummy_batch is not NotProvided:
            self._disable_initialize_loss_from_dummy_batch = _disable_initialize_loss_from_dummy_batch
        return self

    @property
    def rl_module_spec(self):
        if False:
            while True:
                i = 10
        default_rl_module_spec = self.get_default_rl_module_spec()
        _check_rl_module_spec(default_rl_module_spec)
        if self._rl_module_spec is not None:
            _check_rl_module_spec(self._rl_module_spec)
            if isinstance(self._rl_module_spec, SingleAgentRLModuleSpec):
                if isinstance(default_rl_module_spec, SingleAgentRLModuleSpec):
                    default_rl_module_spec.update(self._rl_module_spec)
                    return default_rl_module_spec
                elif isinstance(default_rl_module_spec, MultiAgentRLModuleSpec):
                    raise ValueError('Cannot merge MultiAgentRLModuleSpec with SingleAgentRLModuleSpec!')
        else:
            return default_rl_module_spec

    @property
    def learner_class(self) -> Type['Learner']:
        if False:
            i = 10
            return i + 15
        'Returns the Learner sub-class to use by this Algorithm.\n\n        Either\n        a) User sets a specific learner class via calling `.training(learner_class=...)`\n        b) User leaves learner class unset (None) and the AlgorithmConfig itself\n        figures out the actual learner class by calling its own\n        `.get_default_learner_class()` method.\n        '
        return self._learner_class or self.get_default_learner_class()

    @property
    def is_atari(self) -> bool:
        if False:
            print('Hello World!')
        'True if if specified env is an Atari env.'
        if self._is_atari is None:
            if type(self.env) is not str:
                return False
            try:
                env = gym.make(self.env)
            except gym.error.Error:
                return False
            self._is_atari = is_atari(env)
            env.close()
        return self._is_atari

    def get_rollout_fragment_length(self, worker_index: int=0) -> int:
        if False:
            i = 10
            return i + 15
        'Automatically infers a proper rollout_fragment_length setting if "auto".\n\n        Uses the simple formula:\n        `rollout_fragment_length` = `train_batch_size` /\n        (`num_envs_per_worker` * `num_rollout_workers`)\n\n        If result is not a fraction AND `worker_index` is provided, will make\n        those workers add another timestep, such that the overall batch size (across\n        the workers) will add up to exactly the `train_batch_size`.\n\n        Returns:\n            The user-provided `rollout_fragment_length` or a computed one (if user\n            value is "auto").\n        '
        if self.rollout_fragment_length == 'auto':
            rollout_fragment_length = self.train_batch_size / (self.num_envs_per_worker * (self.num_rollout_workers or 1))
            if int(rollout_fragment_length) != rollout_fragment_length:
                diff = self.train_batch_size - int(rollout_fragment_length) * self.num_envs_per_worker * (self.num_rollout_workers or 1)
                if worker_index * self.num_envs_per_worker <= diff:
                    return int(rollout_fragment_length) + 1
            return int(rollout_fragment_length)
        else:
            return self.rollout_fragment_length

    def get_evaluation_config_object(self) -> Optional['AlgorithmConfig']:
        if False:
            print('Hello World!')
        'Creates a full AlgorithmConfig object from `self.evaluation_config`.\n\n        Returns:\n            A fully valid AlgorithmConfig object that can be used for the evaluation\n            WorkerSet. If `self` is already an evaluation config object, return None.\n        '
        if self.in_evaluation:
            assert self.evaluation_config is None
            return None
        evaluation_config = self.evaluation_config
        if isinstance(evaluation_config, AlgorithmConfig):
            eval_config_obj = evaluation_config.copy(copy_frozen=False)
        else:
            eval_config_obj = self.copy(copy_frozen=False)
            eval_config_obj.update_from_dict(evaluation_config or {})
        eval_config_obj.in_evaluation = True
        eval_config_obj.evaluation_config = None
        if self.evaluation_duration_unit == 'episodes':
            eval_config_obj.batch_mode = 'complete_episodes'
            eval_config_obj.rollout_fragment_length = 1
        else:
            eval_config_obj.batch_mode = 'truncate_episodes'
            eval_config_obj.rollout_fragment_length = 10 if self.evaluation_duration == 'auto' else int(math.ceil(self.evaluation_duration / (self.evaluation_num_workers or 1)))
        return eval_config_obj

    def get_multi_agent_setup(self, *, policies: Optional[MultiAgentPolicyConfigDict]=None, env: Optional[EnvType]=None, spaces: Optional[Dict[PolicyID, Tuple[Space, Space]]]=None, default_policy_class: Optional[Type[Policy]]=None) -> Tuple[MultiAgentPolicyConfigDict, Callable[[PolicyID, SampleBatchType], bool]]:
        if False:
            i = 10
            return i + 15
        'Compiles complete multi-agent config (dict) from the information in `self`.\n\n        Infers the observation- and action spaces, the policy classes, and the policy\'s\n        configs. The returned `MultiAgentPolicyConfigDict` is fully unified and strictly\n        maps PolicyIDs to complete PolicySpec objects (with all their fields not-None).\n\n        Examples:\n        .. testcode::\n\n            import gymnasium as gym\n            from ray.rllib.algorithms.ppo import PPOConfig\n            config = (\n              PPOConfig()\n              .environment("CartPole-v1")\n              .framework("torch")\n              .multi_agent(policies={"pol1", "pol2"}, policies_to_train=["pol1"])\n            )\n            policy_dict, is_policy_to_train = config.get_multi_agent_setup(\n                env=gym.make("CartPole-v1"))\n            is_policy_to_train("pol1")\n            is_policy_to_train("pol2")\n\n        Args:\n            policies: An optional multi-agent `policies` dict, mapping policy IDs\n                to PolicySpec objects. If not provided, will use `self.policies`\n                instead. Note that the `policy_class`, `observation_space`, and\n                `action_space` properties in these PolicySpecs may be None and must\n                therefore be inferred here.\n            env: An optional env instance, from which to infer the different spaces for\n                the different policies. If not provided, will try to infer from\n                `spaces`. Otherwise from `self.observation_space` and\n                `self.action_space`. If no information on spaces can be infered, will\n                raise an error.\n            spaces: Optional dict mapping policy IDs to tuples of 1) observation space\n                and 2) action space that should be used for the respective policy.\n                These spaces were usually provided by an already instantiated remote\n                EnvRunner. If not provided, will try to infer from `env`. Otherwise\n                from `self.observation_space` and `self.action_space`. If no\n                information on spaces can be inferred, will raise an error.\n            default_policy_class: The Policy class to use should a PolicySpec have its\n                policy_class property set to None.\n\n        Returns:\n            A tuple consisting of 1) a MultiAgentPolicyConfigDict and 2) a\n            `is_policy_to_train(PolicyID, SampleBatchType) -> bool` callable.\n\n        Raises:\n            ValueError: In case, no spaces can be infered for the policy/ies.\n            ValueError: In case, two agents in the env map to the same PolicyID\n                (according to `self.policy_mapping_fn`), but have different action- or\n                observation spaces according to the infered space information.\n        '
        policies = copy.deepcopy(policies or self.policies)
        if isinstance(policies, (set, list, tuple)):
            policies = {pid: PolicySpec() for pid in policies}
        env_obs_space = None
        env_act_space = None
        if isinstance(env, ray.actor.ActorHandle):
            (env_obs_space, env_act_space) = ray.get(env._get_spaces.remote())
        elif env is not None:
            if hasattr(env, 'single_observation_space') and isinstance(env.single_observation_space, gym.Space):
                env_obs_space = env.single_observation_space
            elif hasattr(env, 'observation_space') and isinstance(env.observation_space, gym.Space):
                env_obs_space = env.observation_space
            if hasattr(env, 'single_action_space') and isinstance(env.single_action_space, gym.Space):
                env_act_space = env.single_action_space
            elif hasattr(env, 'action_space') and isinstance(env.action_space, gym.Space):
                env_act_space = env.action_space
        if spaces is not None:
            if env_obs_space is None:
                env_obs_space = spaces.get('__env__', [None])[0]
            if env_act_space is None:
                env_act_space = spaces.get('__env__', [None, None])[1]
        for (pid, policy_spec) in policies.copy().items():
            if not isinstance(policy_spec, PolicySpec):
                policies[pid] = policy_spec = PolicySpec(*policy_spec)
            if policy_spec.policy_class is None and default_policy_class is not None:
                policies[pid].policy_class = default_policy_class
            if old_gym and isinstance(policy_spec.observation_space, old_gym.Space):
                policies[pid].observation_space = convert_old_gym_space_to_gymnasium_space(policy_spec.observation_space)
            elif policy_spec.observation_space is None:
                if spaces is not None and pid in spaces:
                    obs_space = spaces[pid][0]
                elif env_obs_space is not None:
                    if isinstance(env, MultiAgentEnv) and hasattr(env, '_obs_space_in_preferred_format') and env._obs_space_in_preferred_format:
                        obs_space = None
                        mapping_fn = self.policy_mapping_fn
                        one_obs_space = next(iter(env_obs_space.values()))
                        if all((s == one_obs_space for s in env_obs_space.values())):
                            obs_space = one_obs_space
                        elif mapping_fn:
                            for aid in env.get_agent_ids():
                                if mapping_fn(aid, None, worker=None) == pid:
                                    if obs_space is not None and env_obs_space[aid] != obs_space:
                                        raise ValueError('Two agents in your environment map to the same policyID (as per your `policy_mapping_fn`), however, these agents also have different observation spaces!')
                                    obs_space = env_obs_space[aid]
                    else:
                        obs_space = env_obs_space
                elif self.observation_space:
                    obs_space = self.observation_space
                else:
                    raise ValueError(f"`observation_space` not provided in PolicySpec for {pid} and env does not have an observation space OR no spaces received from other workers' env(s) OR no `observation_space` specified in config!")
                policies[pid].observation_space = obs_space
            if old_gym and isinstance(policy_spec.action_space, old_gym.Space):
                policies[pid].action_space = convert_old_gym_space_to_gymnasium_space(policy_spec.action_space)
            elif policy_spec.action_space is None:
                if spaces is not None and pid in spaces:
                    act_space = spaces[pid][1]
                elif env_act_space is not None:
                    if isinstance(env, MultiAgentEnv) and hasattr(env, '_action_space_in_preferred_format') and env._action_space_in_preferred_format:
                        act_space = None
                        mapping_fn = self.policy_mapping_fn
                        one_act_space = next(iter(env_act_space.values()))
                        if all((s == one_act_space for s in env_act_space.values())):
                            act_space = one_act_space
                        elif mapping_fn:
                            for aid in env.get_agent_ids():
                                if mapping_fn(aid, None, worker=None) == pid:
                                    if act_space is not None and env_act_space[aid] != act_space:
                                        raise ValueError('Two agents in your environment map to the same policyID (as per your `policy_mapping_fn`), however, these agents also have different action spaces!')
                                    act_space = env_act_space[aid]
                    else:
                        act_space = env_act_space
                elif self.action_space:
                    act_space = self.action_space
                else:
                    raise ValueError(f"`action_space` not provided in PolicySpec for {pid} and env does not have an action space OR no spaces received from other workers' env(s) OR no `action_space` specified in config!")
                policies[pid].action_space = act_space
            if not isinstance(policies[pid].config, AlgorithmConfig):
                assert policies[pid].config is None or isinstance(policies[pid].config, dict)
                policies[pid].config = self.copy(copy_frozen=False).update_from_dict(policies[pid].config or {})
        if self.policies_to_train is not None and (not callable(self.policies_to_train)):
            pols = set(self.policies_to_train)

            def is_policy_to_train(pid, batch=None):
                if False:
                    return 10
                return pid in pols
        else:
            is_policy_to_train = self.policies_to_train
        return (policies, is_policy_to_train)

    def validate_train_batch_size_vs_rollout_fragment_length(self) -> None:
        if False:
            print('Hello World!')
        'Detects mismatches for `train_batch_size` vs `rollout_fragment_length`.\n\n        Only applicable for algorithms, whose train_batch_size should be directly\n        dependent on rollout_fragment_length (synchronous sampling, on-policy PG algos).\n\n        If rollout_fragment_length != "auto", makes sure that the product of\n        `rollout_fragment_length` x `num_rollout_workers` x `num_envs_per_worker`\n        roughly (10%) matches the provided `train_batch_size`. Otherwise, errors with\n        asking the user to set rollout_fragment_length to `auto` or to a matching\n        value.\n\n        Also, only checks this if `train_batch_size` > 0 (DDPPO sets this\n        to -1 to auto-calculate the actual batch size later).\n\n        Raises:\n            ValueError: If there is a mismatch between user provided\n            `rollout_fragment_length` and `train_batch_size`.\n        '
        if self.rollout_fragment_length != 'auto' and (not self.in_evaluation) and (self.train_batch_size > 0):
            min_batch_size = max(self.num_rollout_workers, 1) * self.num_envs_per_worker * self.rollout_fragment_length
            batch_size = min_batch_size
            while batch_size < self.train_batch_size:
                batch_size += min_batch_size
            if batch_size - self.train_batch_size > 0.1 * self.train_batch_size or batch_size - min_batch_size - self.train_batch_size > 0.1 * self.train_batch_size:
                suggested_rollout_fragment_length = self.train_batch_size // (self.num_envs_per_worker * (self.num_rollout_workers or 1))
                raise ValueError(f"Your desired `train_batch_size` ({self.train_batch_size}) or a value 10% off of that cannot be achieved with your other settings (num_rollout_workers={self.num_rollout_workers}; num_envs_per_worker={self.num_envs_per_worker}; rollout_fragment_length={self.rollout_fragment_length})! Try setting `rollout_fragment_length` to 'auto' OR {suggested_rollout_fragment_length}.")

    def get_torch_compile_learner_config(self):
        if False:
            i = 10
            return i + 15
        'Returns the TorchCompileConfig to use on learners.'
        from ray.rllib.core.rl_module.torch.torch_compile_config import TorchCompileConfig
        return TorchCompileConfig(torch_dynamo_backend=self.torch_compile_learner_dynamo_backend, torch_dynamo_mode=self.torch_compile_learner_dynamo_mode)

    def get_torch_compile_worker_config(self):
        if False:
            return 10
        'Returns the TorchCompileConfig to use on workers.'
        from ray.rllib.core.rl_module.torch.torch_compile_config import TorchCompileConfig
        return TorchCompileConfig(torch_dynamo_backend=self.torch_compile_worker_dynamo_backend, torch_dynamo_mode=self.torch_compile_worker_dynamo_mode)

    def get_default_rl_module_spec(self) -> ModuleSpec:
        if False:
            for i in range(10):
                print('nop')
        "Returns the RLModule spec to use for this algorithm.\n\n        Override this method in the sub-class to return the RLModule spec given\n        the input framework.\n\n        Returns:\n            The ModuleSpec (SingleAgentRLModuleSpec or MultiAgentRLModuleSpec) to use\n            for this algorithm's RLModule.\n        "
        raise NotImplementedError

    def get_default_learner_class(self) -> Union[Type['Learner'], str]:
        if False:
            while True:
                i = 10
        'Returns the Learner class to use for this algorithm.\n\n        Override this method in the sub-class to return the Learner class type given\n        the input framework.\n\n        Returns:\n            The Learner class to use for this algorithm either as a class type or as\n            a string (e.g. ray.rllib.core.learner.testing.torch.BC).\n        '
        raise NotImplementedError

    def get_marl_module_spec(self, *, policy_dict: Dict[str, PolicySpec], single_agent_rl_module_spec: Optional[SingleAgentRLModuleSpec]=None) -> MultiAgentRLModuleSpec:
        if False:
            while True:
                i = 10
        "Returns the MultiAgentRLModule spec based on the given policy spec dict.\n\n        policy_dict could be a partial dict of the policies that we need to turn into\n        an equivalent multi-agent RLModule spec.\n\n        Args:\n            policy_dict: The policy spec dict. Using this dict, we can determine the\n                inferred values for observation_space, action_space, and config for\n                each policy. If the module spec does not have these values specified,\n                they will get auto-filled with these values obtrained from the policy\n                spec dict. Here we are relying on the policy's logic for infering these\n                values from other sources of information (e.g. environement)\n            single_agent_rl_module_spec: The SingleAgentRLModuleSpec to use for\n                constructing a MultiAgentRLModuleSpec. If None, the already\n                configured spec (`self._rl_module_spec`) or the default ModuleSpec for\n                this algorithm (`self.get_default_rl_module_spec()`) will be used.\n        "
        default_rl_module_spec = self.get_default_rl_module_spec()
        current_rl_module_spec = self._rl_module_spec or default_rl_module_spec
        if isinstance(current_rl_module_spec, SingleAgentRLModuleSpec):
            single_agent_rl_module_spec = single_agent_rl_module_spec or current_rl_module_spec
            marl_module_spec = MultiAgentRLModuleSpec(module_specs={k: copy.deepcopy(single_agent_rl_module_spec) for k in policy_dict.keys()})
        else:
            assert isinstance(current_rl_module_spec, MultiAgentRLModuleSpec)
            if isinstance(default_rl_module_spec, SingleAgentRLModuleSpec):
                if isinstance(current_rl_module_spec.module_specs, SingleAgentRLModuleSpec):
                    single_agent_spec = single_agent_rl_module_spec or current_rl_module_spec.module_specs
                    module_specs = {k: copy.deepcopy(single_agent_spec) for k in policy_dict.keys()}
                else:
                    single_agent_spec = single_agent_rl_module_spec or default_rl_module_spec
                    module_specs = {k: copy.deepcopy(current_rl_module_spec.module_specs.get(k, single_agent_spec)) for k in policy_dict.keys()}
                marl_module_spec = current_rl_module_spec.__class__(marl_module_class=current_rl_module_spec.marl_module_class, module_specs=module_specs, modules_to_load=current_rl_module_spec.modules_to_load, load_state_path=current_rl_module_spec.load_state_path)
            else:
                if single_agent_rl_module_spec is not None:
                    pass
                elif isinstance(current_rl_module_spec.module_specs, SingleAgentRLModuleSpec):
                    single_agent_rl_module_spec = current_rl_module_spec.module_specs
                else:
                    raise ValueError(f"We have a MultiAgentRLModuleSpec ({current_rl_module_spec}), but no `SingleAgentRLModuleSpec`s to compile the individual RLModules' specs! Use `AlgorithmConfig.get_marl_module_spec(policy_dict=.., single_agent_rl_module_spec=..)`.")
                marl_module_spec = current_rl_module_spec.__class__(marl_module_class=current_rl_module_spec.marl_module_class, module_specs={k: copy.deepcopy(single_agent_rl_module_spec) for k in policy_dict.keys()}, modules_to_load=current_rl_module_spec.modules_to_load, load_state_path=current_rl_module_spec.load_state_path)
        if set(policy_dict.keys()) != set(marl_module_spec.module_specs.keys()):
            raise ValueError(f'Policy dict and module spec have different keys! \npolicy_dict keys: {list(policy_dict.keys())} \nmodule_spec keys: {list(marl_module_spec.module_specs.keys())}')
        for module_id in policy_dict:
            policy_spec = policy_dict[module_id]
            module_spec = marl_module_spec.module_specs[module_id]
            if module_spec.module_class is None:
                if isinstance(default_rl_module_spec, SingleAgentRLModuleSpec):
                    module_spec.module_class = default_rl_module_spec.module_class
                elif isinstance(default_rl_module_spec.module_specs, SingleAgentRLModuleSpec):
                    module_class = default_rl_module_spec.module_specs.module_class
                    if module_class is None:
                        raise ValueError('The default rl_module spec cannot have an empty module_class under its SingleAgentRLModuleSpec.')
                    module_spec.module_class = module_class
                elif module_id in default_rl_module_spec.module_specs:
                    module_spec.module_class = default_rl_module_spec.module_specs[module_id].module_class
                else:
                    raise ValueError(f'Module class for module {module_id} cannot be inferred. It is neither provided in the rl_module_spec that is passed in nor in the default module spec used in the algorithm.')
            if module_spec.catalog_class is None:
                if isinstance(default_rl_module_spec, SingleAgentRLModuleSpec):
                    module_spec.catalog_class = default_rl_module_spec.catalog_class
                elif isinstance(default_rl_module_spec.module_specs, SingleAgentRLModuleSpec):
                    catalog_class = default_rl_module_spec.module_specs.catalog_class
                    module_spec.catalog_class = catalog_class
                elif module_id in default_rl_module_spec.module_specs:
                    module_spec.catalog_class = default_rl_module_spec.module_specs[module_id].catalog_class
                else:
                    raise ValueError(f'Catalog class for module {module_id} cannot be inferred. It is neither provided in the rl_module_spec that is passed in nor in the default module spec used in the algorithm.')
            if module_spec.observation_space is None:
                module_spec.observation_space = policy_spec.observation_space
            if module_spec.action_space is None:
                module_spec.action_space = policy_spec.action_space
            if module_spec.model_config_dict is None:
                module_spec.model_config_dict = policy_spec.config.get('model', {})
        return marl_module_spec

    def get_learner_group_config(self, module_spec: ModuleSpec) -> LearnerGroupConfig:
        if False:
            print('Hello World!')
        if not self._is_frozen:
            raise ValueError('Cannot call `get_learner_group_config()` on an unfrozen AlgorithmConfig! Please call `AlgorithmConfig.freeze()` first.')
        config = LearnerGroupConfig().module(module_spec).learner(learner_class=self.learner_class, learner_hyperparameters=self.get_learner_hyperparameters()).resources(num_learner_workers=self.num_learner_workers, num_cpus_per_learner_worker=self.num_cpus_per_learner_worker if not self.num_gpus_per_learner_worker else 0, num_gpus_per_learner_worker=self.num_gpus_per_learner_worker, local_gpu_idx=self.local_gpu_idx)
        if self.framework_str == 'torch':
            config.framework(torch_compile=self.torch_compile_learner, torch_compile_cfg=self.get_torch_compile_learner_config(), torch_compile_what_to_compile=self.torch_compile_learner_what_to_compile)
        elif self.framework_str == 'tf2':
            config.framework(eager_tracing=self.eager_tracing)
        return config

    def get_learner_hyperparameters(self) -> LearnerHyperparameters:
        if False:
            print('Hello World!')
        "Returns a new LearnerHyperparameters instance for the respective Learner.\n\n        The LearnerHyperparameters is a dataclass containing only those config settings\n        from AlgorithmConfig that are used by the algorithm's specific Learner\n        sub-class. They allow distributing only those settings relevant for learning\n        across a set of learner workers (instead of having to distribute the entire\n        AlgorithmConfig object).\n\n        Note that LearnerHyperparameters should always be derived directly from a\n        AlgorithmConfig object's own settings and considered frozen/read-only.\n\n        Returns:\n             A LearnerHyperparameters instance for the respective Learner.\n        "
        per_module_learner_hp_overrides = {}
        if self.algorithm_config_overrides_per_module:
            for (module_id, overrides) in self.algorithm_config_overrides_per_module.items():
                config_for_module = self.copy(copy_frozen=False).update_from_dict(overrides)
                config_for_module.algorithm_config_overrides_per_module = None
                per_module_learner_hp_overrides[module_id] = config_for_module.get_learner_hyperparameters()
        return LearnerHyperparameters(learning_rate=self.lr, grad_clip=self.grad_clip, grad_clip_by=self.grad_clip_by, _per_module_overrides=per_module_learner_hp_overrides, seed=self.seed)

    def __setattr__(self, key, value):
        if False:
            print('Hello World!')
        'Gatekeeper in case we are in frozen state and need to error.'
        if hasattr(self, '_is_frozen') and self._is_frozen:
            if key not in ['simple_optimizer', 'worker_index', '_is_frozen']:
                raise AttributeError(f'Cannot set attribute ({key}) of an already frozen AlgorithmConfig!')
        if key == 'rl_module_spec':
            key = '_rl_module_spec'
        super().__setattr__(key, value)

    def __getitem__(self, item):
        if False:
            i = 10
            return i + 15
        'Shim method to still support accessing properties by key lookup.\n\n        This way, an AlgorithmConfig object can still be used as if a dict, e.g.\n        by Ray Tune.\n\n        Examples:\n            .. testcode::\n\n                from ray.rllib.algorithms.algorithm_config import AlgorithmConfig\n                config = AlgorithmConfig()\n                print(config["lr"])\n\n            .. testoutput::\n\n                0.001\n        '
        item = self._translate_special_keys(item)
        return getattr(self, item)

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        if key == 'multiagent':
            raise AttributeError('Cannot set `multiagent` key in an AlgorithmConfig!\nTry setting the multi-agent components of your AlgorithmConfig object via the `multi_agent()` method and its arguments.\nE.g. `config.multi_agent(policies=.., policy_mapping_fn.., policies_to_train=..)`.')
        super().__setattr__(key, value)

    def __contains__(self, item) -> bool:
        if False:
            while True:
                i = 10
        'Shim method to help pretend we are a dict.'
        prop = self._translate_special_keys(item, warn_deprecated=False)
        return hasattr(self, prop)

    def get(self, key, default=None):
        if False:
            for i in range(10):
                print('nop')
        'Shim method to help pretend we are a dict.'
        prop = self._translate_special_keys(key, warn_deprecated=False)
        return getattr(self, prop, default)

    def pop(self, key, default=None):
        if False:
            for i in range(10):
                print('nop')
        'Shim method to help pretend we are a dict.'
        return self.get(key, default)

    def keys(self):
        if False:
            print('Hello World!')
        'Shim method to help pretend we are a dict.'
        return self.to_dict().keys()

    def values(self):
        if False:
            return 10
        'Shim method to help pretend we are a dict.'
        return self.to_dict().values()

    def items(self):
        if False:
            print('Hello World!')
        'Shim method to help pretend we are a dict.'
        return self.to_dict().items()

    @staticmethod
    def _serialize_dict(config):
        if False:
            i = 10
            return i + 15
        config['callbacks'] = serialize_type(config['callbacks'])
        config['sample_collector'] = serialize_type(config['sample_collector'])
        if isinstance(config['env'], type):
            config['env'] = serialize_type(config['env'])
        if 'replay_buffer_config' in config and isinstance(config['replay_buffer_config'].get('type'), type):
            config['replay_buffer_config']['type'] = serialize_type(config['replay_buffer_config']['type'])
        if isinstance(config['exploration_config'].get('type'), type):
            config['exploration_config']['type'] = serialize_type(config['exploration_config']['type'])
        if isinstance(config['model'].get('custom_model'), type):
            config['model']['custom_model'] = serialize_type(config['model']['custom_model'])
        ma_config = config.get('multiagent')
        if ma_config is not None:
            if isinstance(ma_config.get('policies'), (set, tuple)):
                ma_config['policies'] = list(ma_config['policies'])
            if ma_config.get('policy_mapping_fn'):
                ma_config['policy_mapping_fn'] = NOT_SERIALIZABLE
            if ma_config.get('policies_to_train'):
                ma_config['policies_to_train'] = NOT_SERIALIZABLE
        if isinstance(config.get('policies'), (set, tuple)):
            config['policies'] = list(config['policies'])
        if config.get('policy_mapping_fn'):
            config['policy_mapping_fn'] = NOT_SERIALIZABLE
        if config.get('policies_to_train'):
            config['policies_to_train'] = NOT_SERIALIZABLE
        return config

    @staticmethod
    def _translate_special_keys(key: str, warn_deprecated: bool=True) -> str:
        if False:
            for i in range(10):
                print('nop')
        if key == 'callbacks':
            key = 'callbacks_class'
        elif key == 'create_env_on_driver':
            key = 'create_env_on_local_worker'
        elif key == 'custom_eval_function':
            key = 'custom_evaluation_function'
        elif key == 'framework':
            key = 'framework_str'
        elif key == 'input':
            key = 'input_'
        elif key == 'lambda':
            key = 'lambda_'
        elif key == 'num_cpus_for_driver':
            key = 'num_cpus_for_local_worker'
        elif key == 'num_workers':
            key = 'num_rollout_workers'
        if warn_deprecated:
            if key == 'collect_metrics_timeout':
                deprecation_warning(old='collect_metrics_timeout', new='metrics_episode_collection_timeout_s', error=True)
            elif key == 'metrics_smoothing_episodes':
                deprecation_warning(old='config.metrics_smoothing_episodes', new='config.metrics_num_episodes_for_smoothing', error=True)
            elif key == 'min_iter_time_s':
                deprecation_warning(old='config.min_iter_time_s', new='config.min_time_s_per_iteration', error=True)
            elif key == 'min_time_s_per_reporting':
                deprecation_warning(old='config.min_time_s_per_reporting', new='config.min_time_s_per_iteration', error=True)
            elif key == 'min_sample_timesteps_per_reporting':
                deprecation_warning(old='config.min_sample_timesteps_per_reporting', new='config.min_sample_timesteps_per_iteration', error=True)
            elif key == 'min_train_timesteps_per_reporting':
                deprecation_warning(old='config.min_train_timesteps_per_reporting', new='config.min_train_timesteps_per_iteration', error=True)
            elif key == 'timesteps_per_iteration':
                deprecation_warning(old='config.timesteps_per_iteration', new='`config.min_sample_timesteps_per_iteration` OR `config.min_train_timesteps_per_iteration`', error=True)
            elif key == 'evaluation_num_episodes':
                deprecation_warning(old='config.evaluation_num_episodes', new='`config.evaluation_duration` and `config.evaluation_duration_unit=episodes`', error=True)
        return key

    def _check_if_correct_nn_framework_installed(self, _tf1, _tf, _torch):
        if False:
            for i in range(10):
                print('nop')
        'Check if tf/torch experiment is running and tf/torch installed.'
        if self.framework_str in {'tf', 'tf2'}:
            if not (_tf1 or _tf):
                raise ImportError('TensorFlow was specified as the framework to use (via `config.framework([tf|tf2])`)! However, no installation was found. You can install TensorFlow via `pip install tensorflow`')
        elif self.framework_str == 'torch':
            if not _torch:
                raise ImportError("PyTorch was specified as the framework to use (via `config.framework('torch')`)! However, no installation was found. You can install PyTorch via `pip install torch`.")

    def _resolve_tf_settings(self, _tf1, _tfv):
        if False:
            return 10
        'Check and resolve tf settings.'
        if _tf1 and self.framework_str == 'tf2':
            if self.framework_str == 'tf2' and _tfv < 2:
                raise ValueError('You configured `framework`=tf2, but your installed pip tf-version is < 2.0! Make sure your TensorFlow version is >= 2.x.')
            if not _tf1.executing_eagerly():
                _tf1.enable_eager_execution()
            logger.info(f"Executing eagerly (framework='{self.framework_str}'), with eager_tracing={self.eager_tracing}. For production workloads, make sure to set eager_tracing=True  in order to match the speed of tf-static-graph (framework='tf'). For debugging purposes, `eager_tracing=False` is the best choice.")
        elif _tf1 and self.framework_str == 'tf':
            logger.info("Your framework setting is 'tf', meaning you are using static-graph mode. Set framework='tf2' to enable eager execution with tf2.x. You may also then want to set eager_tracing=True in order to reach similar execution speed as with static-graph mode.")

    @property
    @Deprecated(old="AlgorithmConfig.multiagent['[some key]']", new='AlgorithmConfig.[some key]', error=True)
    def multiagent(self):
        if False:
            i = 10
            return i + 15
        pass

    @property
    @Deprecated(new='AlgorithmConfig.rollouts(num_rollout_workers=..)', error=True)
    def num_workers(self):
        if False:
            print('Hello World!')
        pass