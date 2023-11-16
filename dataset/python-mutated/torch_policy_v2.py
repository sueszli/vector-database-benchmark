import copy
import functools
import logging
import math
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type, Union
import gymnasium as gym
import numpy as np
import tree
import ray
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.core.rl_module import RLModule
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import _directStepOptimizerSingleton
from ray.rllib.utils import NullContextManager, force_list
from ray.rllib.utils.annotations import DeveloperAPI, OverrideToImplementCustomLogic, OverrideToImplementCustomLogic_CallToSuperRecommended, is_overridden, override
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.error import ERR_MSG_TORCH_POLICY_CANNOT_SAVE_MODEL
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY, NUM_AGENT_STEPS_TRAINED, NUM_GRAD_UPDATES_LIFETIME
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import AlgorithmConfigDict, GradInfoDict, ModelGradients, ModelWeights, PolicyState, TensorStructType, TensorType
if TYPE_CHECKING:
    from ray.rllib.evaluation import Episode
(torch, nn) = try_import_torch()
logger = logging.getLogger(__name__)

@DeveloperAPI
class TorchPolicyV2(Policy):
    """PyTorch specific Policy class to use with RLlib."""

    @DeveloperAPI
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict, *, max_seq_len: int=20):
        if False:
            for i in range(10):
                print('nop')
        "Initializes a TorchPolicy instance.\n\n        Args:\n            observation_space: Observation space of the policy.\n            action_space: Action space of the policy.\n            config: The Policy's config dict.\n            max_seq_len: Max sequence length for LSTM training.\n        "
        self.framework = config['framework'] = 'torch'
        self._loss_initialized = False
        super().__init__(observation_space, action_space, config)
        if self.config.get('_enable_new_api_stack', False):
            model = self.make_rl_module()
            dist_class = None
        else:
            (model, dist_class) = self._init_model_and_dist_class()
        num_gpus = self._get_num_gpus_for_policy()
        gpu_ids = list(range(torch.cuda.device_count()))
        logger.info(f'Found {len(gpu_ids)} visible cuda devices.')
        if config['_fake_gpus'] or num_gpus == 0 or (not gpu_ids):
            self.device = torch.device('cpu')
            self.devices = [self.device for _ in range(int(math.ceil(num_gpus)) or 1)]
            self.model_gpu_towers = [model if i == 0 else copy.deepcopy(model) for i in range(int(math.ceil(num_gpus)) or 1)]
            if hasattr(self, 'target_model'):
                self.target_models = {m: self.target_model for m in self.model_gpu_towers}
            self.model = model
        else:
            if ray._private.worker._mode() == ray._private.worker.WORKER_MODE:
                gpu_ids = ray.get_gpu_ids()
            if len(gpu_ids) < num_gpus:
                raise ValueError(f'TorchPolicy was not able to find enough GPU IDs! Found {gpu_ids}, but num_gpus={num_gpus}.')
            self.devices = [torch.device('cuda:{}'.format(i)) for (i, id_) in enumerate(gpu_ids) if i < num_gpus]
            self.device = self.devices[0]
            ids = [id_ for (i, id_) in enumerate(gpu_ids) if i < num_gpus]
            self.model_gpu_towers = []
            for (i, _) in enumerate(ids):
                model_copy = copy.deepcopy(model)
                self.model_gpu_towers.append(model_copy.to(self.devices[i]))
            if hasattr(self, 'target_model'):
                self.target_models = {m: copy.deepcopy(self.target_model).to(self.devices[i]) for (i, m) in enumerate(self.model_gpu_towers)}
            self.model = self.model_gpu_towers[0]
        self.dist_class = dist_class
        self.unwrapped_model = model
        self._lock = threading.RLock()
        self._state_inputs = self.model.get_initial_state()
        self._is_recurrent = len(tree.flatten(self._state_inputs)) > 0
        if self.config.get('_enable_new_api_stack', False):
            self.view_requirements = self.model.update_default_view_requirements(self.view_requirements)
        else:
            self._update_model_view_requirements_from_init_state()
            self.view_requirements.update(self.model.view_requirements)
        if self.config.get('_enable_new_api_stack', False):
            self.exploration = None
        else:
            self.exploration = self._create_exploration()
        if not self.config.get('_enable_new_api_stack', False):
            self._optimizers = force_list(self.optimizer())
            self._loss = None
            self.multi_gpu_param_groups: List[Set[int]] = []
            main_params = {p: i for (i, p) in enumerate(self.model.parameters())}
            for o in self._optimizers:
                param_indices = []
                for (pg_idx, pg) in enumerate(o.param_groups):
                    for p in pg['params']:
                        param_indices.append(main_params[p])
                self.multi_gpu_param_groups.append(set(param_indices))
            num_buffers = self.config.get('num_multi_gpu_tower_stacks', 1)
            self._loaded_batches = [[] for _ in range(num_buffers)]
        self.distributed_world_size = None
        self.batch_divisibility_req = self.get_batch_divisibility_req()
        self.max_seq_len = max_seq_len
        self.tower_stats = {}
        if not hasattr(self.model, 'tower_stats'):
            for model in self.model_gpu_towers:
                self.tower_stats[model] = {}

    def loss_initialized(self):
        if False:
            while True:
                i = 10
        return self._loss_initialized

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    @override(Policy)
    def loss(self, model: ModelV2, dist_class: Type[TorchDistributionWrapper], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        if False:
            while True:
                i = 10
        'Constructs the loss function.\n\n        Args:\n            model: The Model to calculate the loss for.\n            dist_class: The action distr. class.\n            train_batch: The training data.\n\n        Returns:\n            Loss tensor given the input batch.\n        '
        if self.config._enable_new_api_stack:
            for k in model.input_specs_train():
                train_batch[k]
            return None
        else:
            raise NotImplementedError

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def action_sampler_fn(self, model: ModelV2, *, obs_batch: TensorType, state_batches: TensorType, **kwargs) -> Tuple[TensorType, TensorType, TensorType, List[TensorType]]:
        if False:
            for i in range(10):
                print('nop')
        'Custom function for sampling new actions given policy.\n\n        Args:\n            model: Underlying model.\n            obs_batch: Observation tensor batch.\n            state_batches: Action sampling state batch.\n\n        Returns:\n            Sampled action\n            Log-likelihood\n            Action distribution inputs\n            Updated state\n        '
        return (None, None, None, None)

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def action_distribution_fn(self, model: ModelV2, *, obs_batch: TensorType, state_batches: TensorType, **kwargs) -> Tuple[TensorType, type, List[TensorType]]:
        if False:
            i = 10
            return i + 15
        'Action distribution function for this Policy.\n\n        Args:\n            model: Underlying model.\n            obs_batch: Observation tensor batch.\n            state_batches: Action sampling state batch.\n\n        Returns:\n            Distribution input.\n            ActionDistribution class.\n            State outs.\n        '
        return (None, None, None)

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def make_model(self) -> ModelV2:
        if False:
            while True:
                i = 10
        'Create model.\n\n        Note: only one of make_model or make_model_and_action_dist\n        can be overridden.\n\n        Returns:\n            ModelV2 model.\n        '
        return None

    @ExperimentalAPI
    @override(Policy)
    def maybe_remove_time_dimension(self, input_dict: Dict[str, TensorType]):
        if False:
            return 10
        assert self.config.get('_enable_new_api_stack', False), 'This is a helper method for the new learner API.'
        if self.config.get('_enable_new_api_stack', False) and self.model.is_stateful():
            ret = {}

            def fold_mapping(item):
                if False:
                    i = 10
                    return i + 15
                item = torch.as_tensor(item)
                size = item.size()
                (b_dim, t_dim) = list(size[:2])
                other_dims = list(size[2:])
                return item.reshape([b_dim * t_dim] + other_dims)
            for (k, v) in input_dict.items():
                if k not in (STATE_IN, STATE_OUT):
                    ret[k] = tree.map_structure(fold_mapping, v)
                else:
                    ret[k] = v
            return ret
        else:
            return input_dict

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def make_model_and_action_dist(self) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
        if False:
            return 10
        'Create model and action distribution function.\n\n        Returns:\n            ModelV2 model.\n            ActionDistribution class.\n        '
        return (None, None)

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def get_batch_divisibility_req(self) -> int:
        if False:
            return 10
        'Get batch divisibility request.\n\n        Returns:\n            Size N. A sample batch must be of size K*N.\n        '
        return 1

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        if False:
            return 10
        'Stats function. Returns a dict of statistics.\n\n        Args:\n            train_batch: The SampleBatch (already) used for training.\n\n        Returns:\n            The stats dict.\n        '
        return {}

    @DeveloperAPI
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def extra_grad_process(self, optimizer: 'torch.optim.Optimizer', loss: TensorType) -> Dict[str, TensorType]:
        if False:
            i = 10
            return i + 15
        'Called after each optimizer.zero_grad() + loss.backward() call.\n\n        Called for each self._optimizers/loss-value pair.\n        Allows for gradient processing before optimizer.step() is called.\n        E.g. for gradient clipping.\n\n        Args:\n            optimizer: A torch optimizer object.\n            loss: The loss tensor associated with the optimizer.\n\n        Returns:\n            An dict with information on the gradient processing step.\n        '
        return {}

    @DeveloperAPI
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def extra_compute_grad_fetches(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'Extra values to fetch and return from compute_gradients().\n\n        Returns:\n            Extra fetch dict to be added to the fetch dict of the\n            `compute_gradients` call.\n        '
        return {LEARNER_STATS_KEY: {}}

    @DeveloperAPI
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def extra_action_out(self, input_dict: Dict[str, TensorType], state_batches: List[TensorType], model: TorchModelV2, action_dist: TorchDistributionWrapper) -> Dict[str, TensorType]:
        if False:
            return 10
        'Returns dict of extra info to include in experience batch.\n\n        Args:\n            input_dict: Dict of model input tensors.\n            state_batches: List of state tensors.\n            model: Reference to the model object.\n            action_dist: Torch action dist object\n                to get log-probs (e.g. for already sampled actions).\n\n        Returns:\n            Extra outputs to return in a `compute_actions_from_input_dict()`\n            call (3rd return value).\n        '
        return {}

    @override(Policy)
    @DeveloperAPI
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def postprocess_trajectory(self, sample_batch: SampleBatch, other_agent_batches: Optional[Dict[Any, SampleBatch]]=None, episode: Optional['Episode']=None) -> SampleBatch:
        if False:
            i = 10
            return i + 15
        "Postprocesses a trajectory and returns the processed trajectory.\n\n        The trajectory contains only data from one episode and from one agent.\n        - If  `config.batch_mode=truncate_episodes` (default), sample_batch may\n        contain a truncated (at-the-end) episode, in case the\n        `config.rollout_fragment_length` was reached by the sampler.\n        - If `config.batch_mode=complete_episodes`, sample_batch will contain\n        exactly one episode (no matter how long).\n        New columns can be added to sample_batch and existing ones may be altered.\n\n        Args:\n            sample_batch: The SampleBatch to postprocess.\n            other_agent_batches (Optional[Dict[PolicyID, SampleBatch]]): Optional\n                dict of AgentIDs mapping to other agents' trajectory data (from the\n                same episode). NOTE: The other agents use the same policy.\n            episode (Optional[Episode]): Optional multi-agent episode\n                object in which the agents operated.\n\n        Returns:\n            SampleBatch: The postprocessed, modified SampleBatch (or a new one).\n        "
        return sample_batch

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def optimizer(self) -> Union[List['torch.optim.Optimizer'], 'torch.optim.Optimizer']:
        if False:
            i = 10
            return i + 15
        'Custom the local PyTorch optimizer(s) to use.\n\n        Returns:\n            The local PyTorch optimizer(s) to use for this Policy.\n        '
        if hasattr(self, 'config'):
            optimizers = [torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])]
        else:
            optimizers = [torch.optim.Adam(self.model.parameters())]
        if self.exploration:
            optimizers = self.exploration.get_exploration_optimizer(optimizers)
        return optimizers

    def _init_model_and_dist_class(self):
        if False:
            while True:
                i = 10
        if is_overridden(self.make_model) and is_overridden(self.make_model_and_action_dist):
            raise ValueError('Only one of make_model or make_model_and_action_dist can be overridden.')
        if is_overridden(self.make_model):
            model = self.make_model()
            (dist_class, _) = ModelCatalog.get_action_dist(self.action_space, self.config['model'], framework=self.framework)
        elif is_overridden(self.make_model_and_action_dist):
            (model, dist_class) = self.make_model_and_action_dist()
        else:
            (dist_class, logit_dim) = ModelCatalog.get_action_dist(self.action_space, self.config['model'], framework=self.framework)
            model = ModelCatalog.get_model_v2(obs_space=self.observation_space, action_space=self.action_space, num_outputs=logit_dim, model_config=self.config['model'], framework=self.framework)
        return (model, dist_class)

    @override(Policy)
    def compute_actions_from_input_dict(self, input_dict: Dict[str, TensorType], explore: bool=None, timestep: Optional[int]=None, **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        if False:
            print('Hello World!')
        seq_lens = None
        with torch.no_grad():
            input_dict = self._lazy_tensor_dict(input_dict)
            input_dict.set_training(True)
            if self.config.get('_enable_new_api_stack', False):
                return self._compute_action_helper(input_dict, state_batches=None, seq_lens=None, explore=explore, timestep=timestep)
            else:
                state_batches = [input_dict[k] for k in input_dict.keys() if 'state_in' in k[:8]]
                if state_batches:
                    seq_lens = torch.tensor([1] * len(state_batches[0]), dtype=torch.long, device=state_batches[0].device)
                return self._compute_action_helper(input_dict, state_batches, seq_lens, explore, timestep)

    @override(Policy)
    @DeveloperAPI
    def compute_actions(self, obs_batch: Union[List[TensorStructType], TensorStructType], state_batches: Optional[List[TensorType]]=None, prev_action_batch: Union[List[TensorStructType], TensorStructType]=None, prev_reward_batch: Union[List[TensorStructType], TensorStructType]=None, info_batch: Optional[Dict[str, list]]=None, episodes: Optional[List['Episode']]=None, explore: Optional[bool]=None, timestep: Optional[int]=None, **kwargs) -> Tuple[TensorStructType, List[TensorType], Dict[str, TensorType]]:
        if False:
            return 10
        with torch.no_grad():
            seq_lens = torch.ones(len(obs_batch), dtype=torch.int32)
            input_dict = self._lazy_tensor_dict({SampleBatch.CUR_OBS: obs_batch, 'is_training': False})
            if prev_action_batch is not None:
                input_dict[SampleBatch.PREV_ACTIONS] = np.asarray(prev_action_batch)
            if prev_reward_batch is not None:
                input_dict[SampleBatch.PREV_REWARDS] = np.asarray(prev_reward_batch)
            state_batches = [convert_to_torch_tensor(s, self.device) for s in state_batches or []]
            return self._compute_action_helper(input_dict, state_batches, seq_lens, explore, timestep)

    @with_lock
    @override(Policy)
    @DeveloperAPI
    def compute_log_likelihoods(self, actions: Union[List[TensorStructType], TensorStructType], obs_batch: Union[List[TensorStructType], TensorStructType], state_batches: Optional[List[TensorType]]=None, prev_action_batch: Optional[Union[List[TensorStructType], TensorStructType]]=None, prev_reward_batch: Optional[Union[List[TensorStructType], TensorStructType]]=None, actions_normalized: bool=True, in_training: bool=True) -> TensorType:
        if False:
            return 10
        if is_overridden(self.action_sampler_fn) and (not is_overridden(self.action_distribution_fn)):
            raise ValueError('Cannot compute log-prob/likelihood w/o an `action_distribution_fn` and a provided `action_sampler_fn`!')
        with torch.no_grad():
            input_dict = self._lazy_tensor_dict({SampleBatch.CUR_OBS: obs_batch, SampleBatch.ACTIONS: actions})
            if prev_action_batch is not None:
                input_dict[SampleBatch.PREV_ACTIONS] = prev_action_batch
            if prev_reward_batch is not None:
                input_dict[SampleBatch.PREV_REWARDS] = prev_reward_batch
            seq_lens = torch.ones(len(obs_batch), dtype=torch.int32)
            state_batches = [convert_to_torch_tensor(s, self.device) for s in state_batches or []]
            if self.exploration:
                self.exploration.before_compute_actions(explore=False)
            if is_overridden(self.action_distribution_fn):
                (dist_inputs, dist_class, state_out) = self.action_distribution_fn(self.model, obs_batch=input_dict, state_batches=state_batches, seq_lens=seq_lens, explore=False, is_training=False)
                action_dist = dist_class(dist_inputs, self.model)
            elif self.config.get('_enable_new_api_stack', False):
                if in_training:
                    output = self.model.forward_train(input_dict)
                    action_dist_cls = self.model.get_train_action_dist_cls()
                    if action_dist_cls is None:
                        raise ValueError('The RLModules must provide an appropriate action distribution class for training if is_eval_mode is False.')
                else:
                    output = self.model.forward_exploration(input_dict)
                    action_dist_cls = self.model.get_exploration_action_dist_cls()
                    if action_dist_cls is None:
                        raise ValueError('The RLModules must provide an appropriate action distribution class for exploration if is_eval_mode is True.')
                action_dist_inputs = output.get(SampleBatch.ACTION_DIST_INPUTS, None)
                if action_dist_inputs is None:
                    raise ValueError('The RLModules must provide inputs to create the action distribution. These should be part of the output of the appropriate forward method under the key SampleBatch.ACTION_DIST_INPUTS.')
                action_dist = action_dist_cls.from_logits(action_dist_inputs)
            else:
                dist_class = self.dist_class
                (dist_inputs, _) = self.model(input_dict, state_batches, seq_lens)
                action_dist = dist_class(dist_inputs, self.model)
            actions = input_dict[SampleBatch.ACTIONS]
            if not actions_normalized and self.config['normalize_actions']:
                actions = normalize_action(actions, self.action_space_struct)
            log_likelihoods = action_dist.logp(actions)
            return log_likelihoods

    @with_lock
    @override(Policy)
    @DeveloperAPI
    def learn_on_batch(self, postprocessed_batch: SampleBatch) -> Dict[str, TensorType]:
        if False:
            return 10
        if self.model:
            self.model.train()
        learn_stats = {}
        self.callbacks.on_learn_on_batch(policy=self, train_batch=postprocessed_batch, result=learn_stats)
        (grads, fetches) = self.compute_gradients(postprocessed_batch)
        self.apply_gradients(_directStepOptimizerSingleton)
        self.num_grad_updates += 1
        if self.model and hasattr(self.model, 'metrics'):
            fetches['model'] = self.model.metrics()
        else:
            fetches['model'] = {}
        fetches.update({'custom_metrics': learn_stats, NUM_AGENT_STEPS_TRAINED: postprocessed_batch.count, NUM_GRAD_UPDATES_LIFETIME: self.num_grad_updates, DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: self.num_grad_updates - 1 - (postprocessed_batch.num_grad_updates or 0)})
        return fetches

    @override(Policy)
    @DeveloperAPI
    def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int=0) -> int:
        if False:
            print('Hello World!')
        batch.set_training(True)
        if len(self.devices) == 1 and self.devices[0].type == 'cpu':
            assert buffer_index == 0
            pad_batch_to_sequences_of_same_size(batch=batch, max_seq_len=self.max_seq_len, shuffle=False, batch_divisibility_req=self.batch_divisibility_req, view_requirements=self.view_requirements, _enable_new_api_stack=self.config.get('_enable_new_api_stack', False), padding='last' if self.config.get('_enable_new_api_stack', False) else 'zero')
            self._lazy_tensor_dict(batch)
            self._loaded_batches[0] = [batch]
            return len(batch)
        slices = batch.timeslices(num_slices=len(self.devices))
        for slice in slices:
            pad_batch_to_sequences_of_same_size(batch=slice, max_seq_len=self.max_seq_len, shuffle=False, batch_divisibility_req=self.batch_divisibility_req, view_requirements=self.view_requirements, _enable_new_api_stack=self.config.get('_enable_new_api_stack', False), padding='last' if self.config.get('_enable_new_api_stack', False) else 'zero')
        slices = [slice.to_device(self.devices[i]) for (i, slice) in enumerate(slices)]
        self._loaded_batches[buffer_index] = slices
        return len(slices[0])

    @override(Policy)
    @DeveloperAPI
    def get_num_samples_loaded_into_buffer(self, buffer_index: int=0) -> int:
        if False:
            for i in range(10):
                print('nop')
        if len(self.devices) == 1 and self.devices[0] == '/cpu:0':
            assert buffer_index == 0
        return sum((len(b) for b in self._loaded_batches[buffer_index]))

    @override(Policy)
    @DeveloperAPI
    def learn_on_loaded_batch(self, offset: int=0, buffer_index: int=0):
        if False:
            for i in range(10):
                print('nop')
        if not self._loaded_batches[buffer_index]:
            raise ValueError('Must call Policy.load_batch_into_buffer() before Policy.learn_on_loaded_batch()!')
        device_batch_size = self.config.get('sgd_minibatch_size', self.config['train_batch_size']) // len(self.devices)
        if self.model_gpu_towers:
            for t in self.model_gpu_towers:
                t.train()
        if len(self.devices) == 1 and self.devices[0].type == 'cpu':
            assert buffer_index == 0
            if device_batch_size >= len(self._loaded_batches[0][0]):
                batch = self._loaded_batches[0][0]
            else:
                batch = self._loaded_batches[0][0][offset:offset + device_batch_size]
            return self.learn_on_batch(batch)
        if len(self.devices) > 1:
            state_dict = self.model.state_dict()
            assert self.model_gpu_towers[0] is self.model
            for tower in self.model_gpu_towers[1:]:
                tower.load_state_dict(state_dict)
        if device_batch_size >= sum((len(s) for s in self._loaded_batches[buffer_index])):
            device_batches = self._loaded_batches[buffer_index]
        else:
            device_batches = [b[offset:offset + device_batch_size] for b in self._loaded_batches[buffer_index]]
        batch_fetches = {}
        for (i, batch) in enumerate(device_batches):
            custom_metrics = {}
            self.callbacks.on_learn_on_batch(policy=self, train_batch=batch, result=custom_metrics)
            batch_fetches[f'tower_{i}'] = {'custom_metrics': custom_metrics}
        tower_outputs = self._multi_gpu_parallel_grad_calc(device_batches)
        all_grads = []
        for i in range(len(tower_outputs[0][0])):
            if tower_outputs[0][0][i] is not None:
                all_grads.append(torch.mean(torch.stack([t[0][i].to(self.device) for t in tower_outputs]), dim=0))
            else:
                all_grads.append(None)
        for (i, p) in enumerate(self.model.parameters()):
            p.grad = all_grads[i]
        self.apply_gradients(_directStepOptimizerSingleton)
        self.num_grad_updates += 1
        for (i, (model, batch)) in enumerate(zip(self.model_gpu_towers, device_batches)):
            batch_fetches[f'tower_{i}'].update({LEARNER_STATS_KEY: self.stats_fn(batch), 'model': {} if self.config.get('_enable_new_api_stack', False) else model.metrics(), NUM_GRAD_UPDATES_LIFETIME: self.num_grad_updates, DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: self.num_grad_updates - 1 - (batch.num_grad_updates or 0)})
        batch_fetches.update(self.extra_compute_grad_fetches())
        return batch_fetches

    @with_lock
    @override(Policy)
    @DeveloperAPI
    def compute_gradients(self, postprocessed_batch: SampleBatch) -> ModelGradients:
        if False:
            i = 10
            return i + 15
        assert len(self.devices) == 1
        if not postprocessed_batch.zero_padded:
            pad_batch_to_sequences_of_same_size(batch=postprocessed_batch, max_seq_len=self.max_seq_len, shuffle=False, batch_divisibility_req=self.batch_divisibility_req, view_requirements=self.view_requirements, _enable_new_api_stack=self.config.get('_enable_new_api_stack', False), padding='last' if self.config.get('_enable_new_api_stack', False) else 'zero')
        postprocessed_batch.set_training(True)
        self._lazy_tensor_dict(postprocessed_batch, device=self.devices[0])
        tower_outputs = self._multi_gpu_parallel_grad_calc([postprocessed_batch])
        (all_grads, grad_info) = tower_outputs[0]
        grad_info['allreduce_latency'] /= len(self._optimizers)
        grad_info.update(self.stats_fn(postprocessed_batch))
        fetches = self.extra_compute_grad_fetches()
        return (all_grads, dict(fetches, **{LEARNER_STATS_KEY: grad_info}))

    @override(Policy)
    @DeveloperAPI
    def apply_gradients(self, gradients: ModelGradients) -> None:
        if False:
            print('Hello World!')
        if gradients == _directStepOptimizerSingleton:
            for (i, opt) in enumerate(self._optimizers):
                opt.step()
        else:
            assert len(self._optimizers) == 1
            for (g, p) in zip(gradients, self.model.parameters()):
                if g is not None:
                    if torch.is_tensor(g):
                        p.grad = g.to(self.device)
                    else:
                        p.grad = torch.from_numpy(g).to(self.device)
            self._optimizers[0].step()

    @DeveloperAPI
    def get_tower_stats(self, stats_name: str) -> List[TensorStructType]:
        if False:
            print('Hello World!')
        "Returns list of per-tower stats, copied to this Policy's device.\n\n        Args:\n            stats_name: The name of the stats to average over (this str\n                must exist as a key inside each tower's `tower_stats` dict).\n\n        Returns:\n            The list of stats tensor (structs) of all towers, copied to this\n            Policy's device.\n\n        Raises:\n            AssertionError: If the `stats_name` cannot be found in any one\n            of the tower's `tower_stats` dicts.\n        "
        data = []
        for model in self.model_gpu_towers:
            if self.tower_stats:
                tower_stats = self.tower_stats[model]
            else:
                tower_stats = model.tower_stats
            if stats_name in tower_stats:
                data.append(tree.map_structure(lambda s: s.to(self.device), tower_stats[stats_name]))
        assert len(data) > 0, f'Stats `{stats_name}` not found in any of the towers (you have {len(self.model_gpu_towers)} towers in total)! Make sure you call the loss function on at least one of the towers.'
        return data

    @override(Policy)
    @DeveloperAPI
    def get_weights(self) -> ModelWeights:
        if False:
            i = 10
            return i + 15
        return {k: v.cpu().detach().numpy() for (k, v) in self.model.state_dict().items()}

    @override(Policy)
    @DeveloperAPI
    def set_weights(self, weights: ModelWeights) -> None:
        if False:
            for i in range(10):
                print('nop')
        weights = convert_to_torch_tensor(weights, device=self.device)
        if self.config.get('_enable_new_api_stack', False):
            self.model.set_state(weights)
        else:
            self.model.load_state_dict(weights)

    @override(Policy)
    @DeveloperAPI
    def is_recurrent(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._is_recurrent

    @override(Policy)
    @DeveloperAPI
    def num_state_tensors(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self.model.get_initial_state())

    @override(Policy)
    @DeveloperAPI
    def get_initial_state(self) -> List[TensorType]:
        if False:
            return 10
        if self.config.get('_enable_new_api_stack', False):
            return tree.map_structure(lambda s: convert_to_numpy(s), self.model.get_initial_state())
        return [s.detach().cpu().numpy() for s in self.model.get_initial_state()]

    @override(Policy)
    @DeveloperAPI
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def get_state(self) -> PolicyState:
        if False:
            for i in range(10):
                print('nop')
        state = super().get_state()
        state['_optimizer_variables'] = []
        if not self.config.get('_enable_new_api_stack', False):
            for (i, o) in enumerate(self._optimizers):
                optim_state_dict = convert_to_numpy(o.state_dict())
                state['_optimizer_variables'].append(optim_state_dict)
        if not self.config.get('_enable_new_api_stack', False) and self.exploration:
            state['_exploration_state'] = self.exploration.get_state()
        return state

    @override(Policy)
    @DeveloperAPI
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def set_state(self, state: PolicyState) -> None:
        if False:
            return 10
        optimizer_vars = state.get('_optimizer_variables', None)
        if optimizer_vars:
            assert len(optimizer_vars) == len(self._optimizers)
            for (o, s) in zip(self._optimizers, optimizer_vars):
                optim_state_dict = {'param_groups': s['param_groups']}
                optim_state_dict['state'] = convert_to_torch_tensor(s['state'], device=self.device)
                o.load_state_dict(optim_state_dict)
        if hasattr(self, 'exploration') and '_exploration_state' in state:
            self.exploration.set_state(state=state['_exploration_state'])
        self.global_timestep = state['global_timestep']
        super().set_state(state)

    @override(Policy)
    @DeveloperAPI
    def export_model(self, export_dir: str, onnx: Optional[int]=None) -> None:
        if False:
            print('Hello World!')
        "Exports the Policy's Model to local directory for serving.\n\n        Creates a TorchScript model and saves it.\n\n        Args:\n            export_dir: Local writable directory or filename.\n            onnx: If given, will export model in ONNX format. The\n                value of this parameter set the ONNX OpSet version to use.\n        "
        os.makedirs(export_dir, exist_ok=True)
        enable_rl_module = self.config.get('_enable_new_api_stack', False)
        if enable_rl_module and onnx:
            raise ValueError('ONNX export not supported for RLModule API.')
        if onnx:
            self._lazy_tensor_dict(self._dummy_batch)
            if 'state_in_0' not in self._dummy_batch:
                self._dummy_batch['state_in_0'] = self._dummy_batch[SampleBatch.SEQ_LENS] = np.array([1.0])
            seq_lens = self._dummy_batch[SampleBatch.SEQ_LENS]
            state_ins = []
            i = 0
            while 'state_in_{}'.format(i) in self._dummy_batch:
                state_ins.append(self._dummy_batch['state_in_{}'.format(i)])
                i += 1
            dummy_inputs = {k: self._dummy_batch[k] for k in self._dummy_batch.keys() if k != 'is_training'}
            file_name = os.path.join(export_dir, 'model.onnx')
            torch.onnx.export(self.model, (dummy_inputs, state_ins, seq_lens), file_name, export_params=True, opset_version=onnx, do_constant_folding=True, input_names=list(dummy_inputs.keys()) + ['state_ins', SampleBatch.SEQ_LENS], output_names=['output', 'state_outs'], dynamic_axes={k: {0: 'batch_size'} for k in list(dummy_inputs.keys()) + ['state_ins', SampleBatch.SEQ_LENS]})
        else:
            filename = os.path.join(export_dir, 'model.pt')
            try:
                torch.save(self.model, f=filename)
            except Exception:
                if os.path.exists(filename):
                    os.remove(filename)
                logger.warning(ERR_MSG_TORCH_POLICY_CANNOT_SAVE_MODEL)

    @override(Policy)
    @DeveloperAPI
    def import_model_from_h5(self, import_file: str) -> None:
        if False:
            print('Hello World!')
        'Imports weights into torch model.'
        return self.model.import_from_h5(import_file)

    @with_lock
    def _compute_action_helper(self, input_dict, state_batches, seq_lens, explore, timestep):
        if False:
            print('Hello World!')
        'Shared forward pass logic (w/ and w/o trajectory view API).\n\n        Returns:\n            A tuple consisting of a) actions, b) state_out, c) extra_fetches.\n            The input_dict is modified in-place to include a numpy copy of the computed\n            actions under `SampleBatch.ACTIONS`.\n        '
        explore = explore if explore is not None else self.config['explore']
        timestep = timestep if timestep is not None else self.global_timestep
        if self.model:
            self.model.eval()
        extra_fetches = dist_inputs = logp = None
        if isinstance(self.model, RLModule):
            if self.model.is_stateful():
                if not seq_lens:
                    if not isinstance(input_dict, SampleBatch):
                        input_dict = SampleBatch(input_dict)
                    seq_lens = np.array([1] * len(input_dict))
                input_dict = self.maybe_add_time_dimension(input_dict, seq_lens=seq_lens)
            input_dict = convert_to_torch_tensor(input_dict, device=self.device)
            if SampleBatch.SEQ_LENS in input_dict:
                del input_dict[SampleBatch.SEQ_LENS]
            if explore:
                fwd_out = self.model.forward_exploration(input_dict)
                fwd_out = self.maybe_remove_time_dimension(fwd_out)
                action_dist = None
                if SampleBatch.ACTION_DIST_INPUTS in fwd_out:
                    dist_inputs = fwd_out[SampleBatch.ACTION_DIST_INPUTS]
                    action_dist_class = self.model.get_exploration_action_dist_cls()
                    action_dist = action_dist_class.from_logits(dist_inputs)
                if SampleBatch.ACTIONS in fwd_out:
                    actions = fwd_out[SampleBatch.ACTIONS]
                else:
                    if action_dist is None:
                        raise KeyError(f"Your RLModule's `forward_exploration()` method must return a dict with either the {SampleBatch.ACTIONS} key or the {SampleBatch.ACTION_DIST_INPUTS} key in it (or both)!")
                    actions = action_dist.sample()
                if action_dist is not None:
                    logp = action_dist.logp(actions)
            else:
                fwd_out = self.model.forward_inference(input_dict)
                fwd_out = self.maybe_remove_time_dimension(fwd_out)
                action_dist = None
                if SampleBatch.ACTION_DIST_INPUTS in fwd_out:
                    dist_inputs = fwd_out[SampleBatch.ACTION_DIST_INPUTS]
                    action_dist_class = self.model.get_inference_action_dist_cls()
                    action_dist = action_dist_class.from_logits(dist_inputs)
                    action_dist = action_dist.to_deterministic()
                if SampleBatch.ACTIONS in fwd_out:
                    actions = fwd_out[SampleBatch.ACTIONS]
                else:
                    if action_dist is None:
                        raise KeyError(f"Your RLModule's `forward_inference()` method must return a dict with either the {SampleBatch.ACTIONS} key or the {SampleBatch.ACTION_DIST_INPUTS} key in it (or both)!")
                    actions = action_dist.sample()
            state_out = fwd_out.pop(STATE_OUT, {})
            extra_fetches = fwd_out
        elif is_overridden(self.action_sampler_fn):
            action_dist = None
            (actions, logp, dist_inputs, state_out) = self.action_sampler_fn(self.model, obs_batch=input_dict, state_batches=state_batches, explore=explore, timestep=timestep)
        else:
            self.exploration.before_compute_actions(explore=explore, timestep=timestep)
            if is_overridden(self.action_distribution_fn):
                (dist_inputs, dist_class, state_out) = self.action_distribution_fn(self.model, obs_batch=input_dict, state_batches=state_batches, seq_lens=seq_lens, explore=explore, timestep=timestep, is_training=False)
            else:
                dist_class = self.dist_class
                (dist_inputs, state_out) = self.model(input_dict, state_batches, seq_lens)
            if not (isinstance(dist_class, functools.partial) or issubclass(dist_class, TorchDistributionWrapper)):
                raise ValueError('`dist_class` ({}) not a TorchDistributionWrapper subclass! Make sure your `action_distribution_fn` or `make_model_and_action_dist` return a correct distribution class.'.format(dist_class.__name__))
            action_dist = dist_class(dist_inputs, self.model)
            (actions, logp) = self.exploration.get_exploration_action(action_distribution=action_dist, timestep=timestep, explore=explore)
        if extra_fetches is None:
            extra_fetches = self.extra_action_out(input_dict, state_batches, self.model, action_dist)
        if dist_inputs is not None:
            extra_fetches[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs
        if logp is not None:
            extra_fetches[SampleBatch.ACTION_PROB] = torch.exp(logp.float())
            extra_fetches[SampleBatch.ACTION_LOGP] = logp
        self.global_timestep += len(input_dict[SampleBatch.CUR_OBS])
        return convert_to_numpy((actions, state_out, extra_fetches))

    def _lazy_tensor_dict(self, postprocessed_batch: SampleBatch, device=None):
        if False:
            i = 10
            return i + 15
        if not isinstance(postprocessed_batch, SampleBatch):
            postprocessed_batch = SampleBatch(postprocessed_batch)
        postprocessed_batch.set_get_interceptor(functools.partial(convert_to_torch_tensor, device=device or self.device))
        return postprocessed_batch

    def _multi_gpu_parallel_grad_calc(self, sample_batches: List[SampleBatch]) -> List[Tuple[List[TensorType], GradInfoDict]]:
        if False:
            while True:
                i = 10
        "Performs a parallelized loss and gradient calculation over the batch.\n\n        Splits up the given train batch into n shards (n=number of this\n        Policy's devices) and passes each data shard (in parallel) through\n        the loss function using the individual devices' models\n        (self.model_gpu_towers). Then returns each tower's outputs.\n\n        Args:\n            sample_batches: A list of SampleBatch shards to\n                calculate loss and gradients for.\n\n        Returns:\n            A list (one item per device) of 2-tuples, each with 1) gradient\n            list and 2) grad info dict.\n        "
        assert len(self.model_gpu_towers) == len(sample_batches)
        lock = threading.Lock()
        results = {}
        grad_enabled = torch.is_grad_enabled()

        def _worker(shard_idx, model, sample_batch, device):
            if False:
                print('Hello World!')
            torch.set_grad_enabled(grad_enabled)
            try:
                with NullContextManager() if device.type == 'cpu' else torch.cuda.device(device):
                    loss_out = force_list(self.loss(model, self.dist_class, sample_batch))
                    if hasattr(model, 'custom_loss'):
                        loss_out = model.custom_loss(loss_out, sample_batch)
                    assert len(loss_out) == len(self._optimizers)
                    grad_info = {'allreduce_latency': 0.0}
                    parameters = list(model.parameters())
                    all_grads = [None for _ in range(len(parameters))]
                    for (opt_idx, opt) in enumerate(self._optimizers):
                        param_indices = self.multi_gpu_param_groups[opt_idx]
                        for (param_idx, param) in enumerate(parameters):
                            if param_idx in param_indices and param.grad is not None:
                                param.grad.data.zero_()
                        loss_out[opt_idx].backward(retain_graph=True)
                        grad_info.update(self.extra_grad_process(opt, loss_out[opt_idx]))
                        grads = []
                        for (param_idx, param) in enumerate(parameters):
                            if param_idx in param_indices:
                                if param.grad is not None:
                                    grads.append(param.grad)
                                all_grads[param_idx] = param.grad
                        if self.distributed_world_size:
                            start = time.time()
                            if torch.cuda.is_available():
                                for g in grads:
                                    torch.distributed.all_reduce(g, op=torch.distributed.ReduceOp.SUM)
                            else:
                                torch.distributed.all_reduce_coalesced(grads, op=torch.distributed.ReduceOp.SUM)
                            for param_group in opt.param_groups:
                                for p in param_group['params']:
                                    if p.grad is not None:
                                        p.grad /= self.distributed_world_size
                            grad_info['allreduce_latency'] += time.time() - start
                with lock:
                    results[shard_idx] = (all_grads, grad_info)
            except Exception as e:
                import traceback
                with lock:
                    results[shard_idx] = (ValueError(e.args[0] + '\n traceback' + traceback.format_exc() + '\n' + 'In tower {} on device {}'.format(shard_idx, device)), e)
        if len(self.devices) == 1 or self.config['_fake_gpus']:
            for (shard_idx, (model, sample_batch, device)) in enumerate(zip(self.model_gpu_towers, sample_batches, self.devices)):
                _worker(shard_idx, model, sample_batch, device)
                last_result = results[len(results) - 1]
                if isinstance(last_result[0], ValueError):
                    raise last_result[0] from last_result[1]
        else:
            threads = [threading.Thread(target=_worker, args=(shard_idx, model, sample_batch, device)) for (shard_idx, (model, sample_batch, device)) in enumerate(zip(self.model_gpu_towers, sample_batches, self.devices))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        outputs = []
        for shard_idx in range(len(sample_batches)):
            output = results[shard_idx]
            if isinstance(output[0], Exception):
                raise output[0] from output[1]
            outputs.append(results[shard_idx])
        return outputs