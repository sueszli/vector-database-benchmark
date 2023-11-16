import copy
import functools
import logging
import math
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import gymnasium as gym
import numpy as np
import tree
import ray
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import NullContextManager, force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.error import ERR_MSG_TORCH_POLICY_CANNOT_SAVE_MODEL
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY, NUM_AGENT_STEPS_TRAINED, NUM_GRAD_UPDATES_LIFETIME
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import AlgorithmConfigDict, GradInfoDict, ModelGradients, ModelWeights, TensorStructType, TensorType
if TYPE_CHECKING:
    from ray.rllib.evaluation import Episode
(torch, nn) = try_import_torch()
logger = logging.getLogger(__name__)

@DeveloperAPI
class TorchPolicy(Policy):
    """PyTorch specific Policy class to use with RLlib."""

    @DeveloperAPI
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict, *, model: Optional[TorchModelV2]=None, loss: Optional[Callable[[Policy, ModelV2, Type[TorchDistributionWrapper], SampleBatch], Union[TensorType, List[TensorType]]]]=None, action_distribution_class: Optional[Type[TorchDistributionWrapper]]=None, action_sampler_fn: Optional[Callable[[TensorType, List[TensorType]], Union[Tuple[TensorType, TensorType, List[TensorType]], Tuple[TensorType, TensorType, TensorType, List[TensorType]]]]]=None, action_distribution_fn: Optional[Callable[[Policy, ModelV2, TensorType, TensorType, TensorType], Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]]]=None, max_seq_len: int=20, get_batch_divisibility_req: Optional[Callable[[Policy], int]]=None):
        if False:
            while True:
                i = 10
        "Initializes a TorchPolicy instance.\n\n        Args:\n            observation_space: Observation space of the policy.\n            action_space: Action space of the policy.\n            config: The Policy's config dict.\n            model: PyTorch policy module. Given observations as\n                input, this module must return a list of outputs where the\n                first item is action logits, and the rest can be any value.\n            loss: Callable that returns one or more (a list of) scalar loss\n                terms.\n            action_distribution_class: Class for a torch action distribution.\n            action_sampler_fn: A callable returning either a sampled action,\n                its log-likelihood and updated state or a sampled action, its\n                log-likelihood, updated state and action distribution inputs\n                given Policy, ModelV2, input_dict, state batches (optional),\n                explore, and timestep. Provide `action_sampler_fn` if you would\n                like to have full control over the action computation step,\n                including the model forward pass, possible sampling from a\n                distribution, and exploration logic.\n                Note: If `action_sampler_fn` is given, `action_distribution_fn`\n                must be None. If both `action_sampler_fn` and\n                `action_distribution_fn` are None, RLlib will simply pass\n                inputs through `self.model` to get distribution inputs, create\n                the distribution object, sample from it, and apply some\n                exploration logic to the results.\n                The callable takes as inputs: Policy, ModelV2, input_dict\n                (SampleBatch), state_batches (optional), explore, and timestep.\n            action_distribution_fn: A callable returning distribution inputs\n                (parameters), a dist-class to generate an action distribution\n                object from, and internal-state outputs (or an empty list if\n                not applicable).\n                Provide `action_distribution_fn` if you would like to only\n                customize the model forward pass call. The resulting\n                distribution parameters are then used by RLlib to create a\n                distribution object, sample from it, and execute any\n                exploration logic.\n                Note: If `action_distribution_fn` is given, `action_sampler_fn`\n                must be None. If both `action_sampler_fn` and\n                `action_distribution_fn` are None, RLlib will simply pass\n                inputs through `self.model` to get distribution inputs, create\n                the distribution object, sample from it, and apply some\n                exploration logic to the results.\n                The callable takes as inputs: Policy, ModelV2, ModelInputDict,\n                explore, timestep, is_training.\n            max_seq_len: Max sequence length for LSTM training.\n            get_batch_divisibility_req: Optional callable that returns the\n                divisibility requirement for sample batches given the Policy.\n        "
        self.framework = config['framework'] = 'torch'
        self._loss_initialized = False
        super().__init__(observation_space, action_space, config)
        if model is None:
            (dist_class, logit_dim) = ModelCatalog.get_action_dist(action_space, self.config['model'], framework=self.framework)
            model = ModelCatalog.get_model_v2(obs_space=self.observation_space, action_space=self.action_space, num_outputs=logit_dim, model_config=self.config['model'], framework=self.framework)
            if action_distribution_class is None:
                action_distribution_class = dist_class
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
        self._lock = threading.RLock()
        self._state_inputs = self.model.get_initial_state()
        self._is_recurrent = len(self._state_inputs) > 0
        self._update_model_view_requirements_from_init_state()
        self.view_requirements.update(self.model.view_requirements)
        self.exploration = self._create_exploration()
        self.unwrapped_model = model
        if loss is not None:
            self._loss = loss
        elif self.loss.__func__.__qualname__ != 'Policy.loss':
            self._loss = self.loss.__func__
        else:
            self._loss = None
        self._optimizers = force_list(self.optimizer())
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
        self.dist_class = action_distribution_class
        self.action_sampler_fn = action_sampler_fn
        self.action_distribution_fn = action_distribution_fn
        self.distributed_world_size = None
        self.max_seq_len = max_seq_len
        self.batch_divisibility_req = get_batch_divisibility_req(self) if callable(get_batch_divisibility_req) else get_batch_divisibility_req or 1

    @override(Policy)
    def compute_actions_from_input_dict(self, input_dict: Dict[str, TensorType], explore: bool=None, timestep: Optional[int]=None, **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        if False:
            while True:
                i = 10
        with torch.no_grad():
            input_dict = self._lazy_tensor_dict(input_dict)
            input_dict.set_training(True)
            state_batches = [input_dict[k] for k in input_dict.keys() if 'state_in' in k[:8]]
            seq_lens = torch.tensor([1] * len(state_batches[0]), dtype=torch.long, device=state_batches[0].device) if state_batches else None
            return self._compute_action_helper(input_dict, state_batches, seq_lens, explore, timestep)

    @override(Policy)
    @DeveloperAPI
    def compute_actions(self, obs_batch: Union[List[TensorStructType], TensorStructType], state_batches: Optional[List[TensorType]]=None, prev_action_batch: Union[List[TensorStructType], TensorStructType]=None, prev_reward_batch: Union[List[TensorStructType], TensorStructType]=None, info_batch: Optional[Dict[str, list]]=None, episodes: Optional[List['Episode']]=None, explore: Optional[bool]=None, timestep: Optional[int]=None, **kwargs) -> Tuple[TensorStructType, List[TensorType], Dict[str, TensorType]]:
        if False:
            print('Hello World!')
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
    def compute_log_likelihoods(self, actions: Union[List[TensorStructType], TensorStructType], obs_batch: Union[List[TensorStructType], TensorStructType], state_batches: Optional[List[TensorType]]=None, prev_action_batch: Optional[Union[List[TensorStructType], TensorStructType]]=None, prev_reward_batch: Optional[Union[List[TensorStructType], TensorStructType]]=None, actions_normalized: bool=True, **kwargs) -> TensorType:
        if False:
            return 10
        if self.action_sampler_fn and self.action_distribution_fn is None:
            raise ValueError('Cannot compute log-prob/likelihood w/o an `action_distribution_fn` and a provided `action_sampler_fn`!')
        with torch.no_grad():
            input_dict = self._lazy_tensor_dict({SampleBatch.CUR_OBS: obs_batch, SampleBatch.ACTIONS: actions})
            if prev_action_batch is not None:
                input_dict[SampleBatch.PREV_ACTIONS] = prev_action_batch
            if prev_reward_batch is not None:
                input_dict[SampleBatch.PREV_REWARDS] = prev_reward_batch
            seq_lens = torch.ones(len(obs_batch), dtype=torch.int32)
            state_batches = [convert_to_torch_tensor(s, self.device) for s in state_batches or []]
            self.exploration.before_compute_actions(explore=False)
            if self.action_distribution_fn:
                try:
                    (dist_inputs, dist_class, state_out) = self.action_distribution_fn(self, self.model, input_dict=input_dict, state_batches=state_batches, seq_lens=seq_lens, explore=False, is_training=False)
                except TypeError as e:
                    if 'positional argument' in e.args[0] or 'unexpected keyword argument' in e.args[0]:
                        (dist_inputs, dist_class, _) = self.action_distribution_fn(policy=self, model=self.model, obs_batch=input_dict[SampleBatch.CUR_OBS], explore=False, is_training=False)
                    else:
                        raise e
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
            i = 10
            return i + 15
        if self.model:
            self.model.train()
        learn_stats = {}
        self.callbacks.on_learn_on_batch(policy=self, train_batch=postprocessed_batch, result=learn_stats)
        (grads, fetches) = self.compute_gradients(postprocessed_batch)
        self.apply_gradients(_directStepOptimizerSingleton)
        self.num_grad_updates += 1
        if self.model:
            fetches['model'] = self.model.metrics()
        fetches.update({'custom_metrics': learn_stats, NUM_AGENT_STEPS_TRAINED: postprocessed_batch.count, NUM_GRAD_UPDATES_LIFETIME: self.num_grad_updates, DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: self.num_grad_updates - 1 - (postprocessed_batch.num_grad_updates or 0)})
        return fetches

    @override(Policy)
    @DeveloperAPI
    def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int=0) -> int:
        if False:
            for i in range(10):
                print('nop')
        batch.set_training(True)
        if len(self.devices) == 1 and self.devices[0].type == 'cpu':
            assert buffer_index == 0
            pad_batch_to_sequences_of_same_size(batch=batch, max_seq_len=self.max_seq_len, shuffle=False, batch_divisibility_req=self.batch_divisibility_req, view_requirements=self.view_requirements)
            self._lazy_tensor_dict(batch)
            self._loaded_batches[0] = [batch]
            return len(batch)
        slices = batch.timeslices(num_slices=len(self.devices))
        for slice in slices:
            pad_batch_to_sequences_of_same_size(batch=slice, max_seq_len=self.max_seq_len, shuffle=False, batch_divisibility_req=self.batch_divisibility_req, view_requirements=self.view_requirements)
        slices = [slice.to_device(self.devices[i]) for (i, slice) in enumerate(slices)]
        self._loaded_batches[buffer_index] = slices
        return len(slices[0])

    @override(Policy)
    @DeveloperAPI
    def get_num_samples_loaded_into_buffer(self, buffer_index: int=0) -> int:
        if False:
            print('Hello World!')
        if len(self.devices) == 1 and self.devices[0] == '/cpu:0':
            assert buffer_index == 0
        return sum((len(b) for b in self._loaded_batches[buffer_index]))

    @override(Policy)
    @DeveloperAPI
    def learn_on_loaded_batch(self, offset: int=0, buffer_index: int=0):
        if False:
            return 10
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
            batch_fetches[f'tower_{i}'].update({LEARNER_STATS_KEY: self.extra_grad_info(batch), 'model': model.metrics(), NUM_GRAD_UPDATES_LIFETIME: self.num_grad_updates, DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: self.num_grad_updates - 1 - (batch.num_grad_updates or 0)})
        batch_fetches.update(self.extra_compute_grad_fetches())
        return batch_fetches

    @with_lock
    @override(Policy)
    @DeveloperAPI
    def compute_gradients(self, postprocessed_batch: SampleBatch) -> ModelGradients:
        if False:
            print('Hello World!')
        assert len(self.devices) == 1
        if not postprocessed_batch.zero_padded:
            pad_batch_to_sequences_of_same_size(batch=postprocessed_batch, max_seq_len=self.max_seq_len, shuffle=False, batch_divisibility_req=self.batch_divisibility_req, view_requirements=self.view_requirements)
        postprocessed_batch.set_training(True)
        self._lazy_tensor_dict(postprocessed_batch, device=self.devices[0])
        tower_outputs = self._multi_gpu_parallel_grad_calc([postprocessed_batch])
        (all_grads, grad_info) = tower_outputs[0]
        grad_info['allreduce_latency'] /= len(self._optimizers)
        grad_info.update(self.extra_grad_info(postprocessed_batch))
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
            for i in range(10):
                print('nop')
        "Returns list of per-tower stats, copied to this Policy's device.\n\n        Args:\n            stats_name: The name of the stats to average over (this str\n                must exist as a key inside each tower's `tower_stats` dict).\n\n        Returns:\n            The list of stats tensor (structs) of all towers, copied to this\n            Policy's device.\n\n        Raises:\n            AssertionError: If the `stats_name` cannot be found in any one\n            of the tower's `tower_stats` dicts.\n        "
        data = []
        for tower in self.model_gpu_towers:
            if stats_name in tower.tower_stats:
                data.append(tree.map_structure(lambda s: s.to(self.device), tower.tower_stats[stats_name]))
        assert len(data) > 0, f'Stats `{stats_name}` not found in any of the towers (you have {len(self.model_gpu_towers)} towers in total)! Make sure you call the loss function on at least one of the towers.'
        return data

    @override(Policy)
    @DeveloperAPI
    def get_weights(self) -> ModelWeights:
        if False:
            while True:
                i = 10
        return {k: v.cpu().detach().numpy() for (k, v) in self.model.state_dict().items()}

    @override(Policy)
    @DeveloperAPI
    def set_weights(self, weights: ModelWeights) -> None:
        if False:
            while True:
                i = 10
        weights = convert_to_torch_tensor(weights, device=self.device)
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
            for i in range(10):
                print('nop')
        return len(self.model.get_initial_state())

    @override(Policy)
    @DeveloperAPI
    def get_initial_state(self) -> List[TensorType]:
        if False:
            return 10
        return [s.detach().cpu().numpy() for s in self.model.get_initial_state()]

    @override(Policy)
    @DeveloperAPI
    def get_state(self) -> PolicyState:
        if False:
            for i in range(10):
                print('nop')
        state = super().get_state()
        state['_optimizer_variables'] = []
        for (i, o) in enumerate(self._optimizers):
            optim_state_dict = convert_to_numpy(o.state_dict())
            state['_optimizer_variables'].append(optim_state_dict)
        if not self.config.get('_enable_new_api_stack', False) and self.exploration:
            state['_exploration_state'] = self.exploration.get_state()
        return state

    @override(Policy)
    @DeveloperAPI
    def set_state(self, state: PolicyState) -> None:
        if False:
            while True:
                i = 10
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

    @DeveloperAPI
    def extra_grad_process(self, optimizer: 'torch.optim.Optimizer', loss: TensorType) -> Dict[str, TensorType]:
        if False:
            while True:
                i = 10
        'Called after each optimizer.zero_grad() + loss.backward() call.\n\n        Called for each self._optimizers/loss-value pair.\n        Allows for gradient processing before optimizer.step() is called.\n        E.g. for gradient clipping.\n\n        Args:\n            optimizer: A torch optimizer object.\n            loss: The loss tensor associated with the optimizer.\n\n        Returns:\n            An dict with information on the gradient processing step.\n        '
        return {}

    @DeveloperAPI
    def extra_compute_grad_fetches(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Extra values to fetch and return from compute_gradients().\n\n        Returns:\n            Extra fetch dict to be added to the fetch dict of the\n            `compute_gradients` call.\n        '
        return {LEARNER_STATS_KEY: {}}

    @DeveloperAPI
    def extra_action_out(self, input_dict: Dict[str, TensorType], state_batches: List[TensorType], model: TorchModelV2, action_dist: TorchDistributionWrapper) -> Dict[str, TensorType]:
        if False:
            print('Hello World!')
        'Returns dict of extra info to include in experience batch.\n\n        Args:\n            input_dict: Dict of model input tensors.\n            state_batches: List of state tensors.\n            model: Reference to the model object.\n            action_dist: Torch action dist object\n                to get log-probs (e.g. for already sampled actions).\n\n        Returns:\n            Extra outputs to return in a `compute_actions_from_input_dict()`\n            call (3rd return value).\n        '
        return {}

    @DeveloperAPI
    def extra_grad_info(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        if False:
            return 10
        'Return dict of extra grad info.\n\n        Args:\n            train_batch: The training batch for which to produce\n                extra grad info for.\n\n        Returns:\n            The info dict carrying grad info per str key.\n        '
        return {}

    @DeveloperAPI
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

    @override(Policy)
    @DeveloperAPI
    def export_model(self, export_dir: str, onnx: Optional[int]=None) -> None:
        if False:
            while True:
                i = 10
        "Exports the Policy's Model to local directory for serving.\n\n        Creates a TorchScript model and saves it.\n\n        Args:\n            export_dir: Local writable directory or filename.\n            onnx: If given, will export model in ONNX format. The\n                value of this parameter set the ONNX OpSet version to use.\n        "
        os.makedirs(export_dir, exist_ok=True)
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
            return 10
        'Imports weights into torch model.'
        return self.model.import_from_h5(import_file)

    @with_lock
    def _compute_action_helper(self, input_dict, state_batches, seq_lens, explore, timestep):
        if False:
            while True:
                i = 10
        'Shared forward pass logic (w/ and w/o trajectory view API).\n\n        Returns:\n            A tuple consisting of a) actions, b) state_out, c) extra_fetches.\n        '
        explore = explore if explore is not None else self.config['explore']
        timestep = timestep if timestep is not None else self.global_timestep
        self._is_recurrent = state_batches is not None and state_batches != []
        if self.model:
            self.model.eval()
        if self.action_sampler_fn:
            action_dist = dist_inputs = None
            action_sampler_outputs = self.action_sampler_fn(self, self.model, input_dict, state_batches, explore=explore, timestep=timestep)
            if len(action_sampler_outputs) == 4:
                (actions, logp, dist_inputs, state_out) = action_sampler_outputs
            else:
                (actions, logp, state_out) = action_sampler_outputs
        else:
            self.exploration.before_compute_actions(explore=explore, timestep=timestep)
            if self.action_distribution_fn:
                try:
                    (dist_inputs, dist_class, state_out) = self.action_distribution_fn(self, self.model, input_dict=input_dict, state_batches=state_batches, seq_lens=seq_lens, explore=explore, timestep=timestep, is_training=False)
                except TypeError as e:
                    if 'positional argument' in e.args[0] or 'unexpected keyword argument' in e.args[0]:
                        (dist_inputs, dist_class, state_out) = self.action_distribution_fn(self, self.model, input_dict[SampleBatch.CUR_OBS], explore=explore, timestep=timestep, is_training=False)
                    else:
                        raise e
            else:
                dist_class = self.dist_class
                (dist_inputs, state_out) = self.model(input_dict, state_batches, seq_lens)
            if not (isinstance(dist_class, functools.partial) or issubclass(dist_class, TorchDistributionWrapper)):
                raise ValueError('`dist_class` ({}) not a TorchDistributionWrapper subclass! Make sure your `action_distribution_fn` or `make_model_and_action_dist` return a correct distribution class.'.format(dist_class.__name__))
            action_dist = dist_class(dist_inputs, self.model)
            (actions, logp) = self.exploration.get_exploration_action(action_distribution=action_dist, timestep=timestep, explore=explore)
        input_dict[SampleBatch.ACTIONS] = actions
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
            for i in range(10):
                print('nop')
        if not isinstance(postprocessed_batch, SampleBatch):
            postprocessed_batch = SampleBatch(postprocessed_batch)
        postprocessed_batch.set_get_interceptor(functools.partial(convert_to_torch_tensor, device=device or self.device))
        return postprocessed_batch

    def _multi_gpu_parallel_grad_calc(self, sample_batches: List[SampleBatch]) -> List[Tuple[List[TensorType], GradInfoDict]]:
        if False:
            print('Hello World!')
        "Performs a parallelized loss and gradient calculation over the batch.\n\n        Splits up the given train batch into n shards (n=number of this\n        Policy's devices) and passes each data shard (in parallel) through\n        the loss function using the individual devices' models\n        (self.model_gpu_towers). Then returns each tower's outputs.\n\n        Args:\n            sample_batches: A list of SampleBatch shards to\n                calculate loss and gradients for.\n\n        Returns:\n            A list (one item per device) of 2-tuples, each with 1) gradient\n            list and 2) grad info dict.\n        "
        assert len(self.model_gpu_towers) == len(sample_batches)
        lock = threading.Lock()
        results = {}
        grad_enabled = torch.is_grad_enabled()

        def _worker(shard_idx, model, sample_batch, device):
            if False:
                for i in range(10):
                    print('nop')
            torch.set_grad_enabled(grad_enabled)
            try:
                with NullContextManager() if device.type == 'cpu' else torch.cuda.device(device):
                    loss_out = force_list(self._loss(self, model, self.dist_class, sample_batch))
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
                    results[shard_idx] = (ValueError(f'Error In tower {shard_idx} on device {device} during multi GPU parallel gradient calculation:: {e}\nTraceback: \n{traceback.format_exc()}\n'), e)
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

@DeveloperAPI
class DirectStepOptimizer:
    """Typesafe method for indicating `apply_gradients` can directly step the
    optimizers with in-place gradients.
    """
    _instance = None

    def __new__(cls):
        if False:
            for i in range(10):
                print('nop')
        if DirectStepOptimizer._instance is None:
            DirectStepOptimizer._instance = super().__new__(cls)
        return DirectStepOptimizer._instance

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return type(self) is type(other)

    def __repr__(self):
        if False:
            return 10
        return 'DirectStepOptimizer'
_directStepOptimizerSingleton = DirectStepOptimizer()