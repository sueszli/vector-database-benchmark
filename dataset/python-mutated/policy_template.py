from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, Union
import gymnasium as gym
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.jax.jax_modelv2 import JAXModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils import add_mixins, NullContextManager
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.framework import try_import_torch, try_import_jax
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import ModelGradients, TensorType, AlgorithmConfigDict
if TYPE_CHECKING:
    from ray.rllib.evaluation.episode import Episode
(jax, _) = try_import_jax()
(torch, _) = try_import_torch()

@DeveloperAPI
def build_policy_class(name: str, framework: str, *, loss_fn: Optional[Callable[[Policy, ModelV2, Type[TorchDistributionWrapper], SampleBatch], Union[TensorType, List[TensorType]]]], get_default_config: Optional[Callable[[], AlgorithmConfigDict]]=None, stats_fn: Optional[Callable[[Policy, SampleBatch], Dict[str, TensorType]]]=None, postprocess_fn: Optional[Callable[[Policy, SampleBatch, Optional[Dict[Any, SampleBatch]], Optional['Episode']], SampleBatch]]=None, extra_action_out_fn: Optional[Callable[[Policy, Dict[str, TensorType], List[TensorType], ModelV2, TorchDistributionWrapper], Dict[str, TensorType]]]=None, extra_grad_process_fn: Optional[Callable[[Policy, 'torch.optim.Optimizer', TensorType], Dict[str, TensorType]]]=None, extra_learn_fetches_fn: Optional[Callable[[Policy], Dict[str, TensorType]]]=None, optimizer_fn: Optional[Callable[[Policy, AlgorithmConfigDict], 'torch.optim.Optimizer']]=None, validate_spaces: Optional[Callable[[Policy, gym.Space, gym.Space, AlgorithmConfigDict], None]]=None, before_init: Optional[Callable[[Policy, gym.Space, gym.Space, AlgorithmConfigDict], None]]=None, before_loss_init: Optional[Callable[[Policy, gym.spaces.Space, gym.spaces.Space, AlgorithmConfigDict], None]]=None, after_init: Optional[Callable[[Policy, gym.Space, gym.Space, AlgorithmConfigDict], None]]=None, _after_loss_init: Optional[Callable[[Policy, gym.spaces.Space, gym.spaces.Space, AlgorithmConfigDict], None]]=None, action_sampler_fn: Optional[Callable[[TensorType, List[TensorType]], Tuple[TensorType, TensorType]]]=None, action_distribution_fn: Optional[Callable[[Policy, ModelV2, TensorType, TensorType, TensorType], Tuple[TensorType, type, List[TensorType]]]]=None, make_model: Optional[Callable[[Policy, gym.spaces.Space, gym.spaces.Space, AlgorithmConfigDict], ModelV2]]=None, make_model_and_action_dist: Optional[Callable[[Policy, gym.spaces.Space, gym.spaces.Space, AlgorithmConfigDict], Tuple[ModelV2, Type[TorchDistributionWrapper]]]]=None, compute_gradients_fn: Optional[Callable[[Policy, SampleBatch], Tuple[ModelGradients, dict]]]=None, apply_gradients_fn: Optional[Callable[[Policy, 'torch.optim.Optimizer'], None]]=None, mixins: Optional[List[type]]=None, get_batch_divisibility_req: Optional[Callable[[Policy], int]]=None) -> Type[TorchPolicy]:
    if False:
        print('Hello World!')
    'Helper function for creating a new Policy class at runtime.\n\n    Supports frameworks JAX and PyTorch.\n\n    Args:\n        name: name of the policy (e.g., "PPOTorchPolicy")\n        framework: Either "jax" or "torch".\n        loss_fn (Optional[Callable[[Policy, ModelV2,\n            Type[TorchDistributionWrapper], SampleBatch], Union[TensorType,\n            List[TensorType]]]]): Callable that returns a loss tensor.\n        get_default_config (Optional[Callable[[None], AlgorithmConfigDict]]):\n            Optional callable that returns the default config to merge with any\n            overrides. If None, uses only(!) the user-provided\n            PartialAlgorithmConfigDict as dict for this Policy.\n        postprocess_fn (Optional[Callable[[Policy, SampleBatch,\n            Optional[Dict[Any, SampleBatch]], Optional["Episode"]],\n            SampleBatch]]): Optional callable for post-processing experience\n            batches (called after the super\'s `postprocess_trajectory` method).\n        stats_fn (Optional[Callable[[Policy, SampleBatch],\n            Dict[str, TensorType]]]): Optional callable that returns a dict of\n            values given the policy and training batch. If None,\n            will use `TorchPolicy.extra_grad_info()` instead. The stats dict is\n            used for logging (e.g. in TensorBoard).\n        extra_action_out_fn (Optional[Callable[[Policy, Dict[str, TensorType],\n            List[TensorType], ModelV2, TorchDistributionWrapper]], Dict[str,\n            TensorType]]]): Optional callable that returns a dict of extra\n            values to include in experiences. If None, no extra computations\n            will be performed.\n        extra_grad_process_fn (Optional[Callable[[Policy,\n            "torch.optim.Optimizer", TensorType], Dict[str, TensorType]]]):\n            Optional callable that is called after gradients are computed and\n            returns a processing info dict. If None, will call the\n            `TorchPolicy.extra_grad_process()` method instead.\n        # TODO: (sven) dissolve naming mismatch between "learn" and "compute.."\n        extra_learn_fetches_fn (Optional[Callable[[Policy],\n            Dict[str, TensorType]]]): Optional callable that returns a dict of\n            extra tensors from the policy after loss evaluation. If None,\n            will call the `TorchPolicy.extra_compute_grad_fetches()` method\n            instead.\n        optimizer_fn (Optional[Callable[[Policy, AlgorithmConfigDict],\n            "torch.optim.Optimizer"]]): Optional callable that returns a\n            torch optimizer given the policy and config. If None, will call\n            the `TorchPolicy.optimizer()` method instead (which returns a\n            torch Adam optimizer).\n        validate_spaces (Optional[Callable[[Policy, gym.Space, gym.Space,\n            AlgorithmConfigDict], None]]): Optional callable that takes the\n            Policy, observation_space, action_space, and config to check for\n            correctness. If None, no spaces checking will be done.\n        before_init (Optional[Callable[[Policy, gym.Space, gym.Space,\n            AlgorithmConfigDict], None]]): Optional callable to run at the\n            beginning of `Policy.__init__` that takes the same arguments as\n            the Policy constructor. If None, this step will be skipped.\n        before_loss_init (Optional[Callable[[Policy, gym.spaces.Space,\n            gym.spaces.Space, AlgorithmConfigDict], None]]): Optional callable to\n            run prior to loss init. If None, this step will be skipped.\n        after_init (Optional[Callable[[Policy, gym.Space, gym.Space,\n            AlgorithmConfigDict], None]]): DEPRECATED: Use `before_loss_init`\n            instead.\n        _after_loss_init (Optional[Callable[[Policy, gym.spaces.Space,\n            gym.spaces.Space, AlgorithmConfigDict], None]]): Optional callable to\n            run after the loss init. If None, this step will be skipped.\n            This will be deprecated at some point and renamed into `after_init`\n            to match `build_tf_policy()` behavior.\n        action_sampler_fn (Optional[Callable[[TensorType, List[TensorType]],\n            Tuple[TensorType, TensorType]]]): Optional callable returning a\n            sampled action and its log-likelihood given some (obs and state)\n            inputs. If None, will either use `action_distribution_fn` or\n            compute actions by calling self.model, then sampling from the\n            so parameterized action distribution.\n        action_distribution_fn (Optional[Callable[[Policy, ModelV2, TensorType,\n            TensorType, TensorType], Tuple[TensorType,\n            Type[TorchDistributionWrapper], List[TensorType]]]]): A callable\n            that takes the Policy, Model, the observation batch, an\n            explore-flag, a timestep, and an is_training flag and returns a\n            tuple of a) distribution inputs (parameters), b) a dist-class to\n            generate an action distribution object from, and c) internal-state\n            outputs (empty list if not applicable). If None, will either use\n            `action_sampler_fn` or compute actions by calling self.model,\n            then sampling from the parameterized action distribution.\n        make_model (Optional[Callable[[Policy, gym.spaces.Space,\n            gym.spaces.Space, AlgorithmConfigDict], ModelV2]]): Optional callable\n            that takes the same arguments as Policy.__init__ and returns a\n            model instance. The distribution class will be determined\n            automatically. Note: Only one of `make_model` or\n            `make_model_and_action_dist` should be provided. If both are None,\n            a default Model will be created.\n        make_model_and_action_dist (Optional[Callable[[Policy,\n            gym.spaces.Space, gym.spaces.Space, AlgorithmConfigDict],\n            Tuple[ModelV2, Type[TorchDistributionWrapper]]]]): Optional\n            callable that takes the same arguments as Policy.__init__ and\n            returns a tuple of model instance and torch action distribution\n            class.\n            Note: Only one of `make_model` or `make_model_and_action_dist`\n            should be provided. If both are None, a default Model will be\n            created.\n        compute_gradients_fn (Optional[Callable[\n            [Policy, SampleBatch], Tuple[ModelGradients, dict]]]): Optional\n            callable that the sampled batch an computes the gradients w.r.\n            to the loss function.\n            If None, will call the `TorchPolicy.compute_gradients()` method\n            instead.\n        apply_gradients_fn (Optional[Callable[[Policy,\n            "torch.optim.Optimizer"], None]]): Optional callable that\n            takes a grads list and applies these to the Model\'s parameters.\n            If None, will call the `TorchPolicy.apply_gradients()` method\n            instead.\n        mixins (Optional[List[type]]): Optional list of any class mixins for\n            the returned policy class. These mixins will be applied in order\n            and will have higher precedence than the TorchPolicy class.\n        get_batch_divisibility_req (Optional[Callable[[Policy], int]]):\n            Optional callable that returns the divisibility requirement for\n            sample batches. If None, will assume a value of 1.\n\n    Returns:\n        Type[TorchPolicy]: TorchPolicy child class constructed from the\n            specified args.\n    '
    original_kwargs = locals().copy()
    parent_cls = TorchPolicy
    base = add_mixins(parent_cls, mixins)

    class policy_cls(base):

        def __init__(self, obs_space, action_space, config):
            if False:
                return 10
            self.config = config
            self.framework = self.config['framework'] = framework
            if validate_spaces:
                validate_spaces(self, obs_space, action_space, self.config)
            if before_init:
                before_init(self, obs_space, action_space, self.config)
            if make_model:
                assert make_model_and_action_dist is None, 'Either `make_model` or `make_model_and_action_dist` must be None!'
                self.model = make_model(self, obs_space, action_space, config)
                (dist_class, _) = ModelCatalog.get_action_dist(action_space, self.config['model'], framework=framework)
            elif make_model_and_action_dist:
                (self.model, dist_class) = make_model_and_action_dist(self, obs_space, action_space, config)
            else:
                (dist_class, logit_dim) = ModelCatalog.get_action_dist(action_space, self.config['model'], framework=framework)
                self.model = ModelCatalog.get_model_v2(obs_space=obs_space, action_space=action_space, num_outputs=logit_dim, model_config=self.config['model'], framework=framework)
            model_cls = TorchModelV2 if framework == 'torch' else JAXModelV2
            assert isinstance(self.model, model_cls), 'ERROR: Generated Model must be a TorchModelV2 object!'
            self.parent_cls = parent_cls
            self.parent_cls.__init__(self, observation_space=obs_space, action_space=action_space, config=config, model=self.model, loss=None if self.config['in_evaluation'] else loss_fn, action_distribution_class=dist_class, action_sampler_fn=action_sampler_fn, action_distribution_fn=action_distribution_fn, max_seq_len=config['model']['max_seq_len'], get_batch_divisibility_req=get_batch_divisibility_req)
            self.view_requirements.update(self.model.view_requirements)
            _before_loss_init = before_loss_init or after_init
            if _before_loss_init:
                _before_loss_init(self, self.observation_space, self.action_space, config)
            self._initialize_loss_from_dummy_batch(auto_remove_unneeded_view_reqs=True, stats_fn=None if self.config['in_evaluation'] else stats_fn)
            if _after_loss_init:
                _after_loss_init(self, obs_space, action_space, config)
            self.global_timestep = 0

        @override(Policy)
        def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
            if False:
                while True:
                    i = 10
            with self._no_grad_context():
                sample_batch = super().postprocess_trajectory(sample_batch, other_agent_batches, episode)
                if postprocess_fn:
                    return postprocess_fn(self, sample_batch, other_agent_batches, episode)
                return sample_batch

        @override(parent_cls)
        def extra_grad_process(self, optimizer, loss):
            if False:
                while True:
                    i = 10
            'Called after optimizer.zero_grad() and loss.backward() calls.\n\n            Allows for gradient processing before optimizer.step() is called.\n            E.g. for gradient clipping.\n            '
            if extra_grad_process_fn:
                return extra_grad_process_fn(self, optimizer, loss)
            else:
                return parent_cls.extra_grad_process(self, optimizer, loss)

        @override(parent_cls)
        def extra_compute_grad_fetches(self):
            if False:
                for i in range(10):
                    print('nop')
            if extra_learn_fetches_fn:
                fetches = convert_to_numpy(extra_learn_fetches_fn(self))
                return dict({LEARNER_STATS_KEY: {}}, **fetches)
            else:
                return parent_cls.extra_compute_grad_fetches(self)

        @override(parent_cls)
        def compute_gradients(self, batch):
            if False:
                while True:
                    i = 10
            if compute_gradients_fn:
                return compute_gradients_fn(self, batch)
            else:
                return parent_cls.compute_gradients(self, batch)

        @override(parent_cls)
        def apply_gradients(self, gradients):
            if False:
                for i in range(10):
                    print('nop')
            if apply_gradients_fn:
                apply_gradients_fn(self, gradients)
            else:
                parent_cls.apply_gradients(self, gradients)

        @override(parent_cls)
        def extra_action_out(self, input_dict, state_batches, model, action_dist):
            if False:
                for i in range(10):
                    print('nop')
            with self._no_grad_context():
                if extra_action_out_fn:
                    stats_dict = extra_action_out_fn(self, input_dict, state_batches, model, action_dist)
                else:
                    stats_dict = parent_cls.extra_action_out(self, input_dict, state_batches, model, action_dist)
                return self._convert_to_numpy(stats_dict)

        @override(parent_cls)
        def optimizer(self):
            if False:
                print('Hello World!')
            if optimizer_fn:
                optimizers = optimizer_fn(self, self.config)
            else:
                optimizers = parent_cls.optimizer(self)
            return optimizers

        @override(parent_cls)
        def extra_grad_info(self, train_batch):
            if False:
                print('Hello World!')
            with self._no_grad_context():
                if stats_fn:
                    stats_dict = stats_fn(self, train_batch)
                else:
                    stats_dict = self.parent_cls.extra_grad_info(self, train_batch)
                return self._convert_to_numpy(stats_dict)

        def _no_grad_context(self):
            if False:
                return 10
            if self.framework == 'torch':
                return torch.no_grad()
            return NullContextManager()

        def _convert_to_numpy(self, data):
            if False:
                return 10
            if self.framework == 'torch':
                return convert_to_numpy(data)
            return data

    def with_updates(**overrides):
        if False:
            i = 10
            return i + 15
        'Creates a Torch|JAXPolicy cls based on settings of another one.\n\n        Keyword Args:\n            **overrides: The settings (passed into `build_torch_policy`) that\n                should be different from the class that this method is called\n                on.\n\n        Returns:\n            type: A new Torch|JAXPolicy sub-class.\n\n        Examples:\n        >> MySpecialDQNPolicyClass = DQNTorchPolicy.with_updates(\n        ..    name="MySpecialDQNPolicyClass",\n        ..    loss_function=[some_new_loss_function],\n        .. )\n        '
        return build_policy_class(**dict(original_kwargs, **overrides))
    policy_cls.with_updates = staticmethod(with_updates)
    policy_cls.__name__ = name
    policy_cls.__qualname__ = name
    return policy_cls