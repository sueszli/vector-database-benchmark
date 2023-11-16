import gymnasium as gym
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, TYPE_CHECKING
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.dynamic_tf_policy import DynamicTFPolicy
from ray.rllib.policy import eager_tf_policy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils import add_mixins, force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.deprecation import deprecation_warning, DEPRECATED_VALUE
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import AgentID, ModelGradients, TensorType, AlgorithmConfigDict
if TYPE_CHECKING:
    from ray.rllib.evaluation import Episode
(tf1, tf, tfv) = try_import_tf()

@DeveloperAPI
def build_tf_policy(name: str, *, loss_fn: Callable[[Policy, ModelV2, Type[TFActionDistribution], SampleBatch], Union[TensorType, List[TensorType]]], get_default_config: Optional[Callable[[None], AlgorithmConfigDict]]=None, postprocess_fn: Optional[Callable[[Policy, SampleBatch, Optional[Dict[AgentID, SampleBatch]], Optional['Episode']], SampleBatch]]=None, stats_fn: Optional[Callable[[Policy, SampleBatch], Dict[str, TensorType]]]=None, optimizer_fn: Optional[Callable[[Policy, AlgorithmConfigDict], 'tf.keras.optimizers.Optimizer']]=None, compute_gradients_fn: Optional[Callable[[Policy, 'tf.keras.optimizers.Optimizer', TensorType], ModelGradients]]=None, apply_gradients_fn: Optional[Callable[[Policy, 'tf.keras.optimizers.Optimizer', ModelGradients], 'tf.Operation']]=None, grad_stats_fn: Optional[Callable[[Policy, SampleBatch, ModelGradients], Dict[str, TensorType]]]=None, extra_action_out_fn: Optional[Callable[[Policy], Dict[str, TensorType]]]=None, extra_learn_fetches_fn: Optional[Callable[[Policy], Dict[str, TensorType]]]=None, validate_spaces: Optional[Callable[[Policy, gym.Space, gym.Space, AlgorithmConfigDict], None]]=None, before_init: Optional[Callable[[Policy, gym.Space, gym.Space, AlgorithmConfigDict], None]]=None, before_loss_init: Optional[Callable[[Policy, gym.spaces.Space, gym.spaces.Space, AlgorithmConfigDict], None]]=None, after_init: Optional[Callable[[Policy, gym.Space, gym.Space, AlgorithmConfigDict], None]]=None, make_model: Optional[Callable[[Policy, gym.spaces.Space, gym.spaces.Space, AlgorithmConfigDict], ModelV2]]=None, action_sampler_fn: Optional[Callable[[TensorType, List[TensorType]], Tuple[TensorType, TensorType]]]=None, action_distribution_fn: Optional[Callable[[Policy, ModelV2, TensorType, TensorType, TensorType], Tuple[TensorType, type, List[TensorType]]]]=None, mixins: Optional[List[type]]=None, get_batch_divisibility_req: Optional[Callable[[Policy], int]]=None, obs_include_prev_action_reward=DEPRECATED_VALUE, extra_action_fetches_fn=None, gradients_fn=None) -> Type[DynamicTFPolicy]:
    if False:
        print('Hello World!')
    'Helper function for creating a dynamic tf policy at runtime.\n\n    Functions will be run in this order to initialize the policy:\n        1. Placeholder setup: postprocess_fn\n        2. Loss init: loss_fn, stats_fn\n        3. Optimizer init: optimizer_fn, gradients_fn, apply_gradients_fn,\n                           grad_stats_fn\n\n    This means that you can e.g., depend on any policy attributes created in\n    the running of `loss_fn` in later functions such as `stats_fn`.\n\n    In eager mode, the following functions will be run repeatedly on each\n    eager execution: loss_fn, stats_fn, gradients_fn, apply_gradients_fn,\n    and grad_stats_fn.\n\n    This means that these functions should not define any variables internally,\n    otherwise they will fail in eager mode execution. Variable should only\n    be created in make_model (if defined).\n\n    Args:\n        name: Name of the policy (e.g., "PPOTFPolicy").\n        loss_fn (Callable[[\n            Policy, ModelV2, Type[TFActionDistribution], SampleBatch],\n            Union[TensorType, List[TensorType]]]): Callable for calculating a\n            loss tensor.\n        get_default_config (Optional[Callable[[None], AlgorithmConfigDict]]):\n            Optional callable that returns the default config to merge with any\n            overrides. If None, uses only(!) the user-provided\n            PartialAlgorithmConfigDict as dict for this Policy.\n        postprocess_fn (Optional[Callable[[Policy, SampleBatch,\n            Optional[Dict[AgentID, SampleBatch]], Episode], None]]):\n            Optional callable for post-processing experience batches (called\n            after the parent class\' `postprocess_trajectory` method).\n        stats_fn (Optional[Callable[[Policy, SampleBatch],\n            Dict[str, TensorType]]]): Optional callable that returns a dict of\n            TF tensors to fetch given the policy and batch input tensors. If\n            None, will not compute any stats.\n        optimizer_fn (Optional[Callable[[Policy, AlgorithmConfigDict],\n            "tf.keras.optimizers.Optimizer"]]): Optional callable that returns\n            a tf.Optimizer given the policy and config. If None, will call\n            the base class\' `optimizer()` method instead (which returns a\n            tf1.train.AdamOptimizer).\n        compute_gradients_fn (Optional[Callable[[Policy,\n            "tf.keras.optimizers.Optimizer", TensorType], ModelGradients]]):\n            Optional callable that returns a list of gradients. If None,\n            this defaults to optimizer.compute_gradients([loss]).\n        apply_gradients_fn (Optional[Callable[[Policy,\n            "tf.keras.optimizers.Optimizer", ModelGradients],\n            "tf.Operation"]]): Optional callable that returns an apply\n            gradients op given policy, tf-optimizer, and grads_and_vars. If\n            None, will call the base class\' `build_apply_op()` method instead.\n        grad_stats_fn (Optional[Callable[[Policy, SampleBatch, ModelGradients],\n            Dict[str, TensorType]]]): Optional callable that returns a dict of\n            TF fetches given the policy, batch input, and gradient tensors. If\n            None, will not collect any gradient stats.\n        extra_action_out_fn (Optional[Callable[[Policy],\n            Dict[str, TensorType]]]): Optional callable that returns\n            a dict of TF fetches given the policy object. If None, will not\n            perform any extra fetches.\n        extra_learn_fetches_fn (Optional[Callable[[Policy],\n            Dict[str, TensorType]]]): Optional callable that returns a dict of\n            extra values to fetch and return when learning on a batch. If None,\n            will call the base class\' `extra_compute_grad_fetches()` method\n            instead.\n        validate_spaces (Optional[Callable[[Policy, gym.Space, gym.Space,\n            AlgorithmConfigDict], None]]): Optional callable that takes the\n            Policy, observation_space, action_space, and config to check\n            the spaces for correctness. If None, no spaces checking will be\n            done.\n        before_init (Optional[Callable[[Policy, gym.Space, gym.Space,\n            AlgorithmConfigDict], None]]): Optional callable to run at the\n            beginning of policy init that takes the same arguments as the\n            policy constructor. If None, this step will be skipped.\n        before_loss_init (Optional[Callable[[Policy, gym.spaces.Space,\n            gym.spaces.Space, AlgorithmConfigDict], None]]): Optional callable to\n            run prior to loss init. If None, this step will be skipped.\n        after_init (Optional[Callable[[Policy, gym.Space, gym.Space,\n            AlgorithmConfigDict], None]]): Optional callable to run at the end of\n            policy init. If None, this step will be skipped.\n        make_model (Optional[Callable[[Policy, gym.spaces.Space,\n            gym.spaces.Space, AlgorithmConfigDict], ModelV2]]): Optional callable\n            that returns a ModelV2 object.\n            All policy variables should be created in this function. If None,\n            a default ModelV2 object will be created.\n        action_sampler_fn (Optional[Callable[[TensorType, List[TensorType]],\n            Tuple[TensorType, TensorType]]]): A callable returning a sampled\n            action and its log-likelihood given observation and state inputs.\n            If None, will either use `action_distribution_fn` or\n            compute actions by calling self.model, then sampling from the\n            so parameterized action distribution.\n        action_distribution_fn (Optional[Callable[[Policy, ModelV2, TensorType,\n            TensorType, TensorType],\n            Tuple[TensorType, type, List[TensorType]]]]): Optional callable\n            returning distribution inputs (parameters), a dist-class to\n            generate an action distribution object from, and internal-state\n            outputs (or an empty list if not applicable). If None, will either\n            use `action_sampler_fn` or compute actions by calling self.model,\n            then sampling from the so parameterized action distribution.\n        mixins (Optional[List[type]]): Optional list of any class mixins for\n            the returned policy class. These mixins will be applied in order\n            and will have higher precedence than the DynamicTFPolicy class.\n        get_batch_divisibility_req (Optional[Callable[[Policy], int]]):\n            Optional callable that returns the divisibility requirement for\n            sample batches. If None, will assume a value of 1.\n\n    Returns:\n        Type[DynamicTFPolicy]: A child class of DynamicTFPolicy based on the\n            specified args.\n    '
    original_kwargs = locals().copy()
    base = add_mixins(DynamicTFPolicy, mixins)
    if obs_include_prev_action_reward != DEPRECATED_VALUE:
        deprecation_warning(old='obs_include_prev_action_reward', error=True)
    if extra_action_fetches_fn is not None:
        deprecation_warning(old='extra_action_fetches_fn', new='extra_action_out_fn', error=True)
    if gradients_fn is not None:
        deprecation_warning(old='gradients_fn', new='compute_gradients_fn', error=True)

    class policy_cls(base):

        def __init__(self, obs_space, action_space, config, existing_model=None, existing_inputs=None):
            if False:
                print('Hello World!')
            if validate_spaces:
                validate_spaces(self, obs_space, action_space, config)
            if before_init:
                before_init(self, obs_space, action_space, config)

            def before_loss_init_wrapper(policy, obs_space, action_space, config):
                if False:
                    for i in range(10):
                        print('nop')
                if before_loss_init:
                    before_loss_init(policy, obs_space, action_space, config)
                if extra_action_out_fn is None or policy._is_tower:
                    extra_action_fetches = {}
                else:
                    extra_action_fetches = extra_action_out_fn(policy)
                if hasattr(policy, '_extra_action_fetches'):
                    policy._extra_action_fetches.update(extra_action_fetches)
                else:
                    policy._extra_action_fetches = extra_action_fetches
            DynamicTFPolicy.__init__(self, obs_space=obs_space, action_space=action_space, config=config, loss_fn=loss_fn, stats_fn=stats_fn, grad_stats_fn=grad_stats_fn, before_loss_init=before_loss_init_wrapper, make_model=make_model, action_sampler_fn=action_sampler_fn, action_distribution_fn=action_distribution_fn, existing_inputs=existing_inputs, existing_model=existing_model, get_batch_divisibility_req=get_batch_divisibility_req)
            if after_init:
                after_init(self, obs_space, action_space, config)
            self.global_timestep = 0

        @override(Policy)
        def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
            if False:
                for i in range(10):
                    print('nop')
            sample_batch = Policy.postprocess_trajectory(self, sample_batch)
            if postprocess_fn:
                return postprocess_fn(self, sample_batch, other_agent_batches, episode)
            return sample_batch

        @override(TFPolicy)
        def optimizer(self):
            if False:
                i = 10
                return i + 15
            if optimizer_fn:
                optimizers = optimizer_fn(self, self.config)
            else:
                optimizers = base.optimizer(self)
            optimizers = force_list(optimizers)
            if self.exploration:
                optimizers = self.exploration.get_exploration_optimizer(optimizers)
            if not optimizers:
                return None
            elif self.config['_tf_policy_handles_more_than_one_loss']:
                return optimizers
            else:
                return optimizers[0]

        @override(TFPolicy)
        def gradients(self, optimizer, loss):
            if False:
                print('Hello World!')
            optimizers = force_list(optimizer)
            losses = force_list(loss)
            if compute_gradients_fn:
                if self.config['_tf_policy_handles_more_than_one_loss']:
                    return compute_gradients_fn(self, optimizers, losses)
                else:
                    return compute_gradients_fn(self, optimizers[0], losses[0])
            else:
                return base.gradients(self, optimizers, losses)

        @override(TFPolicy)
        def build_apply_op(self, optimizer, grads_and_vars):
            if False:
                for i in range(10):
                    print('nop')
            if apply_gradients_fn:
                return apply_gradients_fn(self, optimizer, grads_and_vars)
            else:
                return base.build_apply_op(self, optimizer, grads_and_vars)

        @override(TFPolicy)
        def extra_compute_action_fetches(self):
            if False:
                i = 10
                return i + 15
            return dict(base.extra_compute_action_fetches(self), **self._extra_action_fetches)

        @override(TFPolicy)
        def extra_compute_grad_fetches(self):
            if False:
                return 10
            if extra_learn_fetches_fn:
                return dict({LEARNER_STATS_KEY: {}}, **extra_learn_fetches_fn(self))
            else:
                return base.extra_compute_grad_fetches(self)

    def with_updates(**overrides):
        if False:
            i = 10
            return i + 15
        'Allows creating a TFPolicy cls based on settings of another one.\n\n        Keyword Args:\n            **overrides: The settings (passed into `build_tf_policy`) that\n                should be different from the class that this method is called\n                on.\n\n        Returns:\n            type: A new TFPolicy sub-class.\n\n        Examples:\n        >> MySpecialDQNPolicyClass = DQNTFPolicy.with_updates(\n        ..    name="MySpecialDQNPolicyClass",\n        ..    loss_function=[some_new_loss_function],\n        .. )\n        '
        return build_tf_policy(**dict(original_kwargs, **overrides))

    def as_eager():
        if False:
            while True:
                i = 10
        return eager_tf_policy._build_eager_tf_policy(**original_kwargs)
    policy_cls.with_updates = staticmethod(with_updates)
    policy_cls.as_eager = staticmethod(as_eager)
    policy_cls.__name__ = name
    policy_cls.__qualname__ = name
    return policy_cls