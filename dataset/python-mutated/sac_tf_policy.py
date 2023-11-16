"""
TensorFlow policy class used for SAC.
"""
import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from functools import partial
import logging
from typing import Dict, List, Optional, Tuple, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.dqn.dqn_tf_policy import postprocess_nstep_and_prio, PRIO_WEIGHTS
from ray.rllib.algorithms.sac.sac_tf_model import SACTFModel
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Beta, Categorical, DiagGaussian, Dirichlet, SquashedGaussian, TFActionDistribution
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import get_variable, try_import_tf
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.tf_utils import huber_loss, make_tf_callable
from ray.rllib.utils.typing import AgentID, LocalOptimizer, ModelGradients, TensorType, AlgorithmConfigDict
(tf1, tf, tfv) = try_import_tf()
logger = logging.getLogger(__name__)

def build_sac_model(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> ModelV2:
    if False:
        i = 10
        return i + 15
    'Constructs the necessary ModelV2 for the Policy and returns it.\n\n    Args:\n        policy: The TFPolicy that will use the models.\n        obs_space (gym.spaces.Space): The observation space.\n        action_space (gym.spaces.Space): The action space.\n        config: The SACConfig object.\n\n    Returns:\n        ModelV2: The ModelV2 to be used by the Policy. Note: An additional\n            target model will be created in this function and assigned to\n            `policy.target_model`.\n    '
    policy_model_config = copy.deepcopy(config['model'])
    policy_model_config.update(config['policy_model_config'])
    q_model_config = copy.deepcopy(config['model'])
    q_model_config.update(config['q_model_config'])
    default_model_cls = SACTorchModel if config['framework'] == 'torch' else SACTFModel
    model = ModelCatalog.get_model_v2(obs_space=obs_space, action_space=action_space, num_outputs=None, model_config=config['model'], framework=config['framework'], default_model=default_model_cls, name='sac_model', policy_model_config=policy_model_config, q_model_config=q_model_config, twin_q=config['twin_q'], initial_alpha=config['initial_alpha'], target_entropy=config['target_entropy'])
    assert isinstance(model, default_model_cls)
    policy.target_model = ModelCatalog.get_model_v2(obs_space=obs_space, action_space=action_space, num_outputs=None, model_config=config['model'], framework=config['framework'], default_model=default_model_cls, name='target_sac_model', policy_model_config=policy_model_config, q_model_config=q_model_config, twin_q=config['twin_q'], initial_alpha=config['initial_alpha'], target_entropy=config['target_entropy'])
    assert isinstance(policy.target_model, default_model_cls)
    return model

def postprocess_trajectory(policy: Policy, sample_batch: SampleBatch, other_agent_batches: Optional[Dict[AgentID, SampleBatch]]=None, episode: Optional[Episode]=None) -> SampleBatch:
    if False:
        for i in range(10):
            print('nop')
    "Postprocesses a trajectory and returns the processed trajectory.\n\n    The trajectory contains only data from one episode and from one agent.\n    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may\n    contain a truncated (at-the-end) episode, in case the\n    `config.rollout_fragment_length` was reached by the sampler.\n    - If `config.batch_mode=complete_episodes`, sample_batch will contain\n    exactly one episode (no matter how long).\n    New columns can be added to sample_batch and existing ones may be altered.\n\n    Args:\n        policy: The Policy used to generate the trajectory\n            (`sample_batch`)\n        sample_batch: The SampleBatch to postprocess.\n        other_agent_batches (Optional[Dict[AgentID, SampleBatch]]): Optional\n            dict of AgentIDs mapping to other agents' trajectory data (from the\n            same episode). NOTE: The other agents use the same policy.\n        episode (Optional[Episode]): Optional multi-agent episode\n            object in which the agents operated.\n\n    Returns:\n        SampleBatch: The postprocessed, modified SampleBatch (or a new one).\n    "
    return postprocess_nstep_and_prio(policy, sample_batch)

def _get_dist_class(policy: Policy, config: AlgorithmConfigDict, action_space: gym.spaces.Space) -> Type[TFActionDistribution]:
    if False:
        i = 10
        return i + 15
    "Helper function to return a dist class based on config and action space.\n\n    Args:\n        policy: The policy for which to return the action\n            dist class.\n        config: The Algorithm's config dict.\n        action_space (gym.spaces.Space): The action space used.\n\n    Returns:\n        Type[TFActionDistribution]: A TF distribution class.\n    "
    if hasattr(policy, 'dist_class') and policy.dist_class is not None:
        return policy.dist_class
    elif config['model'].get('custom_action_dist'):
        (action_dist_class, _) = ModelCatalog.get_action_dist(action_space, config['model'], framework='tf')
        return action_dist_class
    elif isinstance(action_space, Discrete):
        return Categorical
    elif isinstance(action_space, Simplex):
        return Dirichlet
    else:
        assert isinstance(action_space, Box)
        if config['normalize_actions']:
            return SquashedGaussian if not config['_use_beta_distribution'] else Beta
        else:
            return DiagGaussian

def get_distribution_inputs_and_class(policy: Policy, model: ModelV2, obs_batch: TensorType, *, explore: bool=True, **kwargs) -> Tuple[TensorType, Type[TFActionDistribution], List[TensorType]]:
    if False:
        i = 10
        return i + 15
    'The action distribution function to be used the algorithm.\n\n    An action distribution function is used to customize the choice of action\n    distribution class and the resulting action distribution inputs (to\n    parameterize the distribution object).\n    After parameterizing the distribution, a `sample()` call\n    will be made on it to generate actions.\n\n    Args:\n        policy: The Policy being queried for actions and calling this\n            function.\n        model: The SAC specific Model to use to generate the\n            distribution inputs (see sac_tf|torch_model.py). Must support the\n            `get_action_model_outputs` method.\n        obs_batch: The observations to be used as inputs to the\n            model.\n        explore: Whether to activate exploration or not.\n\n    Returns:\n        Tuple[TensorType, Type[TFActionDistribution], List[TensorType]]: The\n            dist inputs, dist class, and a list of internal state outputs\n            (in the RNN case).\n    '
    (forward_out, state_out) = model(SampleBatch(obs=obs_batch, _is_training=policy._get_is_training_placeholder()), [], None)
    (distribution_inputs, _) = model.get_action_model_outputs(forward_out)
    action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
    return (distribution_inputs, action_dist_class, state_out)

def sac_actor_critic_loss(policy: Policy, model: ModelV2, dist_class: Type[TFActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    if False:
        i = 10
        return i + 15
    'Constructs the loss for the Soft Actor Critic.\n\n    Args:\n        policy: The Policy to calculate the loss for.\n        model (ModelV2): The Model to calculate the loss for.\n        dist_class (Type[ActionDistribution]: The action distr. class.\n        train_batch: The training data.\n\n    Returns:\n        Union[TensorType, List[TensorType]]: A single loss tensor or a list\n            of loss tensors.\n    '
    deterministic = policy.config['_deterministic_loss']
    _is_training = policy._get_is_training_placeholder()
    (model_out_t, _) = model(SampleBatch(obs=train_batch[SampleBatch.CUR_OBS], _is_training=_is_training), [], None)
    (model_out_tp1, _) = model(SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=_is_training), [], None)
    (target_model_out_tp1, _) = policy.target_model(SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=_is_training), [], None)
    if model.discrete:
        (action_dist_inputs_t, _) = model.get_action_model_outputs(model_out_t)
        log_pis_t = tf.nn.log_softmax(action_dist_inputs_t, -1)
        policy_t = tf.math.exp(log_pis_t)
        (action_dist_inputs_tp1, _) = model.get_action_model_outputs(model_out_tp1)
        log_pis_tp1 = tf.nn.log_softmax(action_dist_inputs_tp1, -1)
        policy_tp1 = tf.math.exp(log_pis_tp1)
        (q_t, _) = model.get_q_values(model_out_t)
        (q_tp1, _) = policy.target_model.get_q_values(target_model_out_tp1)
        if policy.config['twin_q']:
            (twin_q_t, _) = model.get_twin_q_values(model_out_t)
            (twin_q_tp1, _) = policy.target_model.get_twin_q_values(target_model_out_tp1)
            q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)
        q_tp1 -= model.alpha * log_pis_tp1
        one_hot = tf.one_hot(train_batch[SampleBatch.ACTIONS], depth=q_t.shape.as_list()[-1])
        q_t_selected = tf.reduce_sum(q_t * one_hot, axis=-1)
        if policy.config['twin_q']:
            twin_q_t_selected = tf.reduce_sum(twin_q_t * one_hot, axis=-1)
        q_tp1_best = tf.reduce_sum(tf.multiply(policy_tp1, q_tp1), axis=-1)
        q_tp1_best_masked = (1.0 - tf.cast(train_batch[SampleBatch.TERMINATEDS], tf.float32)) * q_tp1_best
    else:
        action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
        (action_dist_inputs_t, _) = model.get_action_model_outputs(model_out_t)
        action_dist_t = action_dist_class(action_dist_inputs_t, policy.model)
        policy_t = action_dist_t.sample() if not deterministic else action_dist_t.deterministic_sample()
        log_pis_t = tf.expand_dims(action_dist_t.logp(policy_t), -1)
        (action_dist_inputs_tp1, _) = model.get_action_model_outputs(model_out_tp1)
        action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, policy.model)
        policy_tp1 = action_dist_tp1.sample() if not deterministic else action_dist_tp1.deterministic_sample()
        log_pis_tp1 = tf.expand_dims(action_dist_tp1.logp(policy_tp1), -1)
        (q_t, _) = model.get_q_values(model_out_t, tf.cast(train_batch[SampleBatch.ACTIONS], tf.float32))
        if policy.config['twin_q']:
            (twin_q_t, _) = model.get_twin_q_values(model_out_t, tf.cast(train_batch[SampleBatch.ACTIONS], tf.float32))
        (q_t_det_policy, _) = model.get_q_values(model_out_t, policy_t)
        if policy.config['twin_q']:
            (twin_q_t_det_policy, _) = model.get_twin_q_values(model_out_t, policy_t)
            q_t_det_policy = tf.reduce_min((q_t_det_policy, twin_q_t_det_policy), axis=0)
        (q_tp1, _) = policy.target_model.get_q_values(target_model_out_tp1, policy_tp1)
        if policy.config['twin_q']:
            (twin_q_tp1, _) = policy.target_model.get_twin_q_values(target_model_out_tp1, policy_tp1)
            q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)
        q_t_selected = tf.squeeze(q_t, axis=len(q_t.shape) - 1)
        if policy.config['twin_q']:
            twin_q_t_selected = tf.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
        q_tp1 -= model.alpha * log_pis_tp1
        q_tp1_best = tf.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
        q_tp1_best_masked = (1.0 - tf.cast(train_batch[SampleBatch.TERMINATEDS], tf.float32)) * q_tp1_best
    q_t_selected_target = tf.stop_gradient(tf.cast(train_batch[SampleBatch.REWARDS], tf.float32) + policy.config['gamma'] ** policy.config['n_step'] * q_tp1_best_masked)
    base_td_error = tf.math.abs(q_t_selected - q_t_selected_target)
    if policy.config['twin_q']:
        twin_td_error = tf.math.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error
    prio_weights = tf.cast(train_batch[PRIO_WEIGHTS], tf.float32)
    critic_loss = [tf.reduce_mean(prio_weights * huber_loss(base_td_error))]
    if policy.config['twin_q']:
        critic_loss.append(tf.reduce_mean(prio_weights * huber_loss(twin_td_error)))
    if model.discrete:
        alpha_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.stop_gradient(policy_t), -model.log_alpha * tf.stop_gradient(log_pis_t + model.target_entropy)), axis=-1))
        actor_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(policy_t, model.alpha * log_pis_t - tf.stop_gradient(q_t)), axis=-1))
    else:
        alpha_loss = -tf.reduce_mean(model.log_alpha * tf.stop_gradient(log_pis_t + model.target_entropy))
        actor_loss = tf.reduce_mean(model.alpha * log_pis_t - q_t_det_policy)
    policy.policy_t = policy_t
    policy.q_t = q_t
    policy.td_error = td_error
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.alpha_loss = alpha_loss
    policy.alpha_value = model.alpha
    policy.target_entropy = model.target_entropy
    return actor_loss + tf.math.add_n(critic_loss) + alpha_loss

def compute_and_clip_gradients(policy: Policy, optimizer: LocalOptimizer, loss: TensorType) -> ModelGradients:
    if False:
        for i in range(10):
            print('nop')
    'Gradients computing function (from loss tensor, using local optimizer).\n\n    Note: For SAC, optimizer and loss are ignored b/c we have 3\n    losses and 3 local optimizers (all stored in policy).\n    `optimizer` will be used, though, in the tf-eager case b/c it is then a\n    fake optimizer (OptimizerWrapper) object with a `tape` property to\n    generate a GradientTape object for gradient recording.\n\n    Args:\n        policy: The Policy object that generated the loss tensor and\n            that holds the given local optimizer.\n        optimizer: The tf (local) optimizer object to\n            calculate the gradients with.\n        loss: The loss tensor for which gradients should be\n            calculated.\n\n    Returns:\n        ModelGradients: List of the possibly clipped gradients- and variable\n            tuples.\n    '
    if policy.config['framework'] == 'tf2':
        tape = optimizer.tape
        pol_weights = policy.model.policy_variables()
        actor_grads_and_vars = list(zip(tape.gradient(policy.actor_loss, pol_weights), pol_weights))
        q_weights = policy.model.q_variables()
        if policy.config['twin_q']:
            half_cutoff = len(q_weights) // 2
            grads_1 = tape.gradient(policy.critic_loss[0], q_weights[:half_cutoff])
            grads_2 = tape.gradient(policy.critic_loss[1], q_weights[half_cutoff:])
            critic_grads_and_vars = list(zip(grads_1, q_weights[:half_cutoff])) + list(zip(grads_2, q_weights[half_cutoff:]))
        else:
            critic_grads_and_vars = list(zip(tape.gradient(policy.critic_loss[0], q_weights), q_weights))
        alpha_vars = [policy.model.log_alpha]
        alpha_grads_and_vars = list(zip(tape.gradient(policy.alpha_loss, alpha_vars), alpha_vars))
    else:
        actor_grads_and_vars = policy._actor_optimizer.compute_gradients(policy.actor_loss, var_list=policy.model.policy_variables())
        q_weights = policy.model.q_variables()
        if policy.config['twin_q']:
            half_cutoff = len(q_weights) // 2
            (base_q_optimizer, twin_q_optimizer) = policy._critic_optimizer
            critic_grads_and_vars = base_q_optimizer.compute_gradients(policy.critic_loss[0], var_list=q_weights[:half_cutoff]) + twin_q_optimizer.compute_gradients(policy.critic_loss[1], var_list=q_weights[half_cutoff:])
        else:
            critic_grads_and_vars = policy._critic_optimizer[0].compute_gradients(policy.critic_loss[0], var_list=q_weights)
        alpha_grads_and_vars = policy._alpha_optimizer.compute_gradients(policy.alpha_loss, var_list=[policy.model.log_alpha])
    if policy.config['grad_clip']:
        clip_func = partial(tf.clip_by_norm, clip_norm=policy.config['grad_clip'])
    else:
        clip_func = tf.identity
    policy._actor_grads_and_vars = [(clip_func(g), v) for (g, v) in actor_grads_and_vars if g is not None]
    policy._critic_grads_and_vars = [(clip_func(g), v) for (g, v) in critic_grads_and_vars if g is not None]
    policy._alpha_grads_and_vars = [(clip_func(g), v) for (g, v) in alpha_grads_and_vars if g is not None]
    grads_and_vars = policy._actor_grads_and_vars + policy._critic_grads_and_vars + policy._alpha_grads_and_vars
    return grads_and_vars

def apply_gradients(policy: Policy, optimizer: LocalOptimizer, grads_and_vars: ModelGradients) -> Union['tf.Operation', None]:
    if False:
        for i in range(10):
            print('nop')
    'Gradients applying function (from list of "grad_and_var" tuples).\n\n    Note: For SAC, optimizer and grads_and_vars are ignored b/c we have 3\n    losses and optimizers (stored in policy).\n\n    Args:\n        policy: The Policy object whose Model(s) the given gradients\n            should be applied to.\n        optimizer: The tf (local) optimizer object through\n            which to apply the gradients.\n        grads_and_vars: The list of grad_and_var tuples to\n            apply via the given optimizer.\n\n    Returns:\n        Union[tf.Operation, None]: The tf op to be used to run the apply\n            operation. None for eager mode.\n    '
    actor_apply_ops = policy._actor_optimizer.apply_gradients(policy._actor_grads_and_vars)
    cgrads = policy._critic_grads_and_vars
    half_cutoff = len(cgrads) // 2
    if policy.config['twin_q']:
        critic_apply_ops = [policy._critic_optimizer[0].apply_gradients(cgrads[:half_cutoff]), policy._critic_optimizer[1].apply_gradients(cgrads[half_cutoff:])]
    else:
        critic_apply_ops = [policy._critic_optimizer[0].apply_gradients(cgrads)]
    if policy.config['framework'] == 'tf2':
        policy._alpha_optimizer.apply_gradients(policy._alpha_grads_and_vars)
        return
    else:
        alpha_apply_ops = policy._alpha_optimizer.apply_gradients(policy._alpha_grads_and_vars, global_step=tf1.train.get_or_create_global_step())
        return tf.group([actor_apply_ops, alpha_apply_ops] + critic_apply_ops)

def stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    if False:
        while True:
            i = 10
    'Stats function for SAC. Returns a dict with important loss stats.\n\n    Args:\n        policy: The Policy to generate stats for.\n        train_batch: The SampleBatch (already) used for training.\n\n    Returns:\n        Dict[str, TensorType]: The stats dict.\n    '
    return {'mean_td_error': tf.reduce_mean(policy.td_error), 'actor_loss': tf.reduce_mean(policy.actor_loss), 'critic_loss': tf.reduce_mean(policy.critic_loss), 'alpha_loss': tf.reduce_mean(policy.alpha_loss), 'alpha_value': tf.reduce_mean(policy.alpha_value), 'target_entropy': tf.constant(policy.target_entropy), 'mean_q': tf.reduce_mean(policy.q_t), 'max_q': tf.reduce_max(policy.q_t), 'min_q': tf.reduce_min(policy.q_t)}

class ActorCriticOptimizerMixin:
    """Mixin class to generate the necessary optimizers for actor-critic algos.

    - Creates global step for counting the number of update operations.
    - Creates separate optimizers for actor, critic, and alpha.
    """

    def __init__(self, config):
        if False:
            return 10
        if config['framework'] == 'tf2':
            self.global_step = get_variable(0, tf_name='global_step')
            self._actor_optimizer = tf.keras.optimizers.Adam(learning_rate=config['optimization']['actor_learning_rate'])
            self._critic_optimizer = [tf.keras.optimizers.Adam(learning_rate=config['optimization']['critic_learning_rate'])]
            if config['twin_q']:
                self._critic_optimizer.append(tf.keras.optimizers.Adam(learning_rate=config['optimization']['critic_learning_rate']))
            self._alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=config['optimization']['entropy_learning_rate'])
        else:
            self.global_step = tf1.train.get_or_create_global_step()
            self._actor_optimizer = tf1.train.AdamOptimizer(learning_rate=config['optimization']['actor_learning_rate'])
            self._critic_optimizer = [tf1.train.AdamOptimizer(learning_rate=config['optimization']['critic_learning_rate'])]
            if config['twin_q']:
                self._critic_optimizer.append(tf1.train.AdamOptimizer(learning_rate=config['optimization']['critic_learning_rate']))
            self._alpha_optimizer = tf1.train.AdamOptimizer(learning_rate=config['optimization']['entropy_learning_rate'])

def setup_early_mixins(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> None:
    if False:
        while True:
            i = 10
    "Call mixin classes' constructors before Policy's initialization.\n\n    Adds the necessary optimizers to the given Policy.\n\n    Args:\n        policy: The Policy object.\n        obs_space (gym.spaces.Space): The Policy's observation space.\n        action_space (gym.spaces.Space): The Policy's action space.\n        config: The Policy's config.\n    "
    ActorCriticOptimizerMixin.__init__(policy, config)

class ComputeTDErrorMixin:

    def __init__(self, loss_fn):
        if False:
            while True:
                i = 10

        @make_tf_callable(self.get_session(), dynamic_shape=True)
        def compute_td_error(obs_t, act_t, rew_t, obs_tp1, terminateds_mask, importance_weights):
            if False:
                while True:
                    i = 10
            loss_fn(self, self.model, None, {SampleBatch.CUR_OBS: tf.convert_to_tensor(obs_t), SampleBatch.ACTIONS: tf.convert_to_tensor(act_t), SampleBatch.REWARDS: tf.convert_to_tensor(rew_t), SampleBatch.NEXT_OBS: tf.convert_to_tensor(obs_tp1), SampleBatch.TERMINATEDS: tf.convert_to_tensor(terminateds_mask), PRIO_WEIGHTS: tf.convert_to_tensor(importance_weights)})
            return self.td_error
        self.compute_td_error = compute_td_error

def setup_mid_mixins(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> None:
    if False:
        print('Hello World!')
    "Call mixin classes' constructors before Policy's loss initialization.\n\n    Adds the `compute_td_error` method to the given policy.\n    Calling `compute_td_error` with batch data will re-calculate the loss\n    on that batch AND return the per-batch-item TD-error for prioritized\n    replay buffer record weight updating (in case a prioritized replay buffer\n    is used).\n\n    Args:\n        policy: The Policy object.\n        obs_space (gym.spaces.Space): The Policy's observation space.\n        action_space (gym.spaces.Space): The Policy's action space.\n        config: The Policy's config.\n    "
    ComputeTDErrorMixin.__init__(policy, sac_actor_critic_loss)

def setup_late_mixins(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> None:
    if False:
        i = 10
        return i + 15
    'Call mixin classes\' constructors after Policy initialization.\n\n    Adds the `update_target` method to the given policy.\n    Calling `update_target` updates all target Q-networks\' weights from their\n    respective "main" Q-metworks, based on tau (smooth, partial updating).\n\n    Args:\n        policy: The Policy object.\n        obs_space (gym.spaces.Space): The Policy\'s observation space.\n        action_space (gym.spaces.Space): The Policy\'s action space.\n        config: The Policy\'s config.\n    '
    TargetNetworkMixin.__init__(policy)

def validate_spaces(policy: Policy, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> None:
    if False:
        i = 10
        return i + 15
    "Validates the observation- and action spaces used for the Policy.\n\n    Args:\n        policy: The policy, whose spaces are being validated.\n        observation_space (gym.spaces.Space): The observation space to\n            validate.\n        action_space (gym.spaces.Space): The action space to validate.\n        config: The Policy's config dict.\n\n    Raises:\n        UnsupportedSpaceException: If one of the spaces is not supported.\n    "
    if not isinstance(action_space, (Box, Discrete, Simplex)):
        raise UnsupportedSpaceException('Action space ({}) of {} is not supported for SAC. Must be [Box|Discrete|Simplex].'.format(action_space, policy))
    elif isinstance(action_space, (Box, Simplex)) and len(action_space.shape) > 1:
        raise UnsupportedSpaceException('Action space ({}) of {} has multiple dimensions {}. '.format(action_space, policy, action_space.shape) + 'Consider reshaping this into a single dimension, using a Tuple action space, or the multi-agent API.')
SACTFPolicy = build_tf_policy(name='SACTFPolicy', get_default_config=lambda : ray.rllib.algorithms.sac.sac.SACConfig(), make_model=build_sac_model, postprocess_fn=postprocess_trajectory, action_distribution_fn=get_distribution_inputs_and_class, loss_fn=sac_actor_critic_loss, stats_fn=stats, compute_gradients_fn=compute_and_clip_gradients, apply_gradients_fn=apply_gradients, extra_learn_fetches_fn=lambda policy: {'td_error': policy.td_error}, mixins=[TargetNetworkMixin, ActorCriticOptimizerMixin, ComputeTDErrorMixin], validate_spaces=validate_spaces, before_init=setup_early_mixins, before_loss_init=setup_mid_mixins, after_init=setup_late_mixins)