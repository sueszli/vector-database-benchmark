"""Functions to compute V-trace off-policy actor critic targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.

In addition to the original paper's code, changes have been made
to support MultiDiscrete action spaces. behaviour_policy_logits,
target_policy_logits and actions parameters in the entry point
multi_from_logits method accepts lists of tensors instead of just
tensors.
"""
import collections
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.utils.framework import try_import_tf
(tf1, tf, tfv) = try_import_tf()
VTraceFromLogitsReturns = collections.namedtuple('VTraceFromLogitsReturns', ['vs', 'pg_advantages', 'log_rhos', 'behaviour_action_log_probs', 'target_action_log_probs'])
VTraceReturns = collections.namedtuple('VTraceReturns', 'vs pg_advantages')

def log_probs_from_logits_and_actions(policy_logits, actions, dist_class=Categorical, model=None):
    if False:
        i = 10
        return i + 15
    return multi_log_probs_from_logits_and_actions([policy_logits], [actions], dist_class, model)[0]

def multi_log_probs_from_logits_and_actions(policy_logits, actions, dist_class, model):
    if False:
        i = 10
        return i + 15
    'Computes action log-probs from policy logits and actions.\n\n    In the notation used throughout documentation and comments, T refers to the\n    time dimension ranging from 0 to T-1. B refers to the batch size and\n    ACTION_SPACE refers to the list of numbers each representing a number of\n    actions.\n\n    Args:\n        policy_logits: A list with length of ACTION_SPACE of float32\n            tensors of shapes [T, B, ACTION_SPACE[0]], ...,\n            [T, B, ACTION_SPACE[-1]] with un-normalized log-probabilities\n            parameterizing a softmax policy.\n        actions: A list with length of ACTION_SPACE of tensors of shapes\n            [T, B, ...], ..., [T, B, ...]\n            with actions.\n        dist_class: Python class of the action distribution.\n\n    Returns:\n        A list with length of ACTION_SPACE of float32 tensors of shapes\n            [T, B], ..., [T, B] corresponding to the sampling log probability\n            of the chosen action w.r.t. the policy.\n    '
    log_probs = []
    for i in range(len(policy_logits)):
        p_shape = tf.shape(policy_logits[i])
        a_shape = tf.shape(actions[i])
        policy_logits_flat = tf.reshape(policy_logits[i], tf.concat([[-1], p_shape[2:]], axis=0))
        actions_flat = tf.reshape(actions[i], tf.concat([[-1], a_shape[2:]], axis=0))
        log_probs.append(tf.reshape(dist_class(policy_logits_flat, model).logp(actions_flat), a_shape[:2]))
    return log_probs

def from_logits(behaviour_policy_logits, target_policy_logits, actions, discounts, rewards, values, bootstrap_value, dist_class=Categorical, model=None, clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0, name='vtrace_from_logits'):
    if False:
        return 10
    'multi_from_logits wrapper used only for tests'
    res = multi_from_logits([behaviour_policy_logits], [target_policy_logits], [actions], discounts, rewards, values, bootstrap_value, dist_class, model, clip_rho_threshold=clip_rho_threshold, clip_pg_rho_threshold=clip_pg_rho_threshold, name=name)
    return VTraceFromLogitsReturns(vs=res.vs, pg_advantages=res.pg_advantages, log_rhos=res.log_rhos, behaviour_action_log_probs=tf.squeeze(res.behaviour_action_log_probs, axis=0), target_action_log_probs=tf.squeeze(res.target_action_log_probs, axis=0))

def multi_from_logits(behaviour_policy_logits, target_policy_logits, actions, discounts, rewards, values, bootstrap_value, dist_class, model, behaviour_action_log_probs=None, clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0, name='vtrace_from_logits'):
    if False:
        return 10
    'V-trace for softmax policies.\n\n    Calculates V-trace actor critic targets for softmax polices as described in\n\n    "IMPALA: Scalable Distributed Deep-RL with\n    Importance Weighted Actor-Learner Architectures"\n    by Espeholt, Soyer, Munos et al.\n\n    Target policy refers to the policy we are interested in improving and\n    behaviour policy refers to the policy that generated the given\n    rewards and actions.\n\n    In the notation used throughout documentation and comments, T refers to the\n    time dimension ranging from 0 to T-1. B refers to the batch size and\n    ACTION_SPACE refers to the list of numbers each representing a number of\n    actions.\n\n    Args:\n      behaviour_policy_logits: A list with length of ACTION_SPACE of float32\n        tensors of shapes\n        [T, B, ACTION_SPACE[0]],\n        ...,\n        [T, B, ACTION_SPACE[-1]]\n        with un-normalized log-probabilities parameterizing the softmax behaviour\n        policy.\n      target_policy_logits: A list with length of ACTION_SPACE of float32\n        tensors of shapes\n        [T, B, ACTION_SPACE[0]],\n        ...,\n        [T, B, ACTION_SPACE[-1]]\n        with un-normalized log-probabilities parameterizing the softmax target\n        policy.\n      actions: A list with length of ACTION_SPACE of\n        tensors of shapes\n        [T, B, ...],\n        ...,\n        [T, B, ...]\n        with actions sampled from the behaviour policy.\n      discounts: A float32 tensor of shape [T, B] with the discount encountered\n        when following the behaviour policy.\n      rewards: A float32 tensor of shape [T, B] with the rewards generated by\n        following the behaviour policy.\n      values: A float32 tensor of shape [T, B] with the value function estimates\n        wrt. the target policy.\n      bootstrap_value: A float32 of shape [B] with the value function estimate at\n        time T.\n      dist_class: action distribution class for the logits.\n      model: backing ModelV2 instance\n      behaviour_action_log_probs: precalculated values of the behaviour actions\n      clip_rho_threshold: A scalar float32 tensor with the clipping threshold for\n        importance weights (rho) when calculating the baseline targets (vs).\n        rho^bar in the paper.\n      clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold\n        on rho_s in \\rho_s \\delta log \\pi(a|x) (r + \\gamma v_{s+1} - V(x_s)).\n      name: The name scope that all V-trace operations will be created in.\n\n    Returns:\n      A `VTraceFromLogitsReturns` namedtuple with the following fields:\n        vs: A float32 tensor of shape [T, B]. Can be used as target to train a\n            baseline (V(x_t) - vs_t)^2.\n        pg_advantages: A float 32 tensor of shape [T, B]. Can be used as an\n          estimate of the advantage in the calculation of policy gradients.\n        log_rhos: A float32 tensor of shape [T, B] containing the log importance\n          sampling weights (log rhos).\n        behaviour_action_log_probs: A float32 tensor of shape [T, B] containing\n          behaviour policy action log probabilities (log \\mu(a_t)).\n        target_action_log_probs: A float32 tensor of shape [T, B] containing\n          target policy action probabilities (log \\pi(a_t)).\n    '
    for i in range(len(behaviour_policy_logits)):
        behaviour_policy_logits[i] = tf.convert_to_tensor(behaviour_policy_logits[i], dtype=tf.float32)
        target_policy_logits[i] = tf.convert_to_tensor(target_policy_logits[i], dtype=tf.float32)
        behaviour_policy_logits[i].shape.assert_has_rank(3)
        target_policy_logits[i].shape.assert_has_rank(3)
    with tf1.name_scope(name, values=[behaviour_policy_logits, target_policy_logits, actions, discounts, rewards, values, bootstrap_value]):
        target_action_log_probs = multi_log_probs_from_logits_and_actions(target_policy_logits, actions, dist_class, model)
        if len(behaviour_policy_logits) > 1 or behaviour_action_log_probs is None:
            behaviour_action_log_probs = multi_log_probs_from_logits_and_actions(behaviour_policy_logits, actions, dist_class, model)
        log_rhos = get_log_rhos(target_action_log_probs, behaviour_action_log_probs)
        vtrace_returns = from_importance_weights(log_rhos=log_rhos, discounts=discounts, rewards=rewards, values=values, bootstrap_value=bootstrap_value, clip_rho_threshold=clip_rho_threshold, clip_pg_rho_threshold=clip_pg_rho_threshold)
        return VTraceFromLogitsReturns(log_rhos=log_rhos, behaviour_action_log_probs=behaviour_action_log_probs, target_action_log_probs=target_action_log_probs, **vtrace_returns._asdict())

def from_importance_weights(log_rhos, discounts, rewards, values, bootstrap_value, clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0, name='vtrace_from_importance_weights'):
    if False:
        print('Hello World!')
    'V-trace from log importance weights.\n\n    Calculates V-trace actor critic targets as described in\n\n    "IMPALA: Scalable Distributed Deep-RL with\n    Importance Weighted Actor-Learner Architectures"\n    by Espeholt, Soyer, Munos et al.\n\n    In the notation used throughout documentation and comments, T refers to the\n    time dimension ranging from 0 to T-1. B refers to the batch size. This code\n    also supports the case where all tensors have the same number of additional\n    dimensions, e.g., `rewards` is [T, B, C], `values` is [T, B, C],\n    `bootstrap_value` is [B, C].\n\n    Args:\n      log_rhos: A float32 tensor of shape [T, B] representing the\n        log importance sampling weights, i.e.\n        log(target_policy(a) / behaviour_policy(a)). V-trace performs operations\n        on rhos in log-space for numerical stability.\n      discounts: A float32 tensor of shape [T, B] with discounts encountered when\n        following the behaviour policy.\n      rewards: A float32 tensor of shape [T, B] containing rewards generated by\n        following the behaviour policy.\n      values: A float32 tensor of shape [T, B] with the value function estimates\n        wrt. the target policy.\n      bootstrap_value: A float32 of shape [B] with the value function estimate at\n        time T.\n      clip_rho_threshold: A scalar float32 tensor with the clipping threshold for\n        importance weights (rho) when calculating the baseline targets (vs).\n        rho^bar in the paper. If None, no clipping is applied.\n      clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold\n        on rho_s in \\rho_s \\delta log \\pi(a|x) (r + \\gamma v_{s+1} - V(x_s)). If\n        None, no clipping is applied.\n      name: The name scope that all V-trace operations will be created in.\n\n    Returns:\n      A VTraceReturns namedtuple (vs, pg_advantages) where:\n        vs: A float32 tensor of shape [T, B]. Can be used as target to\n          train a baseline (V(x_t) - vs_t)^2.\n        pg_advantages: A float32 tensor of shape [T, B]. Can be used as the\n          advantage in the calculation of policy gradients.\n    '
    log_rhos = tf.convert_to_tensor(log_rhos, dtype=tf.float32)
    discounts = tf.convert_to_tensor(discounts, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    values = tf.convert_to_tensor(values, dtype=tf.float32)
    bootstrap_value = tf.convert_to_tensor(bootstrap_value, dtype=tf.float32)
    if clip_rho_threshold is not None:
        clip_rho_threshold = tf.convert_to_tensor(clip_rho_threshold, dtype=tf.float32)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold = tf.convert_to_tensor(clip_pg_rho_threshold, dtype=tf.float32)
    rho_rank = log_rhos.shape.ndims
    values.shape.assert_has_rank(rho_rank)
    bootstrap_value.shape.assert_has_rank(rho_rank - 1)
    discounts.shape.assert_has_rank(rho_rank)
    rewards.shape.assert_has_rank(rho_rank)
    if clip_rho_threshold is not None:
        clip_rho_threshold.shape.assert_has_rank(0)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold.shape.assert_has_rank(0)
    with tf1.name_scope(name, values=[log_rhos, discounts, rewards, values, bootstrap_value]):
        rhos = tf.math.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = tf.minimum(clip_rho_threshold, rhos, name='clipped_rhos')
        else:
            clipped_rhos = rhos
        cs = tf.minimum(1.0, rhos, name='cs')
        values_t_plus_1 = tf.concat([values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)
        sequences = (tf.reverse(discounts, axis=[0]), tf.reverse(cs, axis=[0]), tf.reverse(deltas, axis=[0]))

        def scanfunc(acc, sequence_item):
            if False:
                while True:
                    i = 10
            (discount_t, c_t, delta_t) = sequence_item
            return delta_t + discount_t * c_t * acc
        initial_values = tf.zeros_like(bootstrap_value)
        vs_minus_v_xs = tf.nest.map_structure(tf.stop_gradient, tf.scan(fn=scanfunc, elems=sequences, initializer=initial_values, parallel_iterations=1, name='scan'))
        vs_minus_v_xs = tf.reverse(vs_minus_v_xs, [0], name='vs_minus_v_xs')
        vs = tf.add(vs_minus_v_xs, values, name='vs')
        vs_t_plus_1 = tf.concat([vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = tf.minimum(clip_pg_rho_threshold, rhos, name='clipped_pg_rhos')
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)
        return VTraceReturns(vs=tf.stop_gradient(vs), pg_advantages=tf.stop_gradient(pg_advantages))

def get_log_rhos(target_action_log_probs, behaviour_action_log_probs):
    if False:
        i = 10
        return i + 15
    'With the selected log_probs for multi-discrete actions of behaviour\n    and target policies we compute the log_rhos for calculating the vtrace.'
    t = tf.stack(target_action_log_probs)
    b = tf.stack(behaviour_action_log_probs)
    log_rhos = tf.reduce_sum(t - b, axis=0)
    return log_rhos