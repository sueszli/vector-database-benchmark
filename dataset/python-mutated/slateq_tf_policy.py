"""TensorFlow policy class used for SlateQ."""
import functools
import logging
from typing import Dict
import gymnasium as gym
import numpy as np
from rllib_slate_q.slate_q.slateq_tf_model import SlateQTFModel
from ray.rllib.algorithms.dqn.dqn_tf_policy import clip_gradients
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import SlateMultiCategorical
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import LearningRateSchedule, TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import huber_loss
from ray.rllib.utils.typing import AlgorithmConfigDict, TensorType
(tf1, tf, tfv) = try_import_tf()
logger = logging.getLogger(__name__)

def build_slateq_model(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> SlateQTFModel:
    if False:
        print('Hello World!')
    "Build models for the SlateQTFPolicy.\n\n    Args:\n        policy: The policy, which will use the model for optimization.\n        obs_space: The policy's observation space.\n        action_space: The policy's action space.\n        config: The Algorithm's config dict.\n\n    Returns:\n        The slate-Q specific Q-model instance.\n    "
    model = SlateQTFModel(obs_space, action_space, num_outputs=action_space.nvec[0], model_config=config['model'], name='slateq_model', fcnet_hiddens_per_candidate=config['fcnet_hiddens_per_candidate'])
    policy.target_model = SlateQTFModel(obs_space, action_space, num_outputs=action_space.nvec[0], model_config=config['model'], name='target_slateq_model', fcnet_hiddens_per_candidate=config['fcnet_hiddens_per_candidate'])
    return model

def build_slateq_losses(policy: Policy, model: ModelV2, _, train_batch: SampleBatch) -> TensorType:
    if False:
        while True:
            i = 10
    'Constructs the choice- and Q-value losses for the SlateQTorchPolicy.\n\n    Args:\n        policy: The Policy to calculate the loss for.\n        model: The Model to calculate the loss for.\n        train_batch: The training data.\n\n    Returns:\n        The Q-value loss tensor.\n    '
    observation = train_batch[SampleBatch.OBS]
    user_obs = observation['user']
    batch_size = tf.shape(user_obs)[0]
    doc_obs = list(observation['doc'].values())
    actions = train_batch[SampleBatch.ACTIONS]
    click_indicator = tf.cast(tf.stack([k['click'] for k in observation['response']], 1), tf.float32)
    item_reward = tf.stack([k['watch_time'] for k in observation['response']], 1)
    q_values = model.get_q_values(user_obs, doc_obs)
    slate_q_values = tf.gather(q_values, tf.cast(actions, dtype=tf.int32), batch_dims=-1)
    replay_click_q = tf.reduce_sum(input_tensor=slate_q_values * click_indicator, axis=1, name='replay_click_q')
    next_obs = train_batch[SampleBatch.NEXT_OBS]
    user_next_obs = next_obs['user']
    doc_next_obs = list(next_obs['doc'].values())
    reward = tf.reduce_sum(input_tensor=item_reward * click_indicator, axis=1)
    next_q_values = policy.target_model.get_q_values(user_obs, doc_obs)
    (scores, score_no_click) = score_documents(user_next_obs, doc_next_obs)
    next_q_values_slate = tf.gather(next_q_values, policy.slates, axis=1)
    scores_slate = tf.gather(scores, policy.slates, axis=1)
    score_no_click_slate = tf.reshape(tf.tile(score_no_click, tf.shape(input=policy.slates)[:1]), [batch_size, -1])
    next_q_target_slate = tf.reduce_sum(input_tensor=next_q_values_slate * scores_slate, axis=2) / (tf.reduce_sum(input_tensor=scores_slate, axis=2) + score_no_click_slate)
    next_q_target_max = tf.reduce_max(input_tensor=next_q_target_slate, axis=1)
    target = reward + policy.config['gamma'] * next_q_target_max * (1.0 - tf.cast(train_batch[SampleBatch.TERMINATEDS], tf.float32))
    target = tf.stop_gradient(target)
    clicked = tf.reduce_sum(input_tensor=click_indicator, axis=1)
    clicked_indices = tf.squeeze(tf.where(tf.equal(clicked, 1)), axis=1)
    q_clicked = tf.gather(replay_click_q, clicked_indices)
    target_clicked = tf.gather(target, clicked_indices)
    td_error = tf.where(tf.cast(clicked, tf.bool), replay_click_q - target, tf.zeros_like(train_batch[SampleBatch.REWARDS]))
    if policy.config['use_huber']:
        loss = huber_loss(td_error, delta=policy.config['huber_threshold'])
    else:
        loss = tf.math.square(td_error)
    loss = tf.reduce_mean(loss)
    td_error = tf.abs(td_error)
    mean_td_error = tf.reduce_mean(td_error)
    policy._q_values = tf.reduce_mean(q_values)
    policy._q_clicked = tf.reduce_mean(q_clicked)
    policy._scores = tf.reduce_mean(scores)
    policy._score_no_click = tf.reduce_mean(score_no_click)
    policy._slate_q_values = tf.reduce_mean(slate_q_values)
    policy._replay_click_q = tf.reduce_mean(replay_click_q)
    policy._bellman_reward = tf.reduce_mean(reward)
    policy._next_q_values = tf.reduce_mean(next_q_values)
    policy._target = tf.reduce_mean(target)
    policy._next_q_target_slate = tf.reduce_mean(next_q_target_slate)
    policy._next_q_target_max = tf.reduce_mean(next_q_target_max)
    policy._target_clicked = tf.reduce_mean(target_clicked)
    policy._q_loss = loss
    policy._td_error = td_error
    policy._mean_td_error = mean_td_error
    policy._mean_actions = tf.reduce_mean(actions)
    return loss

def build_slateq_stats(policy: Policy, batch) -> Dict[str, TensorType]:
    if False:
        print('Hello World!')
    stats = {'q_values': policy._q_values, 'q_clicked': policy._q_clicked, 'scores': policy._scores, 'score_no_click': policy._score_no_click, 'slate_q_values': policy._slate_q_values, 'replay_click_q': policy._replay_click_q, 'bellman_reward': policy._bellman_reward, 'next_q_values': policy._next_q_values, 'target': policy._target, 'next_q_target_slate': policy._next_q_target_slate, 'next_q_target_max': policy._next_q_target_max, 'target_clicked': policy._target_clicked, 'mean_td_error': policy._mean_td_error, 'q_loss': policy._q_loss, 'mean_actions': policy._mean_actions}
    return stats

def action_distribution_fn(policy: Policy, model: SlateQTFModel, input_dict, *, explore, is_training, **kwargs):
    if False:
        i = 10
        return i + 15
    'Determine which action to take.'
    observation = input_dict[SampleBatch.OBS]
    user_obs = observation['user']
    doc_obs = list(observation['doc'].values())
    (scores, score_no_click) = score_documents(user_obs, doc_obs)
    q_values = model.get_q_values(user_obs, doc_obs)
    with tf.name_scope('select_slate'):
        per_slate_q_values = get_per_slate_q_values(policy.slates, score_no_click, scores, q_values)
    return (per_slate_q_values, functools.partial(SlateMultiCategorical, action_space=policy.action_space, all_slates=policy.slates), [])

def get_per_slate_q_values(slates, s_no_click, s, q):
    if False:
        while True:
            i = 10
    slate_q_values = tf.gather(s * q, slates, axis=1)
    slate_scores = tf.gather(s, slates, axis=1)
    slate_normalizer = tf.reduce_sum(input_tensor=slate_scores, axis=2) + tf.expand_dims(s_no_click, 1)
    slate_q_values = slate_q_values / tf.expand_dims(slate_normalizer, 2)
    slate_sum_q_values = tf.reduce_sum(input_tensor=slate_q_values, axis=2)
    return slate_sum_q_values

def score_documents(user_obs, doc_obs, no_click_score=1.0, multinomial_logits=False, min_normalizer=-1.0):
    if False:
        for i in range(10):
            print('nop')
    'Computes dot-product scores for user vs doc (plus no-click) feature vectors.'
    scores_per_candidate = tf.reduce_sum(tf.multiply(tf.expand_dims(user_obs, 1), tf.stack(doc_obs, axis=1)), 2)
    score_no_click = tf.fill([tf.shape(user_obs)[0], 1], no_click_score)
    all_scores = tf.concat([scores_per_candidate, score_no_click], axis=1)
    if multinomial_logits:
        all_scores = tf.nn.softmax(all_scores)
    else:
        all_scores = all_scores - min_normalizer
    return (all_scores[:, :-1], all_scores[:, -1])

def setup_early(policy, obs_space, action_space, config):
    if False:
        for i in range(10):
            print('nop')
    'Obtain all possible slates given current docs in the candidate set.'
    num_candidates = action_space.nvec[0]
    slate_size = len(action_space.nvec)
    num_all_slates = np.prod([num_candidates - i for i in range(slate_size)])
    mesh_args = [list(range(num_candidates))] * slate_size
    slates = tf.stack(tf.meshgrid(*mesh_args), axis=-1)
    slates = tf.reshape(slates, shape=(-1, slate_size))
    unique_mask = tf.map_fn(lambda x: tf.equal(tf.size(input=x), tf.size(input=tf.unique(x)[0])), slates, dtype=tf.bool)
    slates = tf.boolean_mask(tensor=slates, mask=unique_mask)
    slates.set_shape([num_all_slates, slate_size])
    policy.slates = slates

def setup_mid_mixins(policy: Policy, obs_space, action_space, config) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Call mixin classes' constructors before SlateQTorchPolicy loss initialization.\n\n    Args:\n        policy: The Policy object.\n        obs_space: The Policy's observation space.\n        action_space: The Policy's action space.\n        config: The Policy's config.\n    "
    LearningRateSchedule.__init__(policy, config['lr'], config['lr_schedule'])

def setup_late_mixins(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> None:
    if False:
        while True:
            i = 10
    "Call mixin classes' constructors after SlateQTorchPolicy loss initialization.\n\n    Args:\n        policy: The Policy object.\n        obs_space: The Policy's observation space.\n        action_space: The Policy's action space.\n        config: The Policy's config.\n    "
    TargetNetworkMixin.__init__(policy)

def rmsprop_optimizer(policy: Policy, config: AlgorithmConfigDict) -> 'tf.keras.optimizers.Optimizer':
    if False:
        while True:
            i = 10
    if policy.config['framework'] == 'tf2':
        return tf.keras.optimizers.RMSprop(learning_rate=policy.cur_lr, epsilon=config['rmsprop_epsilon'], weight_decay=0.95, momentum=0.0, centered=True)
    else:
        return tf1.train.RMSPropOptimizer(learning_rate=policy.cur_lr, epsilon=config['rmsprop_epsilon'], decay=0.95, momentum=0.0, centered=True)
SlateQTFPolicy = build_tf_policy(name='SlateQTFPolicy', get_default_config=lambda : rllib_slate_q.slate_q.slateq.SlateQConfig(), make_model=build_slateq_model, loss_fn=build_slateq_losses, stats_fn=build_slateq_stats, extra_learn_fetches_fn=lambda policy: {'td_error': policy._td_error}, optimizer_fn=rmsprop_optimizer, action_distribution_fn=action_distribution_fn, compute_gradients_fn=clip_gradients, before_init=setup_early, before_loss_init=setup_mid_mixins, after_init=setup_late_mixins, mixins=[LearningRateSchedule, TargetNetworkMixin])