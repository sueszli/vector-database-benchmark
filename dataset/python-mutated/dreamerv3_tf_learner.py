"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""
from typing import Any, Dict, Mapping, Tuple
import gymnasium as gym
from ray.rllib.algorithms.dreamerv3.dreamerv3_learner import DreamerV3Learner, DreamerV3LearnerHyperparameters
from ray.rllib.core.rl_module.marl_module import ModuleID
from ray.rllib.core.learner.learner import ParamDict
from ray.rllib.core.learner.tf.tf_learner import TfLearner
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.tf_utils import symlog, two_hot, clip_gradients
from ray.rllib.utils.typing import TensorType
(_, tf, _) = try_import_tf()
tfp = try_import_tfp()

class DreamerV3TfLearner(DreamerV3Learner, TfLearner):
    """Implements DreamerV3 losses and gradient-based update logic in TensorFlow.

    The critic EMA-copy update step can be found in the `DreamerV3Learner` base class,
    as it is framework independent.

    We define 3 local TensorFlow optimizers for the sub components "world_model",
    "actor", and "critic". Each of these optimizers might use a different learning rate,
    epsilon parameter, and gradient clipping thresholds and procedures.
    """

    @override(TfLearner)
    def configure_optimizers_for_module(self, module_id: ModuleID, hps: DreamerV3LearnerHyperparameters):
        if False:
            for i in range(10):
                print('nop')
        "Create the 3 optimizers for Dreamer learning: world_model, actor, critic.\n\n        The learning rates used are described in [1] and the epsilon values used here\n        - albeit probably not that important - are used by the author's own\n        implementation.\n        "
        dreamerv3_module = self._module[module_id]
        optim_world_model = tf.keras.optimizers.Adam(epsilon=1e-08)
        optim_world_model.build(dreamerv3_module.world_model.trainable_variables)
        params_world_model = self.get_parameters(dreamerv3_module.world_model)
        self.register_optimizer(module_id=module_id, optimizer_name='world_model', optimizer=optim_world_model, params=params_world_model, lr_or_lr_schedule=hps.world_model_lr)
        optim_actor = tf.keras.optimizers.Adam(epsilon=1e-05)
        optim_actor.build(dreamerv3_module.actor.trainable_variables)
        params_actor = self.get_parameters(dreamerv3_module.actor)
        self.register_optimizer(module_id=module_id, optimizer_name='actor', optimizer=optim_actor, params=params_actor, lr_or_lr_schedule=hps.actor_lr)
        optim_critic = tf.keras.optimizers.Adam(epsilon=1e-05)
        optim_critic.build(dreamerv3_module.critic.trainable_variables)
        params_critic = self.get_parameters(dreamerv3_module.critic)
        self.register_optimizer(module_id=module_id, optimizer_name='critic', optimizer=optim_critic, params=params_critic, lr_or_lr_schedule=hps.critic_lr)

    @override(TfLearner)
    def postprocess_gradients_for_module(self, *, module_id: ModuleID, hps: DreamerV3LearnerHyperparameters, module_gradients_dict: Dict[str, Any]) -> ParamDict:
        if False:
            i = 10
            return i + 15
        "Performs gradient clipping on the 3 module components' computed grads.\n\n        Note that different grad global-norm clip values are used for the 3\n        module components: world model, actor, and critic.\n        "
        for (optimizer_name, optimizer) in self.get_optimizers_for_module(module_id=module_id):
            grads_sub_dict = self.filter_param_dict_for_optimizer(module_gradients_dict, optimizer)
            grad_clip = hps.world_model_grad_clip_by_global_norm if optimizer_name == 'world_model' else hps.actor_grad_clip_by_global_norm if optimizer_name == 'actor' else hps.critic_grad_clip_by_global_norm
            global_norm = clip_gradients(grads_sub_dict, grad_clip=grad_clip, grad_clip_by='global_norm')
            module_gradients_dict.update(grads_sub_dict)
            self.register_metric(module_id, optimizer_name.upper() + '_gradients_global_norm', global_norm)
            self.register_metric(module_id, optimizer_name.upper() + '_gradients_maxabs_after_clipping', tf.reduce_max([tf.reduce_max(tf.math.abs(g)) for g in grads_sub_dict.values()]))
        return module_gradients_dict

    @override(TfLearner)
    def compute_gradients(self, loss_per_module, gradient_tape, **kwargs):
        if False:
            return 10
        grads = {}
        for component in ['world_model', 'actor', 'critic']:
            grads.update(gradient_tape.gradient(self._metrics[DEFAULT_POLICY_ID][component.upper() + '_L_total'], self.filter_param_dict_for_optimizer(self._params, self.get_optimizer(optimizer_name=component))))
        del gradient_tape
        return grads

    @override(TfLearner)
    def compute_loss_for_module(self, module_id: ModuleID, hps: DreamerV3LearnerHyperparameters, batch: SampleBatch, fwd_out: Mapping[str, TensorType]) -> TensorType:
        if False:
            i = 10
            return i + 15
        prediction_losses = self._compute_world_model_prediction_losses(hps=hps, rewards_B_T=batch[SampleBatch.REWARDS], continues_B_T=1.0 - tf.cast(batch['is_terminated'], tf.float32), fwd_out=fwd_out)
        (L_dyn_B_T, L_rep_B_T) = self._compute_world_model_dynamics_and_representation_loss(hps=hps, fwd_out=fwd_out)
        L_dyn = tf.reduce_mean(L_dyn_B_T)
        L_rep = tf.reduce_mean(L_rep_B_T)
        tf.assert_equal(L_dyn, L_rep)
        L_world_model_total_B_T = 1.0 * prediction_losses['L_prediction_B_T'] + 0.5 * L_dyn_B_T + 0.1 * L_rep_B_T
        L_world_model_total = tf.reduce_mean(L_world_model_total_B_T)
        self.register_metrics(module_id=module_id, metrics_dict={'WORLD_MODEL_learned_initial_h': self.module[module_id].world_model.initial_h, 'WORLD_MODEL_L_decoder': prediction_losses['L_decoder'], 'WORLD_MODEL_L_reward': prediction_losses['L_reward'], 'WORLD_MODEL_L_continue': prediction_losses['L_continue'], 'WORLD_MODEL_L_prediction': prediction_losses['L_prediction'], 'WORLD_MODEL_L_dynamics': L_dyn, 'WORLD_MODEL_L_representation': L_rep, 'WORLD_MODEL_L_total': L_world_model_total})
        if hps.report_individual_batch_item_stats:
            self.register_metrics(module_id=module_id, metrics_dict={'WORLD_MODEL_L_decoder_B_T': prediction_losses['L_decoder_B_T'], 'WORLD_MODEL_L_reward_B_T': prediction_losses['L_reward_B_T'], 'WORLD_MODEL_L_continue_B_T': prediction_losses['L_continue_B_T'], 'WORLD_MODEL_L_prediction_B_T': prediction_losses['L_prediction_B_T'], 'WORLD_MODEL_L_dynamics_B_T': L_dyn_B_T, 'WORLD_MODEL_L_representation_B_T': L_rep_B_T, 'WORLD_MODEL_L_total_B_T': L_world_model_total_B_T})
        dream_data = self.module[module_id].dreamer_model.dream_trajectory(start_states={'h': fwd_out['h_states_BxT'], 'z': fwd_out['z_posterior_states_BxT']}, start_is_terminated=tf.reshape(batch['is_terminated'], [-1]))
        if hps.report_dream_data:
            self.register_metrics(module_id, {'DREAM_DATA_' + key[:-1] + '1': value[:, hps.batch_size_B] for (key, value) in dream_data.items() if key.endswith('H_BxT')})
        value_targets_t0_to_Hm1_BxT = self._compute_value_targets(hps=hps, rewards_t0_to_H_BxT=dream_data['rewards_dreamed_t0_to_H_BxT'], intrinsic_rewards_t1_to_H_BxT=dream_data['rewards_intrinsic_t1_to_H_B'] if hps.use_curiosity else None, continues_t0_to_H_BxT=dream_data['continues_dreamed_t0_to_H_BxT'], value_predictions_t0_to_H_BxT=dream_data['values_dreamed_t0_to_H_BxT'])
        self.register_metric(module_id, 'VALUE_TARGETS_H_BxT', value_targets_t0_to_Hm1_BxT)
        CRITIC_L_total = self._compute_critic_loss(module_id=module_id, hps=hps, dream_data=dream_data, value_targets_t0_to_Hm1_BxT=value_targets_t0_to_Hm1_BxT)
        if hps.train_actor:
            ACTOR_L_total = self._compute_actor_loss(module_id=module_id, hps=hps, dream_data=dream_data, value_targets_t0_to_Hm1_BxT=value_targets_t0_to_Hm1_BxT)
        else:
            ACTOR_L_total = 0.0
        return L_world_model_total + CRITIC_L_total + ACTOR_L_total

    def _compute_world_model_prediction_losses(self, *, hps: DreamerV3LearnerHyperparameters, rewards_B_T: TensorType, continues_B_T: TensorType, fwd_out: Mapping[str, TensorType]) -> Mapping[str, TensorType]:
        if False:
            for i in range(10):
                print('nop')
        'Helper method computing all world-model related prediction losses.\n\n        Prediction losses are used to train the predictors of the world model, which\n        are: Reward predictor, continue predictor, and the decoder (which predicts\n        observations).\n\n        Args:\n            hps: The DreamerV3LearnerHyperparameters to use.\n            rewards_B_T: The rewards batch in the shape (B, T) and of type float32.\n            continues_B_T: The continues batch in the shape (B, T) and of type float32\n                (1.0 -> continue; 0.0 -> end of episode).\n            fwd_out: The `forward_train` outputs of the DreamerV3RLModule.\n        '
        obs_BxT = fwd_out['sampled_obs_symlog_BxT']
        obs_distr_means = fwd_out['obs_distribution_means_BxT']
        obs_BxT = tf.reshape(obs_BxT, shape=[-1, tf.reduce_prod(obs_BxT.shape[1:])])
        decoder_loss_BxT = tf.reduce_sum(tf.math.square(obs_distr_means - obs_BxT), axis=-1)
        decoder_loss_B_T = tf.reshape(decoder_loss_BxT, (hps.batch_size_B, hps.batch_length_T))
        L_decoder = tf.reduce_mean(decoder_loss_B_T)
        reward_logits_BxT = fwd_out['reward_logits_BxT']
        rewards_symlog_B_T = symlog(tf.cast(rewards_B_T, tf.float32))
        rewards_symlog_BxT = tf.reshape(rewards_symlog_B_T, shape=[-1])
        two_hot_rewards_symlog_BxT = two_hot(rewards_symlog_BxT)
        reward_log_pred_BxT = reward_logits_BxT - tf.math.reduce_logsumexp(reward_logits_BxT, axis=-1, keepdims=True)
        reward_loss_two_hot_BxT = -tf.reduce_sum(reward_log_pred_BxT * two_hot_rewards_symlog_BxT, axis=-1)
        reward_loss_two_hot_B_T = tf.reshape(reward_loss_two_hot_BxT, (hps.batch_size_B, hps.batch_length_T))
        L_reward_two_hot = tf.reduce_mean(reward_loss_two_hot_B_T)
        continue_distr = fwd_out['continue_distribution_BxT']
        continues_BxT = tf.reshape(continues_B_T, shape=[-1])
        continue_loss_BxT = -continue_distr.log_prob(continues_BxT)
        continue_loss_B_T = tf.reshape(continue_loss_BxT, (hps.batch_size_B, hps.batch_length_T))
        L_continue = tf.reduce_mean(continue_loss_B_T)
        L_pred_B_T = decoder_loss_B_T + reward_loss_two_hot_B_T + continue_loss_B_T
        L_pred = tf.reduce_mean(L_pred_B_T)
        return {'L_decoder_B_T': decoder_loss_B_T, 'L_decoder': L_decoder, 'L_reward': L_reward_two_hot, 'L_reward_B_T': reward_loss_two_hot_B_T, 'L_continue': L_continue, 'L_continue_B_T': continue_loss_B_T, 'L_prediction': L_pred, 'L_prediction_B_T': L_pred_B_T}

    def _compute_world_model_dynamics_and_representation_loss(self, *, hps: DreamerV3LearnerHyperparameters, fwd_out: Dict[str, Any]) -> Tuple[TensorType, TensorType]:
        if False:
            for i in range(10):
                print('nop')
        "Helper method computing the world-model's dynamics and representation losses.\n\n        Args:\n            hps: The DreamerV3LearnerHyperparameters to use.\n            fwd_out: The `forward_train` outputs of the DreamerV3RLModule.\n\n        Returns:\n            Tuple consisting of a) dynamics loss: Trains the prior network, predicting\n            z^ prior states from h-states and b) representation loss: Trains posterior\n            network, predicting z posterior states from h-states and (encoded)\n            observations.\n        "
        z_posterior_probs_BxT = fwd_out['z_posterior_probs_BxT']
        z_posterior_distr_BxT = tfp.distributions.Independent(tfp.distributions.OneHotCategorical(probs=z_posterior_probs_BxT), reinterpreted_batch_ndims=1)
        z_prior_probs_BxT = fwd_out['z_prior_probs_BxT']
        z_prior_distr_BxT = tfp.distributions.Independent(tfp.distributions.OneHotCategorical(probs=z_prior_probs_BxT), reinterpreted_batch_ndims=1)
        sg_z_posterior_distr_BxT = tfp.distributions.Independent(tfp.distributions.OneHotCategorical(probs=tf.stop_gradient(z_posterior_probs_BxT)), reinterpreted_batch_ndims=1)
        sg_z_prior_distr_BxT = tfp.distributions.Independent(tfp.distributions.OneHotCategorical(probs=tf.stop_gradient(z_prior_probs_BxT)), reinterpreted_batch_ndims=1)
        L_dyn_BxT = tf.math.maximum(1.0, tfp.distributions.kl_divergence(sg_z_posterior_distr_BxT, z_prior_distr_BxT))
        L_dyn_B_T = tf.reshape(L_dyn_BxT, (hps.batch_size_B, hps.batch_length_T))
        L_rep_BxT = tf.math.maximum(1.0, tfp.distributions.kl_divergence(z_posterior_distr_BxT, sg_z_prior_distr_BxT))
        L_rep_B_T = tf.reshape(L_rep_BxT, (hps.batch_size_B, hps.batch_length_T))
        return (L_dyn_B_T, L_rep_B_T)

    def _compute_actor_loss(self, *, module_id: ModuleID, hps: DreamerV3LearnerHyperparameters, dream_data: Dict[str, TensorType], value_targets_t0_to_Hm1_BxT: TensorType) -> TensorType:
        if False:
            print('Hello World!')
        "Helper method computing the actor's loss terms.\n\n        Args:\n            module_id: The module_id for which to compute the actor loss.\n            hps: The DreamerV3LearnerHyperparameters to use.\n            dream_data: The data generated by dreaming for H steps (horizon) starting\n                from any BxT state (sampled from the buffer for the train batch).\n            value_targets_t0_to_Hm1_BxT: The computed value function targets of the\n                shape (t0 to H-1, BxT).\n\n        Returns:\n            The total actor loss tensor.\n        "
        actor = self.module[module_id].actor
        scaled_value_targets_t0_to_Hm1_B = self._compute_scaled_value_targets(module_id=module_id, hps=hps, value_targets_t0_to_Hm1_BxT=value_targets_t0_to_Hm1_BxT, value_predictions_t0_to_Hm1_BxT=dream_data['values_dreamed_t0_to_H_BxT'][:-1])
        actions_dreamed = tf.stop_gradient(dream_data['actions_dreamed_t0_to_H_BxT'])[:-1]
        actions_dreamed_dist_params_t0_to_Hm1_B = dream_data['actions_dreamed_dist_params_t0_to_H_BxT'][:-1]
        dist_t0_to_Hm1_B = actor.get_action_dist_object(actions_dreamed_dist_params_t0_to_Hm1_B)
        if isinstance(self.module[module_id].actor.action_space, gym.spaces.Discrete):
            logp_actions_t0_to_Hm1_B = actions_dreamed_dist_params_t0_to_Hm1_B
            logp_actions_dreamed_t0_to_Hm1_B = tf.reduce_sum(actions_dreamed * logp_actions_t0_to_Hm1_B, axis=-1)
            logp_loss_H_B = logp_actions_dreamed_t0_to_Hm1_B * tf.stop_gradient(scaled_value_targets_t0_to_Hm1_B)
        else:
            logp_actions_dreamed_t0_to_Hm1_B = dist_t0_to_Hm1_B.log_prob(actions_dreamed)
            logp_loss_H_B = scaled_value_targets_t0_to_Hm1_B
        assert len(logp_loss_H_B.shape) == 2
        entropy_H_B = dist_t0_to_Hm1_B.entropy()
        assert len(entropy_H_B.shape) == 2
        entropy = tf.reduce_mean(entropy_H_B)
        L_actor_reinforce_term_H_B = -logp_loss_H_B
        L_actor_action_entropy_term_H_B = -hps.entropy_scale * entropy_H_B
        L_actor_H_B = L_actor_reinforce_term_H_B + L_actor_action_entropy_term_H_B
        L_actor_H_B *= tf.stop_gradient(dream_data['dream_loss_weights_t0_to_H_BxT'])[:-1]
        L_actor = tf.reduce_mean(L_actor_H_B)
        self.register_metrics(module_id, metrics_dict={'ACTOR_L_total': L_actor, 'ACTOR_value_targets_pct95_ema': actor.ema_value_target_pct95, 'ACTOR_value_targets_pct5_ema': actor.ema_value_target_pct5, 'ACTOR_action_entropy': entropy, 'ACTOR_L_neglogp_reinforce_term': tf.reduce_mean(L_actor_reinforce_term_H_B), 'ACTOR_L_neg_entropy_term': tf.reduce_mean(L_actor_action_entropy_term_H_B)})
        if hps.report_individual_batch_item_stats:
            self.register_metrics(module_id, metrics_dict={'ACTOR_L_total_H_BxT': L_actor_H_B, 'ACTOR_logp_actions_dreamed_H_BxT': logp_actions_dreamed_t0_to_Hm1_B, 'ACTOR_scaled_value_targets_H_BxT': scaled_value_targets_t0_to_Hm1_B, 'ACTOR_action_entropy_H_BxT': entropy_H_B, 'ACTOR_L_neglogp_reinforce_term_H_BxT': L_actor_reinforce_term_H_B, 'ACTOR_L_neg_entropy_term_H_BxT': L_actor_action_entropy_term_H_B})
        return L_actor

    def _compute_critic_loss(self, *, module_id: ModuleID, hps: DreamerV3LearnerHyperparameters, dream_data: Dict[str, TensorType], value_targets_t0_to_Hm1_BxT: TensorType) -> TensorType:
        if False:
            print('Hello World!')
        "Helper method computing the critic's loss terms.\n\n        Args:\n            module_id: The ModuleID for which to compute the critic loss.\n            hps: The DreamerV3LearnerHyperparameters to use.\n            dream_data: The data generated by dreaming for H steps (horizon) starting\n                from any BxT state (sampled from the buffer for the train batch).\n            value_targets_t0_to_Hm1_BxT: The computed value function targets of the\n                shape (t0 to H-1, BxT).\n\n        Returns:\n            The total critic loss tensor.\n        "
        (H, B) = dream_data['rewards_dreamed_t0_to_H_BxT'].shape[:2]
        Hm1 = H - 1
        value_targets_t0_to_Hm1_B = tf.stop_gradient(value_targets_t0_to_Hm1_BxT)
        value_symlog_targets_t0_to_Hm1_B = symlog(value_targets_t0_to_Hm1_B)
        value_symlog_targets_HxB = tf.reshape(value_symlog_targets_t0_to_Hm1_B, (-1,))
        value_symlog_targets_two_hot_HxB = two_hot(value_symlog_targets_HxB)
        value_symlog_targets_two_hot_t0_to_Hm1_B = tf.reshape(value_symlog_targets_two_hot_HxB, shape=[Hm1, B, value_symlog_targets_two_hot_HxB.shape[-1]])
        value_symlog_logits_HxB = dream_data['values_symlog_dreamed_logits_t0_to_HxBxT']
        value_symlog_logits_t0_to_Hm1_B = tf.reshape(value_symlog_logits_HxB, shape=[H, B, value_symlog_logits_HxB.shape[-1]])[:-1]
        values_log_pred_Hm1_B = value_symlog_logits_t0_to_Hm1_B - tf.math.reduce_logsumexp(value_symlog_logits_t0_to_Hm1_B, axis=-1, keepdims=True)
        value_loss_two_hot_H_B = -tf.reduce_sum(values_log_pred_Hm1_B * value_symlog_targets_two_hot_t0_to_Hm1_B, axis=-1)
        value_symlog_ema_t0_to_Hm1_B = tf.stop_gradient(dream_data['v_symlog_dreamed_ema_t0_to_H_BxT'])[:-1]
        value_symlog_ema_HxB = tf.reshape(value_symlog_ema_t0_to_Hm1_B, (-1,))
        value_symlog_ema_two_hot_HxB = two_hot(value_symlog_ema_HxB)
        value_symlog_ema_two_hot_t0_to_Hm1_B = tf.reshape(value_symlog_ema_two_hot_HxB, shape=[Hm1, B, value_symlog_ema_two_hot_HxB.shape[-1]])
        ema_regularization_loss_H_B = -tf.reduce_sum(values_log_pred_Hm1_B * value_symlog_ema_two_hot_t0_to_Hm1_B, axis=-1)
        L_critic_H_B = value_loss_two_hot_H_B + ema_regularization_loss_H_B
        L_critic_H_B *= tf.stop_gradient(dream_data['dream_loss_weights_t0_to_H_BxT'])[:-1]
        L_critic = tf.reduce_mean(L_critic_H_B)
        self.register_metrics(module_id=module_id, metrics_dict={'CRITIC_L_total': L_critic, 'CRITIC_L_neg_logp_of_value_targets': tf.reduce_mean(value_loss_two_hot_H_B), 'CRITIC_L_slow_critic_regularization': tf.reduce_mean(ema_regularization_loss_H_B)})
        if hps.report_individual_batch_item_stats:
            self.register_metrics(module_id=module_id, metrics_dict={'VALUE_TARGETS_symlog_H_BxT': value_symlog_targets_t0_to_Hm1_B, 'CRITIC_L_total_H_BxT': L_critic_H_B, 'CRITIC_L_neg_logp_of_value_targets_H_BxT': value_loss_two_hot_H_B, 'CRITIC_L_slow_critic_regularization_H_BxT': ema_regularization_loss_H_B})
        return L_critic

    def _compute_value_targets(self, *, hps: DreamerV3LearnerHyperparameters, rewards_t0_to_H_BxT: TensorType, intrinsic_rewards_t1_to_H_BxT: TensorType, continues_t0_to_H_BxT: TensorType, value_predictions_t0_to_H_BxT: TensorType) -> TensorType:
        if False:
            while True:
                i = 10
        "Helper method computing the value targets.\n\n        All args are (H, BxT, ...) and in non-symlog'd (real) reward space.\n        Non-symlog is important b/c log(a+b) != log(a) + log(b).\n        See [1] eq. 8 and 10.\n        Thus, targets are always returned in real (non-symlog'd space).\n        They need to be re-symlog'd before computing the critic loss from them (b/c the\n        critic produces predictions in symlog space).\n        Note that the original B and T ranks together form the new batch dimension\n        (folded into BxT) and the new time rank is the dream horizon (hence: [H, BxT]).\n\n        Variable names nomenclature:\n        `H`=1+horizon_H (start state + H steps dreamed),\n        `BxT`=batch_size * batch_length (meaning the original trajectory time rank has\n        been folded).\n\n        Rewards, continues, and value predictions are all of shape [t0-H, BxT]\n        (time-major), whereas returned targets are [t0 to H-1, B] (last timestep missing\n        b/c the target value equals vf prediction in that location anyways.\n\n        Args:\n            hps: The DreamerV3LearnerHyperparameters to use.\n            rewards_t0_to_H_BxT: The reward predictor's predictions over the\n                dreamed trajectory t0 to H (and for the batch BxT).\n            intrinsic_rewards_t1_to_H_BxT: The predicted intrinsic rewards over the\n                dreamed trajectory t0 to H (and for the batch BxT).\n            continues_t0_to_H_BxT: The continue predictor's predictions over the\n                dreamed trajectory t0 to H (and for the batch BxT).\n            value_predictions_t0_to_H_BxT: The critic's value predictions over the\n                dreamed trajectory t0 to H (and for the batch BxT).\n\n        Returns:\n            The value targets in the shape: [t0toH-1, BxT]. Note that the last step (H)\n            does not require a value target as it matches the critic's value prediction\n            anyways.\n        "
        rewards_t1_to_H_BxT = rewards_t0_to_H_BxT[1:]
        if intrinsic_rewards_t1_to_H_BxT is not None:
            rewards_t1_to_H_BxT += intrinsic_rewards_t1_to_H_BxT
        discount = continues_t0_to_H_BxT[1:] * hps.gamma
        Rs = [value_predictions_t0_to_H_BxT[-1]]
        intermediates = rewards_t1_to_H_BxT + discount * (1 - hps.gae_lambda) * value_predictions_t0_to_H_BxT[1:]
        for t in reversed(range(discount.shape[0])):
            Rs.append(intermediates[t] + discount[t] * hps.gae_lambda * Rs[-1])
        targets_t0toHm1_BxT = tf.stack(list(reversed(Rs))[:-1], axis=0)
        return targets_t0toHm1_BxT

    def _compute_scaled_value_targets(self, *, module_id: ModuleID, hps: DreamerV3LearnerHyperparameters, value_targets_t0_to_Hm1_BxT: TensorType, value_predictions_t0_to_Hm1_BxT: TensorType) -> TensorType:
        if False:
            print('Hello World!')
        "Helper method computing the scaled value targets.\n\n        Args:\n            module_id: The module_id to compute value targets for.\n            hps: The DreamerV3LearnerHyperparameters to use.\n            value_targets_t0_to_Hm1_BxT: The value targets computed by\n                `self._compute_value_targets` in the shape of (t0 to H-1, BxT)\n                and of type float32.\n            value_predictions_t0_to_Hm1_BxT: The critic's value predictions over the\n                dreamed trajectories (w/o the last timestep). The shape of this\n                tensor is (t0 to H-1, BxT) and the type is float32.\n\n        Returns:\n            The scaled value targets used by the actor for REINFORCE policy updates\n            (using scaled advantages). See [1] eq. 12 for more details.\n        "
        actor = self.module[module_id].actor
        value_targets_H_B = value_targets_t0_to_Hm1_BxT
        value_predictions_H_B = value_predictions_t0_to_Hm1_BxT
        Per_R_5 = tfp.stats.percentile(value_targets_H_B, 5)
        Per_R_95 = tfp.stats.percentile(value_targets_H_B, 95)
        new_val_pct5 = tf.where(tf.math.is_nan(actor.ema_value_target_pct5), Per_R_5, hps.return_normalization_decay * actor.ema_value_target_pct5 + (1.0 - hps.return_normalization_decay) * Per_R_5)
        actor.ema_value_target_pct5.assign(new_val_pct5)
        new_val_pct95 = tf.where(tf.math.is_nan(actor.ema_value_target_pct95), Per_R_95, hps.return_normalization_decay * actor.ema_value_target_pct95 + (1.0 - hps.return_normalization_decay) * Per_R_95)
        actor.ema_value_target_pct95.assign(new_val_pct95)
        offset = actor.ema_value_target_pct5
        invscale = tf.math.maximum(1e-08, actor.ema_value_target_pct95 - actor.ema_value_target_pct5)
        scaled_value_targets_H_B = (value_targets_H_B - offset) / invscale
        scaled_value_predictions_H_B = (value_predictions_H_B - offset) / invscale
        return scaled_value_targets_H_B - scaled_value_predictions_H_B