import logging
from typing import Any, Dict, List, Optional, Type, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.dynamic_tf_policy_v2 import DynamicTFPolicyV2
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import ValueNetworkMixin, compute_gradients
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, get_variable
from ray.rllib.utils.tf_utils import explained_variance
from ray.rllib.utils.typing import LocalOptimizer, ModelGradients, TensorType
(tf1, tf, tfv) = try_import_tf()
logger = logging.getLogger(__name__)

class PostprocessAdvantages:
    """Marwil's custom trajectory post-processing mixin."""

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def postprocess_trajectory(self, sample_batch: SampleBatch, other_agent_batches: Optional[Dict[Any, SampleBatch]]=None, episode: Optional['Episode']=None):
        if False:
            return 10
        sample_batch = super().postprocess_trajectory(sample_batch, other_agent_batches, episode)
        if sample_batch[SampleBatch.TERMINATEDS][-1]:
            last_r = 0.0
        else:
            index = 'last' if SampleBatch.NEXT_OBS in sample_batch else -1
            input_dict = sample_batch.get_single_step_input_dict(self.view_requirements, index=index)
            last_r = self._value(**input_dict)
        return compute_advantages(sample_batch, last_r, self.config['gamma'], use_gae=False, use_critic=False)

class MARWILLoss:

    def __init__(self, policy: Policy, value_estimates: TensorType, action_dist: ActionDistribution, train_batch: SampleBatch, vf_loss_coeff: float, beta: float):
        if False:
            return 10
        logprobs = action_dist.logp(train_batch[SampleBatch.ACTIONS])
        if beta != 0.0:
            cumulative_rewards = train_batch[Postprocessing.ADVANTAGES]
            adv = cumulative_rewards - value_estimates
            adv_squared = tf.reduce_mean(tf.math.square(adv))
            self.v_loss = 0.5 * adv_squared
            rate = policy.config['moving_average_sqd_adv_norm_update_rate']
            if policy.config['framework'] == 'tf2':
                update_term = adv_squared - policy._moving_average_sqd_adv_norm
                policy._moving_average_sqd_adv_norm.assign_add(rate * update_term)
                c = tf.math.sqrt(policy._moving_average_sqd_adv_norm)
                exp_advs = tf.math.exp(beta * (adv / (1e-08 + c)))
            else:
                update_adv_norm = tf1.assign_add(ref=policy._moving_average_sqd_adv_norm, value=rate * (adv_squared - policy._moving_average_sqd_adv_norm))
                with tf1.control_dependencies([update_adv_norm]):
                    exp_advs = tf.math.exp(beta * tf.math.divide(adv, 1e-08 + tf.math.sqrt(policy._moving_average_sqd_adv_norm)))
            exp_advs = tf.stop_gradient(exp_advs)
            self.explained_variance = tf.reduce_mean(explained_variance(cumulative_rewards, value_estimates))
        else:
            self.v_loss = tf.constant(0.0)
            exp_advs = 1.0
        logstd_coeff = policy.config['bc_logstd_coeff']
        if logstd_coeff > 0.0:
            logstds = tf.reduce_sum(action_dist.log_std, axis=1)
        else:
            logstds = 0.0
        self.p_loss = -1.0 * tf.reduce_mean(exp_advs * (logprobs + logstd_coeff * logstds))
        self.total_loss = self.p_loss + vf_loss_coeff * self.v_loss

def get_marwil_tf_policy(name: str, base: type) -> type:
    if False:
        for i in range(10):
            print('nop')
    'Construct a MARWILTFPolicy inheriting either dynamic or eager base policies.\n\n    Args:\n        base: Base class for this policy. DynamicTFPolicyV2 or EagerTFPolicyV2.\n\n    Returns:\n        A TF Policy to be used with MAML.\n    '

    class MARWILTFPolicy(ValueNetworkMixin, PostprocessAdvantages, base):

        def __init__(self, observation_space, action_space, config, existing_model=None, existing_inputs=None):
            if False:
                for i in range(10):
                    print('nop')
            base.enable_eager_execution_if_necessary()
            base.__init__(self, observation_space, action_space, config, existing_inputs=existing_inputs, existing_model=existing_model)
            ValueNetworkMixin.__init__(self, config)
            PostprocessAdvantages.__init__(self)
            if config['beta'] != 0.0:
                self._moving_average_sqd_adv_norm = get_variable(config['moving_average_sqd_adv_norm_start'], framework='tf', tf_name='moving_average_of_advantage_norm', trainable=False)
            self.maybe_initialize_optimizer_and_loss()

        @override(base)
        def loss(self, model: Union[ModelV2, 'tf.keras.Model'], dist_class: Type[TFActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
            if False:
                while True:
                    i = 10
            (model_out, _) = model(train_batch)
            action_dist = dist_class(model_out, model)
            value_estimates = model.value_function()
            self._marwil_loss = MARWILLoss(self, value_estimates, action_dist, train_batch, self.config['vf_coeff'], self.config['beta'])
            return self._marwil_loss.total_loss

        @override(base)
        def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
            if False:
                print('Hello World!')
            stats = {'policy_loss': self._marwil_loss.p_loss, 'total_loss': self._marwil_loss.total_loss}
            if self.config['beta'] != 0.0:
                stats['moving_average_sqd_adv_norm'] = self._moving_average_sqd_adv_norm
                stats['vf_explained_var'] = self._marwil_loss.explained_variance
                stats['vf_loss'] = self._marwil_loss.v_loss
            return stats

        @override(base)
        def compute_gradients_fn(self, optimizer: LocalOptimizer, loss: TensorType) -> ModelGradients:
            if False:
                print('Hello World!')
            return compute_gradients(self, optimizer, loss)
    MARWILTFPolicy.__name__ = name
    MARWILTFPolicy.__qualname__ = name
    return MARWILTFPolicy
MARWILTF1Policy = get_marwil_tf_policy('MARWILTF1Policy', DynamicTFPolicyV2)
MARWILTF2Policy = get_marwil_tf_policy('MARWILTF2Policy', EagerTFPolicyV2)