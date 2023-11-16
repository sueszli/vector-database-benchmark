from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.schedules import PiecewiseSchedule
(torch, nn) = try_import_torch()

@DeveloperAPI
class LearningRateSchedule:
    """Mixin for TorchPolicy that adds a learning rate schedule."""

    def __init__(self, lr, lr_schedule):
        if False:
            return 10
        self._lr_schedule = None
        if lr_schedule is None:
            self.cur_lr = lr
        else:
            self._lr_schedule = PiecewiseSchedule(lr_schedule, outside_value=lr_schedule[-1][-1], framework=None)
            self.cur_lr = self._lr_schedule.value(0)

    @override(Policy)
    def on_global_var_update(self, global_vars):
        if False:
            return 10
        super().on_global_var_update(global_vars)
        if self._lr_schedule and (not self.config.get('_enable_new_api_stack', False)):
            self.cur_lr = self._lr_schedule.value(global_vars['timestep'])
            for opt in self._optimizers:
                for p in opt.param_groups:
                    p['lr'] = self.cur_lr

@DeveloperAPI
class EntropyCoeffSchedule:
    """Mixin for TorchPolicy that adds entropy coeff decay."""

    def __init__(self, entropy_coeff, entropy_coeff_schedule):
        if False:
            i = 10
            return i + 15
        self._entropy_coeff_schedule = None
        if entropy_coeff_schedule is None or self.config.get('_enable_new_api_stack', False):
            self.entropy_coeff = entropy_coeff
        else:
            if isinstance(entropy_coeff_schedule, list):
                self._entropy_coeff_schedule = PiecewiseSchedule(entropy_coeff_schedule, outside_value=entropy_coeff_schedule[-1][-1], framework=None)
            else:
                self._entropy_coeff_schedule = PiecewiseSchedule([[0, entropy_coeff], [entropy_coeff_schedule, 0.0]], outside_value=0.0, framework=None)
            self.entropy_coeff = self._entropy_coeff_schedule.value(0)

    @override(Policy)
    def on_global_var_update(self, global_vars):
        if False:
            print('Hello World!')
        super(EntropyCoeffSchedule, self).on_global_var_update(global_vars)
        if self._entropy_coeff_schedule is not None:
            self.entropy_coeff = self._entropy_coeff_schedule.value(global_vars['timestep'])

@DeveloperAPI
class KLCoeffMixin:
    """Assigns the `update_kl()` method to a TorchPolicy.

    This is used by Algorithms to update the KL coefficient
    after each learning step based on `config.kl_target` and
    the measured KL value (from the train_batch).
    """

    def __init__(self, config):
        if False:
            return 10
        self.kl_coeff = config['kl_coeff']
        self.kl_target = config['kl_target']

    def update_kl(self, sampled_kl):
        if False:
            return 10
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff *= 0.5
        return self.kl_coeff

    @override(TorchPolicy)
    def get_state(self) -> PolicyState:
        if False:
            return 10
        state = super().get_state()
        state['current_kl_coeff'] = self.kl_coeff
        return state

    @override(TorchPolicy)
    def set_state(self, state: PolicyState) -> None:
        if False:
            return 10
        self.kl_coeff = state.pop('current_kl_coeff', self.config['kl_coeff'])
        super().set_state(state)

@DeveloperAPI
class ValueNetworkMixin:
    """Assigns the `_value()` method to a TorchPolicy.

    This way, Policy can call `_value()` to get the current VF estimate on a
    single(!) observation (as done in `postprocess_trajectory_fn`).
    Note: When doing this, an actual forward pass is being performed.
    This is different from only calling `model.value_function()`, where
    the result of the most recent forward pass is being used to return an
    already calculated tensor.
    """

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        if config.get('use_gae') or config.get('vtrace'):

            def value(**input_dict):
                if False:
                    print('Hello World!')
                input_dict = SampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                (model_out, _) = self.model(input_dict)
                return self.model.value_function()[0].item()
        else:

            def value(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                return 0.0
        self._value = value

    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        if False:
            print('Hello World!')
        "Defines extra fetches per action computation.\n\n        Args:\n            input_dict (Dict[str, TensorType]): The input dict used for the action\n                computing forward pass.\n            state_batches (List[TensorType]): List of state tensors (empty for\n                non-RNNs).\n            model (ModelV2): The Model object of the Policy.\n            action_dist: The instantiated distribution\n                object, resulting from the model's outputs and the given\n                distribution class.\n\n        Returns:\n            Dict[str, TensorType]: Dict with extra tf fetches to perform per\n                action computation.\n        "
        return {SampleBatch.VF_PREDS: model.value_function()}

@DeveloperAPI
class TargetNetworkMixin:
    """Mixin class adding a method for (soft) target net(s) synchronizations.

    - Adds the `update_target` method to the policy.
      Calling `update_target` updates all target Q-networks' weights from their
      respective "main" Q-networks, based on tau (smooth, partial updating).
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        tau = self.config.get('tau', 1.0)
        self.update_target(tau=tau)

    def update_target(self, tau=None):
        if False:
            i = 10
            return i + 15
        tau = tau or self.config.get('tau', 1.0)
        model_state_dict = self.model.state_dict()
        if self.config.get('_enable_new_api_stack', False):
            target_current_network_pairs = self.model.get_target_network_pairs()
            for (target_network, current_network) in target_current_network_pairs:
                current_state_dict = current_network.state_dict()
                new_state_dict = {k: tau * current_state_dict[k] + (1 - tau) * v for (k, v) in target_network.state_dict().items()}
                target_network.load_state_dict(new_state_dict)
        else:
            target_state_dict = next(iter(self.target_models.values())).state_dict()
            model_state_dict = {k: tau * model_state_dict[k] + (1 - tau) * v for (k, v) in target_state_dict.items()}
            for target in self.target_models.values():
                target.load_state_dict(model_state_dict)

    @override(TorchPolicy)
    def set_weights(self, weights):
        if False:
            for i in range(10):
                print('nop')
        TorchPolicy.set_weights(self, weights)
        self.update_target()