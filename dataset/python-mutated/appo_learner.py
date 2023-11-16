import abc
from dataclasses import dataclass
from typing import Any, Mapping
from ray.rllib.algorithms.impala.impala_learner import ImpalaLearner, ImpalaLearnerHyperparameters
from ray.rllib.core.rl_module.marl_module import ModuleID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.metrics import LAST_TARGET_UPDATE_TS, NUM_TARGET_UPDATES
from ray.rllib.utils.schedules.scheduler import Scheduler
LEARNER_RESULTS_KL_KEY = 'mean_kl_loss'
LEARNER_RESULTS_CURR_KL_COEFF_KEY = 'curr_kl_coeff'
OLD_ACTION_DIST_KEY = 'old_action_dist'
OLD_ACTION_DIST_LOGITS_KEY = 'old_action_dist_logits'

@dataclass
class AppoLearnerHyperparameters(ImpalaLearnerHyperparameters):
    """Hyperparameters for the APPOLearner sub-classes (framework specific).

    These should never be set directly by the user. Instead, use the APPOConfig
    class to configure your algorithm.
    See `ray.rllib.algorithms.appo.appo::APPOConfig::training()` for more details on the
    individual properties.
    """
    use_kl_loss: bool = None
    kl_coeff: float = None
    kl_target: float = None
    clip_param: float = None
    tau: float = None
    target_update_frequency_ts: int = None

class AppoLearner(ImpalaLearner):
    """Adds KL coeff updates via `additional_update_for_module()` to Impala logic.

    Framework-specific sub-classes must override `_update_module_target_networks()`
    and `_update_module_kl_coeff()`
    """

    @override(ImpalaLearner)
    def build(self):
        if False:
            print('Hello World!')
        super().build()
        self.curr_kl_coeffs_per_module: LambdaDefaultDict[ModuleID, Scheduler] = LambdaDefaultDict(lambda module_id: self._get_tensor_variable(self.hps.get_hps_for_module(module_id).kl_coeff))

    @override(ImpalaLearner)
    def remove_module(self, module_id: str):
        if False:
            return 10
        super().remove_module(module_id)
        self.curr_kl_coeffs_per_module.pop(module_id)

    @override(ImpalaLearner)
    def additional_update_for_module(self, *, module_id: ModuleID, hps: AppoLearnerHyperparameters, timestep: int, last_update: int, mean_kl_loss_per_module: dict, **kwargs) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        'Updates the target networks and KL loss coefficients (per module).\n\n        Args:\n            module_id:\n        '
        results = super().additional_update_for_module(module_id=module_id, hps=hps, timestep=timestep)
        if timestep - last_update >= hps.target_update_frequency_ts:
            self._update_module_target_networks(module_id, hps)
            results[NUM_TARGET_UPDATES] = 1
            results[LAST_TARGET_UPDATE_TS] = timestep
        else:
            results[NUM_TARGET_UPDATES] = 0
            results[LAST_TARGET_UPDATE_TS] = last_update
        if hps.use_kl_loss and module_id in mean_kl_loss_per_module:
            results.update(self._update_module_kl_coeff(module_id, hps, mean_kl_loss_per_module[module_id]))
        return results

    @abc.abstractmethod
    def _update_module_target_networks(self, module_id: ModuleID, hps: AppoLearnerHyperparameters) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update the target policy of each module with the current policy.\n\n        Do that update via polyak averaging.\n\n        Args:\n            module_id: The module ID, whose target network(s) need to be updated.\n            hps: The hyperparameters specific to the given `module_id`.\n        '

    @abc.abstractmethod
    def _update_module_kl_coeff(self, module_id: ModuleID, hps: AppoLearnerHyperparameters, sampled_kl: float) -> Mapping[str, Any]:
        if False:
            return 10
        'Dynamically update the KL loss coefficients of each module with.\n\n        The update is completed using the mean KL divergence between the action\n        distributions current policy and old policy of each module. That action\n        distribution is computed during the most recent update/call to `compute_loss`.\n\n        Args:\n            module_id: The module whose KL loss coefficient to update.\n            hps: The hyperparameters specific to the given `module_id`.\n            sampled_kl: The computed KL loss for the given Module\n                (KL divergence between the action distributions of the current\n                (most recently updated) module and the old module version).\n        '