from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Union
import numpy as np
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
import optuna
from optuna import pruners
from optuna import samplers
from optuna.integration.botorch import BoTorchSampler
from optuna.integration.cma import PyCmaSampler
_SAMPLERS = {'GridSampler': samplers.GridSampler, 'RandomSampler': samplers.RandomSampler, 'TPESampler': samplers.TPESampler, 'CmaEsSampler': samplers.CmaEsSampler, 'NSGAIISampler': samplers.NSGAIISampler, 'QMCSampler': samplers.QMCSampler, 'BoTorchSampler': BoTorchSampler, 'PyCmaSampler': PyCmaSampler}
_PRUNERS = {'NopPruner': pruners.NopPruner, 'MedianPruner': pruners.MedianPruner, 'PatientPruner': pruners.PatientPruner, 'PercentilePruner': pruners.PercentilePruner, 'SuccessiveHalvingPruner': pruners.SuccessiveHalvingPruner, 'HyperbandPruner': pruners.HyperbandPruner, 'ThresholdPruner': pruners.ThresholdPruner}
Suggestion = Dict[str, Union[int, float]]
ApiConfig = Dict[str, Dict[str, str]]

class OptunaOptimizer(AbstractOptimizer):
    primary_import = 'optuna'

    def __init__(self, api_config: ApiConfig, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(api_config, **kwargs)
        try:
            sampler = _SAMPLERS[kwargs['sampler']]
            sampler_kwargs: dict[str, Any] = kwargs['sampler_kwargs']
        except KeyError:
            raise ValueError('Unknown sampler passed to Optuna optimizer.')
        try:
            pruner = _PRUNERS[kwargs['pruner']]
            pruner_kwargs: dict[str, Any] = kwargs['pruner_kwargs']
        except KeyError:
            raise ValueError('Unknown pruner passed to Optuna optimizer.')
        self.study = optuna.create_study(direction='minimize', sampler=sampler(**sampler_kwargs), pruner=pruner(**pruner_kwargs))
        self.current_trials: dict[int, int] = dict()

    def _suggest(self, trial: optuna.trial.Trial) -> Suggestion:
        if False:
            print('Hello World!')
        suggestions: Suggestion = dict()
        for (name, config) in self.api_config.items():
            (low, high) = config['range']
            log = config['space'] == 'log'
            if config['space'] == 'logit':
                assert 0 < low <= high < 1
                low = np.log(low / (1 - low))
                high = np.log(high / (1 - high))
            if config['type'] == 'real':
                param = trial.suggest_float(name, low, high, log=log)
            elif config['type'] == 'int':
                param = trial.suggest_int(name, low, high, log=log)
            else:
                raise RuntimeError('CategoricalDistribution is not supported in bayesmark.')
            suggestions[name] = param if config['space'] != 'logit' else 1 / (1 + np.exp(-param))
        return suggestions

    def suggest(self, n_suggestions: int) -> list[Suggestion]:
        if False:
            for i in range(10):
                print('nop')
        suggestions: list[Suggestion] = list()
        for _ in range(n_suggestions):
            trial = self.study.ask()
            params = self._suggest(trial)
            sid = hash(frozenset(params.items()))
            self.current_trials[sid] = trial.number
            suggestions.append(params)
        return suggestions

    def observe(self, X: list[Suggestion], y: list[float]) -> None:
        if False:
            while True:
                i = 10
        for (params, objective_value) in zip(X, y):
            sid = hash(frozenset(params.items()))
            trial = self.current_trials.pop(sid)
            self.study.tell(trial, objective_value)
if __name__ == '__main__':
    optuna.logging.disable_default_handler()
    experiment_main(OptunaOptimizer)