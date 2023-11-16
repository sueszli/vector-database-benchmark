"""
.. _user_defined_sampler:

User-Defined Sampler
====================

Thanks to user-defined samplers, you can:

- experiment your own sampling algorithms,
- implement task-specific algorithms to refine the optimization performance, or
- wrap other optimization libraries to integrate them into Optuna pipelines (e.g., :class:`~optuna.integration.BoTorchSampler`).

This section describes the internal behavior of sampler classes and shows an example of implementing a user-defined sampler.


Overview of Sampler
-------------------

A sampler has the responsibility to determine the parameter values to be evaluated in a trial.
When a `suggest` API (e.g., :func:`~optuna.trial.Trial.suggest_float`) is called inside an objective function, the corresponding distribution object (e.g., :class:`~optuna.distributions.FloatDistribution`) is created internally. A sampler samples a parameter value from the distribution. The sampled value is returned to the caller of the `suggest` API and evaluated in the objective function.

To create a new sampler, you need to define a class that inherits :class:`~optuna.samplers.BaseSampler`.
The base class has three abstract methods;
:meth:`~optuna.samplers.BaseSampler.infer_relative_search_space`,
:meth:`~optuna.samplers.BaseSampler.sample_relative`, and
:meth:`~optuna.samplers.BaseSampler.sample_independent`.

As the method names imply, Optuna supports two types of sampling: one is **relative sampling** that can consider the correlation of the parameters in a trial, and the other is **independent sampling** that samples each parameter independently.

At the beginning of a trial, :meth:`~optuna.samplers.BaseSampler.infer_relative_search_space` is called to provide the relative search space for the trial. Then, :meth:`~optuna.samplers.BaseSampler.sample_relative` is invoked to sample relative parameters from the search space. During the execution of the objective function, :meth:`~optuna.samplers.BaseSampler.sample_independent` is used to sample parameters that don't belong to the relative search space.

.. note::
    Please refer to the document of :class:`~optuna.samplers.BaseSampler` for further details.


An Example: Implementing SimulatedAnnealingSampler
--------------------------------------------------

For example, the following code defines a sampler based on
`Simulated Annealing (SA) <https://en.wikipedia.org/wiki/Simulated_annealing>`_:
"""
import numpy as np
import optuna

class SimulatedAnnealingSampler(optuna.samplers.BaseSampler):

    def __init__(self, temperature=100):
        if False:
            while True:
                i = 10
        self._rng = np.random.RandomState()
        self._temperature = temperature
        self._current_trial = None

    def sample_relative(self, study, trial, search_space):
        if False:
            i = 10
            return i + 15
        if search_space == {}:
            return {}
        prev_trial = study.trials[-2]
        if self._current_trial is None or prev_trial.value <= self._current_trial.value:
            probability = 1.0
        else:
            probability = np.exp((self._current_trial.value - prev_trial.value) / self._temperature)
        self._temperature *= 0.9
        if self._rng.uniform(0, 1) < probability:
            self._current_trial = prev_trial
        params = {}
        for (param_name, param_distribution) in search_space.items():
            if not isinstance(param_distribution, optuna.distributions.FloatDistribution) or (param_distribution.step is not None and param_distribution.step != 1) or param_distribution.log:
                msg = 'Only suggest_float() with `step` `None` or 1.0 and `log` `False` is supported'
                raise NotImplementedError(msg)
            current_value = self._current_trial.params[param_name]
            width = (param_distribution.high - param_distribution.low) * 0.1
            neighbor_low = max(current_value - width, param_distribution.low)
            neighbor_high = min(current_value + width, param_distribution.high)
            params[param_name] = self._rng.uniform(neighbor_low, neighbor_high)
        return params

    def infer_relative_search_space(self, study, trial):
        if False:
            i = 10
            return i + 15
        return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))

    def sample_independent(self, study, trial, param_name, param_distribution):
        if False:
            print('Hello World!')
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(study, trial, param_name, param_distribution)

def objective(trial):
    if False:
        i = 10
        return i + 15
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -5, 5)
    return x ** 2 + y
sampler = SimulatedAnnealingSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)
best_trial = study.best_trial
print('Best value: ', best_trial.value)
print('Parameters that achieve the best value: ', best_trial.params)