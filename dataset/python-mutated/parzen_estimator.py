from typing import Callable
from typing import Dict
from typing import NamedTuple
from typing import Optional
import numpy as np
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers._tpe.probability_distributions import _BatchedCategoricalDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDiscreteTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _MixtureOfProductDistribution
EPS = 1e-12

class _ParzenEstimatorParameters(NamedTuple('_ParzenEstimatorParameters', [('consider_prior', bool), ('prior_weight', Optional[float]), ('consider_magic_clip', bool), ('consider_endpoints', bool), ('weights', Callable[[int], np.ndarray]), ('multivariate', bool), ('categorical_distance_func', Dict[str, Callable[[CategoricalChoiceType, CategoricalChoiceType], float]])])):
    pass

class _ParzenEstimator:

    def __init__(self, observations: Dict[str, np.ndarray], search_space: Dict[str, BaseDistribution], parameters: _ParzenEstimatorParameters, predetermined_weights: Optional[np.ndarray]=None) -> None:
        if False:
            print('Hello World!')
        if parameters.consider_prior:
            if parameters.prior_weight is None:
                raise ValueError('Prior weight must be specified when consider_prior==True.')
            elif parameters.prior_weight <= 0:
                raise ValueError('Prior weight must be positive.')
        self._search_space = search_space
        transformed_observations = self._transform(observations)
        assert predetermined_weights is None or len(transformed_observations) == len(predetermined_weights)
        weights = predetermined_weights if predetermined_weights is not None else self._call_weights_func(parameters.weights, len(transformed_observations))
        if len(transformed_observations) == 0:
            weights = np.array([1.0])
        elif parameters.consider_prior:
            assert parameters.prior_weight is not None
            weights = np.append(weights, [parameters.prior_weight])
        weights /= weights.sum()
        self._mixture_distribution = _MixtureOfProductDistribution(weights=weights, distributions=[self._calculate_distributions(transformed_observations[:, i], param, search_space[param], parameters) for (i, param) in enumerate(search_space)])

    def sample(self, rng: np.random.RandomState, size: int) -> Dict[str, np.ndarray]:
        if False:
            while True:
                i = 10
        sampled = self._mixture_distribution.sample(rng, size)
        return self._untransform(sampled)

    def log_pdf(self, samples_dict: Dict[str, np.ndarray]) -> np.ndarray:
        if False:
            while True:
                i = 10
        transformed_samples = self._transform(samples_dict)
        return self._mixture_distribution.log_pdf(transformed_samples)

    @staticmethod
    def _call_weights_func(weights_func: Callable[[int], np.ndarray], n: int) -> np.ndarray:
        if False:
            while True:
                i = 10
        w = np.array(weights_func(n))[:n]
        if np.any(w < 0):
            raise ValueError(f'The `weights` function is not allowed to return negative values {w}. ' + f'The argument of the `weights` function is {n}.')
        if len(w) > 0 and np.sum(w) <= 0:
            raise ValueError(f'The `weight` function is not allowed to return all-zero values {w}.' + f' The argument of the `weights` function is {n}.')
        if not np.all(np.isfinite(w)):
            raise ValueError('The `weights`function is not allowed to return infinite or NaN values ' + f'{w}. The argument of the `weights` function is {n}.')
        return w

    @staticmethod
    def _is_log(dist: BaseDistribution) -> bool:
        if False:
            while True:
                i = 10
        return isinstance(dist, (FloatDistribution, IntDistribution)) and dist.log

    def _transform(self, samples_dict: Dict[str, np.ndarray]) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        return np.array([np.log(samples_dict[param]) if self._is_log(self._search_space[param]) else samples_dict[param] for param in self._search_space]).T

    def _untransform(self, samples_array: np.ndarray) -> Dict[str, np.ndarray]:
        if False:
            return 10
        res = {param: np.exp(samples_array[:, i]) if self._is_log(self._search_space[param]) else samples_array[:, i] for (i, param) in enumerate(self._search_space)}
        return {param: np.clip(dist.low + np.round((res[param] - dist.low) / dist.step) * dist.step, dist.low, dist.high) if isinstance(dist, IntDistribution) else res[param] for (param, dist) in self._search_space.items()}

    def _calculate_distributions(self, transformed_observations: np.ndarray, param_name: str, search_space: BaseDistribution, parameters: _ParzenEstimatorParameters) -> _BatchedDistributions:
        if False:
            return 10
        if isinstance(search_space, CategoricalDistribution):
            return self._calculate_categorical_distributions(transformed_observations, param_name, search_space, parameters)
        else:
            assert isinstance(search_space, (FloatDistribution, IntDistribution))
            if search_space.log:
                low = np.log(search_space.low)
                high = np.log(search_space.high)
            else:
                low = search_space.low
                high = search_space.high
            step = search_space.step
            if step is not None and search_space.log:
                low = np.log(search_space.low - step / 2)
                high = np.log(search_space.high + step / 2)
                step = None
            return self._calculate_numerical_distributions(transformed_observations, low, high, step, parameters)

    def _calculate_categorical_distributions(self, observations: np.ndarray, param_name: str, search_space: CategoricalDistribution, parameters: _ParzenEstimatorParameters) -> _BatchedDistributions:
        if False:
            for i in range(10):
                print('nop')
        consider_prior = parameters.consider_prior or len(observations) == 0
        assert parameters.prior_weight is not None
        weights = np.full(shape=(len(observations) + consider_prior, len(search_space.choices)), fill_value=parameters.prior_weight / (len(observations) + consider_prior))
        if param_name in parameters.categorical_distance_func:
            dist_func = parameters.categorical_distance_func[param_name]
            for (i, observation) in enumerate(observations.astype(int)):
                dists = [dist_func(search_space.choices[observation], search_space.choices[j]) for j in range(len(search_space.choices))]
                exponent = -((np.array(dists) / max(dists)) ** 2 * np.log((len(observations) + consider_prior) / parameters.prior_weight) * (np.log(len(search_space.choices)) / np.log(6)))
                weights[i] = np.exp(exponent)
        else:
            weights[np.arange(len(observations)), observations.astype(int)] += 1
        weights /= weights.sum(axis=1, keepdims=True)
        return _BatchedCategoricalDistributions(weights)

    def _calculate_numerical_distributions(self, observations: np.ndarray, low: float, high: float, step: Optional[float], parameters: _ParzenEstimatorParameters) -> _BatchedDistributions:
        if False:
            return 10
        step_or_0 = step or 0
        mus = observations
        consider_prior = parameters.consider_prior or len(observations) == 0

        def compute_sigmas() -> np.ndarray:
            if False:
                print('Hello World!')
            if parameters.multivariate:
                SIGMA0_MAGNITUDE = 0.2
                sigma = SIGMA0_MAGNITUDE * max(len(observations), 1) ** (-1.0 / (len(self._search_space) + 4)) * (high - low + step_or_0)
                sigmas = np.full(shape=(len(observations),), fill_value=sigma)
            else:
                prior_mu = 0.5 * (low + high)
                mus_with_prior = np.append(mus, prior_mu) if consider_prior else mus
                sorted_indices = np.argsort(mus_with_prior)
                sorted_mus = mus_with_prior[sorted_indices]
                sorted_mus_with_endpoints = np.empty(len(mus_with_prior) + 2, dtype=float)
                sorted_mus_with_endpoints[0] = low - step_or_0 / 2
                sorted_mus_with_endpoints[1:-1] = sorted_mus
                sorted_mus_with_endpoints[-1] = high + step_or_0 / 2
                sorted_sigmas = np.maximum(sorted_mus_with_endpoints[1:-1] - sorted_mus_with_endpoints[0:-2], sorted_mus_with_endpoints[2:] - sorted_mus_with_endpoints[1:-1])
                if not parameters.consider_endpoints and sorted_mus_with_endpoints.shape[0] >= 4:
                    sorted_sigmas[0] = sorted_mus_with_endpoints[2] - sorted_mus_with_endpoints[1]
                    sorted_sigmas[-1] = sorted_mus_with_endpoints[-2] - sorted_mus_with_endpoints[-3]
                sigmas = sorted_sigmas[np.argsort(sorted_indices)][:len(observations)]
            maxsigma = 1.0 * (high - low + step_or_0)
            if parameters.consider_magic_clip:
                minsigma = 1.0 * (high - low + step_or_0) / min(100.0, 1.0 + len(observations) + consider_prior)
            else:
                minsigma = EPS
            return np.asarray(np.clip(sigmas, minsigma, maxsigma))
        sigmas = compute_sigmas()
        if consider_prior:
            prior_mu = 0.5 * (low + high)
            prior_sigma = 1.0 * (high - low + step_or_0)
            mus = np.append(mus, [prior_mu])
            sigmas = np.append(sigmas, [prior_sigma])
        if step is None:
            return _BatchedTruncNormDistributions(mus, sigmas, low, high)
        else:
            return _BatchedDiscreteTruncNormDistributions(mus, sigmas, low, high, step)