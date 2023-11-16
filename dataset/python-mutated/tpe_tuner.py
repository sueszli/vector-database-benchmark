"""
Tree-structured Parzen Estimator (TPE) tuner.

Paper: https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf

Official code: https://github.com/hyperopt/hyperopt/blob/master/hyperopt/tpe.py

This is a slightly modified re-implementation of the algorithm.
"""
from __future__ import annotations
__all__ = ['TpeTuner', 'TpeArguments']
from collections import defaultdict
import logging
import math
from typing import Any, NamedTuple
import numpy as np
from scipy.special import erf
from typing_extensions import Literal
import nni
from nni.common.hpo_utils import Deduplicator, OptimizeMode, format_search_space, deformat_parameters, format_parameters
from nni.tuner import Tuner
from nni.utils import extract_scalar_reward
from . import random_tuner
_logger = logging.getLogger('nni.tuner.tpe')

class TpeArguments(NamedTuple):
    """
    Hyperparameters of TPE algorithm itself.

    To avoid confusing with trials' hyperparameters to be tuned, these are called "arguments" here.

    Parameters
    ----------
    constant_liar_type
        TPE algorithm itself does not support parallel tuning.
        This parameter specifies how to optimize for trial_concurrency > 1.

        None (or "null" in YAML) means do not optimize. This is the default behavior in legacy version.

        How each liar works is explained in paper's section 6.1.
        In general "best" suit for small trial number and "worst" suit for large trial number.
        (:doc:`experiment result </sharings/parallelizing_tpe_search>`)

    n_startup_jobs
        The first N hyperparameters are generated fully randomly for warming up.
        If the search space is large, you can increase this value.
        Or if max_trial_number is small, you may want to decrease it.

    n_ei_candidates
        For each iteration TPE samples EI for N sets of parameters and choose the best one. (loosely speaking)

    linear_forgetting
        TPE will lower the weights of old trials.
        This controls how many iterations it takes for a trial to start decay.

    prior_weight
        TPE treats user provided search space as prior.
        When generating new trials, it also incorporates the prior in trial history by transforming the search space to
        one trial configuration (i.e., each parameter of this configuration chooses the mean of its candidate range).
        Here, prior_weight determines the weight of this trial configuration in the history trial configurations.

        With prior weight 1.0, the search space is treated as one good trial.
        For example, "normal(0, 1)" effectly equals to a trial with x = 0 which has yielded good result.

    gamma
        Controls how many trials are considered "good".
        The number is calculated as "min(gamma * sqrt(N), linear_forgetting)".
    """
    constant_liar_type: Literal['best', 'worst', 'mean'] | None = 'best'
    n_startup_jobs: int = 20
    n_ei_candidates: int = 24
    linear_forgetting: int = 25
    prior_weight: float = 1.0
    gamma: float = 0.25

class TpeTuner(Tuner):
    """
    Tree-structured Parzen Estimator (TPE) tuner.

    TPE is a lightweight tuner that has no extra dependency and supports all search space types,
    designed to be the default tuner.

    It has the drawback that TPE cannot discover relationship between different hyperparameters.

    **Implementation**

    TPE is an SMBO algorithm.
    It models P(x|y) and P(y) where x represents hyperparameters and y the evaluation result.
    P(x|y) is modeled by transforming the generative process of hyperparameters,
    replacing the distributions of the configuration prior with non-parametric densities.

    Paper: `Algorithms for Hyper-Parameter Optimization
    <https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf>`__

    Examples
    --------

    .. code-block::

        ## minimal config ##

        config.tuner.name = 'TPE'
        config.tuner.class_args = {
            'optimize_mode': 'maximize'
        }

    .. code-block::

        ## advanced config ##

        config.tuner.name = 'TPE'
        config.tuner.class_args = {
            'optimize_mode': maximize,
            'seed': 12345,
            'tpe_args': {
                'constant_liar_type': 'mean',
                'n_startup_jobs': 10,
                'n_ei_candidates': 20,
                'linear_forgetting': 100,
                'prior_weight': 0,
                'gamma': 0.5
            }
        }

    Parameters
    ----------
    optimze_mode: Literal['minimize', 'maximize']
        Whether optimize to minimize or maximize trial result.
    seed
        The random seed.
    tpe_args
        Advanced users can use this to customize TPE tuner.
        See :class:`TpeArguments` for details.
    """

    def __init__(self, optimize_mode: Literal['minimize', 'maximize']='minimize', seed: int | None=None, tpe_args: dict[str, Any] | None=None):
        if False:
            for i in range(10):
                print('nop')
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.args = TpeArguments(**tpe_args or {})
        self.space = None
        self.liar = create_liar(self.args.constant_liar_type)
        self.dedup = None
        if seed is None:
            seed = np.random.default_rng().integers(2 ** 31)
        self.rng = np.random.default_rng(seed)
        _logger.info(f'Using random seed {seed}')
        self._params = {}
        self._running_params = {}
        self._history = defaultdict(list)

    def update_search_space(self, space):
        if False:
            i = 10
            return i + 15
        self.space = format_search_space(space)
        self.dedup = Deduplicator(self.space)

    def generate_parameters(self, parameter_id, **kwargs):
        if False:
            print('Hello World!')
        if self.liar and self._running_params:
            history = defaultdict(list, {key: records.copy() for (key, records) in self._history.items()})
            lie = self.liar.lie()
            for param in self._running_params.values():
                for (key, value) in param.items():
                    history[key].append(Record(value, lie))
        else:
            history = self._history
        params = suggest(self.args, self.rng, self.space, history)
        params = self.dedup(params)
        self._params[parameter_id] = params
        self._running_params[parameter_id] = params
        return deformat_parameters(params, self.space)

    def receive_trial_result(self, parameter_id, _parameters, value, **kwargs):
        if False:
            print('Hello World!')
        if self.optimize_mode is OptimizeMode.Minimize:
            loss = extract_scalar_reward(value)
        else:
            loss = -extract_scalar_reward(value)
        if self.liar:
            self.liar.update(loss)
        params = self._running_params.pop(parameter_id)
        for (key, value) in params.items():
            self._history[key].append(Record(value, loss))

    def trial_end(self, parameter_id, _success, **kwargs):
        if False:
            return 10
        self._running_params.pop(parameter_id, None)

    def import_data(self, data):
        if False:
            i = 10
            return i + 15
        if isinstance(data, str):
            data = nni.load(data)
        for trial in data:
            if isinstance(trial, str):
                trial = nni.load(trial)
            param = format_parameters(trial['parameter'], self.space)
            loss = trial['value']
            if isinstance(loss, dict) and 'default' in loss:
                loss = loss['default']
            if self.optimize_mode is OptimizeMode.Maximize:
                loss = -loss
            for (key, value) in param.items():
                self._history[key].append(Record(value, loss))
                self.dedup.add_history(param)
        _logger.info(f'Replayed {len(data)} FINISHED trials')

def suggest(args, rng, space, history):
    if False:
        i = 10
        return i + 15
    params = {}
    for (key, spec) in space.items():
        if spec.is_activated_in(params):
            params[key] = suggest_parameter(args, rng, spec, history[key])
    return params

def suggest_parameter(args, rng, spec, parameter_history):
    if False:
        return 10
    if len(parameter_history) < args.n_startup_jobs:
        return random_tuner.suggest_parameter(rng, spec)
    if spec.categorical:
        return suggest_categorical(args, rng, parameter_history, spec.size)
    if spec.normal_distributed:
        mu = spec.mu
        sigma = spec.sigma
        clip = None
    else:
        mu = (spec.low + spec.high) * 0.5
        sigma = spec.high - spec.low
        clip = (spec.low, spec.high)
    return suggest_normal(args, rng, parameter_history, mu, sigma, clip)

class Record(NamedTuple):
    param: int | float
    loss: float

class BestLiar:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._best = None

    def update(self, loss):
        if False:
            for i in range(10):
                print('nop')
        if self._best is None or loss < self._best:
            self._best = loss

    def lie(self):
        if False:
            for i in range(10):
                print('nop')
        return 0.0 if self._best is None else self._best

class WorstLiar:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._worst = None

    def update(self, loss):
        if False:
            for i in range(10):
                print('nop')
        if self._worst is None or loss > self._worst:
            self._worst = loss

    def lie(self):
        if False:
            while True:
                i = 10
        return 0.0 if self._worst is None else self._worst

class MeanLiar:

    def __init__(self):
        if False:
            print('Hello World!')
        self._sum = 0.0
        self._n = 0

    def update(self, loss):
        if False:
            while True:
                i = 10
        self._sum += loss
        self._n += 1

    def lie(self):
        if False:
            for i in range(10):
                print('nop')
        return 0.0 if self._n == 0 else self._sum / self._n

def create_liar(liar_type):
    if False:
        while True:
            i = 10
    if liar_type is None or liar_type.lower == 'none':
        return None
    liar_classes = {'best': BestLiar, 'worst': WorstLiar, 'mean': MeanLiar}
    return liar_classes[liar_type.lower()]()

def suggest_categorical(args, rng, param_history, size):
    if False:
        for i in range(10):
            print('nop')
    '\n    Suggest a categorical ("choice" or "randint") parameter.\n    '
    (below, above) = split_history(args, param_history)
    weights = linear_forgetting_weights(args, len(below))
    counts = np.bincount(below, weights, size)
    p = (counts + args.prior_weight) / sum(counts + args.prior_weight)
    samples = rng.choice(size, args.n_ei_candidates, p=p)
    below_llik = np.log(p[samples])
    weights = linear_forgetting_weights(args, len(above))
    counts = np.bincount(above, weights, size)
    p = (counts + args.prior_weight) / sum(counts + args.prior_weight)
    above_llik = np.log(p[samples])
    return samples[np.argmax(below_llik - above_llik)]

def suggest_normal(args, rng, param_history, prior_mu, prior_sigma, clip):
    if False:
        return 10
    '\n    Suggest a normal distributed parameter.\n    Uniform has been converted to normal in the caller function; log and q will be handled by "deformat_parameters".\n    '
    (below, above) = split_history(args, param_history)
    (weights, mus, sigmas) = adaptive_parzen_normal(args, below, prior_mu, prior_sigma)
    samples = gmm1(args, rng, weights, mus, sigmas, clip)
    below_llik = gmm1_lpdf(args, samples, weights, mus, sigmas, clip)
    (weights, mus, sigmas) = adaptive_parzen_normal(args, above, prior_mu, prior_sigma)
    above_llik = gmm1_lpdf(args, samples, weights, mus, sigmas, clip)
    return samples[np.argmax(below_llik - above_llik)]

def split_history(args, param_history):
    if False:
        print('Hello World!')
    '\n    Divide trials into good ones (below) and bad ones (above).\n    '
    n_below = math.ceil(args.gamma * math.sqrt(len(param_history)))
    n_below = min(n_below, args.linear_forgetting)
    order = sorted(range(len(param_history)), key=lambda i: param_history[i].loss)
    below = [param_history[i].param for i in order[:n_below]]
    above = [param_history[i].param for i in order[n_below:]]
    return (np.asarray(below), np.asarray(above))

def linear_forgetting_weights(args, n):
    if False:
        while True:
            i = 10
    '\n    Calculate decayed weights of N trials.\n    '
    lf = args.linear_forgetting
    if n < lf:
        return np.ones(n)
    else:
        ramp = np.linspace(1.0 / n, 1.0, n - lf)
        flat = np.ones(lf)
        return np.concatenate([ramp, flat])

def adaptive_parzen_normal(args, history_mus, prior_mu, prior_sigma):
    if False:
        while True:
            i = 10
    '\n    The "Adaptive Parzen Estimator" described in paper section 4.2, for normal distribution.\n\n    Because TPE internally only supports categorical and normal distributed space (domain),\n    this function is used for everything other than "choice" and "randint".\n\n    Parameters\n    ----------\n    args: TpeArguments\n        Algorithm arguments.\n    history_mus: 1-d array of float\n        Parameter values evaluated in history.\n        These are the "observations" in paper section 4.2. ("placing density in the vicinity of K observations")\n    prior_mu: float\n        µ value of normal search space.\n    piror_sigma: float\n        σ value of normal search space.\n\n    Returns\n    -------\n    Tuple of three 1-d float arrays: (weight, µ, σ).\n\n    The tuple represents N+1 "vicinity of observations" and each one\'s weight,\n    calculated from "N" history and "1" user provided prior.\n\n    The result is sorted by µ.\n    '
    mus = np.append(history_mus, prior_mu)
    order = np.argsort(mus)
    mus = mus[order]
    prior_index = np.searchsorted(mus, prior_mu)
    if len(mus) == 1:
        sigmas = np.asarray([prior_sigma])
    elif len(mus) == 2:
        sigmas = np.asarray([prior_sigma * 0.5, prior_sigma * 0.5])
        sigmas[prior_index] = prior_sigma
    else:
        l_delta = mus[1:-1] - mus[:-2]
        r_delta = mus[2:] - mus[1:-1]
        sigmas_mid = np.maximum(l_delta, r_delta)
        sigmas = np.concatenate([[mus[1] - mus[0]], sigmas_mid, [mus[-1] - mus[-2]]])
        sigmas[prior_index] = prior_sigma
    n = min(100, len(mus) + 1)
    sigmas = np.clip(sigmas, prior_sigma / n, prior_sigma)
    weights = np.append(linear_forgetting_weights(args, len(mus) - 1), args.prior_weight)
    weights = weights[order]
    return (weights / np.sum(weights), mus, sigmas)

def gmm1(args, rng, weights, mus, sigmas, clip=None):
    if False:
        return 10
    '\n    Gaussian Mixture Model 1D.\n    '
    ret = np.asarray([])
    while len(ret) < args.n_ei_candidates:
        n = args.n_ei_candidates - len(ret)
        active = np.argmax(rng.multinomial(1, weights, n), axis=1)
        samples = rng.normal(mus[active], sigmas[active])
        if clip:
            samples = samples[(clip[0] <= samples) & (samples <= clip[1])]
        ret = np.concatenate([ret, samples])
    return ret

def gmm1_lpdf(_args, samples, weights, mus, sigmas, clip=None):
    if False:
        print('Hello World!')
    "\n    Gaussian Mixture Model 1D's log probability distribution function.\n    "
    eps = 1e-12
    if clip:
        normal_cdf_low = erf((clip[0] - mus) / np.maximum(np.sqrt(2) * sigmas, eps)) * 0.5 + 0.5
        normal_cdf_high = erf((clip[1] - mus) / np.maximum(np.sqrt(2) * sigmas, eps)) * 0.5 + 0.5
        p_accept = np.sum(weights * (normal_cdf_high - normal_cdf_low))
    else:
        p_accept = 1
    dist = samples.reshape(-1, 1) - mus
    mahal = (dist / np.maximum(sigmas, eps)) ** 2
    z = np.sqrt(2 * np.pi) * sigmas
    coef = weights / z / p_accept
    normal_lpdf = -0.5 * mahal + np.log(coef)
    m = normal_lpdf.max(axis=1)
    e = np.exp(normal_lpdf - m.reshape(-1, 1))
    return np.log(e.sum(axis=1)) + m