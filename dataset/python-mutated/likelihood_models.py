"""
Likelihood Models
-----------------

The likelihood models contain all the logic needed to train and use Darts' neural network models in
a probabilistic way. This essentially means computing an appropriate training loss and sample from the
distribution, given the parameters of the distribution.

By default, all versions will be trained using their negative log likelihood as a loss function
(hence performing maximum likelihood estimation when training the model).
However, most likelihoods also optionally support specifying time-independent "prior"
beliefs about the distribution parameters.
In such cases, the a KL-divergence term is added to the loss in order to regularise it in the
direction of the specified prior distribution. (Note that this is technically not purely
a Bayesian approach as the priors are actual parameters values, and not distributions).
The parameter `prior_strength` controls the strength of the "prior" regularisation on the loss.

Some distributions (such as ``GaussianLikelihood``, and ``PoissonLikelihood``) are univariate,
in which case they are applied to model each component of multivariate series independently.
Some other distributions (such as ``DirichletLikelihood``) are multivariate,
in which case they will model all components of multivariate time series jointly.

Univariate likelihoods accept either scalar or array-like values for the optional prior parameters.
If a scalar is provided, it is used as a prior for all components of the series. If an array-like is provided,
the i-th value will be used as a prior for the i-th component of the series. Multivariate likelihoods
require array-like objects when specifying priors.

The target series used for training must always lie within the distribution's support, otherwise
errors will be raised during training. You can refer to the individual likelihoods' documentation
to see what is the support. Similarly, the prior parameters also have to lie in some pre-defined domains.
"""
import collections.abc
import inspect
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli as _Bernoulli
from torch.distributions import Beta as _Beta
from torch.distributions import Cauchy as _Cauchy
from torch.distributions import ContinuousBernoulli as _ContinuousBernoulli
from torch.distributions import Dirichlet as _Dirichlet
from torch.distributions import Exponential as _Exponential
from torch.distributions import Gamma as _Gamma
from torch.distributions import Geometric as _Geometric
from torch.distributions import Gumbel as _Gumbel
from torch.distributions import HalfNormal as _HalfNormal
from torch.distributions import Laplace as _Laplace
from torch.distributions import LogNormal as _LogNormal
from torch.distributions import NegativeBinomial as _NegativeBinomial
from torch.distributions import Normal as _Normal
from torch.distributions import Poisson as _Poisson
from torch.distributions import Weibull as _Weibull
from torch.distributions.kl import kl_divergence
from darts import TimeSeries
from darts.utils.utils import _check_quantiles, raise_if_not
MIN_CAUCHY_GAMMA_SAMPLING = 1e-100

def _check(param, predicate, param_name, condition_str):
    if False:
        while True:
            i = 10
    if param is None:
        return
    if isinstance(param, (collections.abc.Sequence, np.ndarray)):
        raise_if_not(all((predicate(p) for p in param)), f'All provided parameters {param_name} must be {condition_str}.')
    else:
        raise_if_not(predicate(param), f'The parameter {param_name} must be {condition_str}.')

def _check_strict_positive(param, param_name=''):
    if False:
        return 10
    _check(param, lambda p: p > 0, param_name, 'strictly positive')

def _check_in_open_0_1_intvl(param, param_name=''):
    if False:
        print('Hello World!')
    _check(param, lambda p: 0 < p < 1, param_name, 'in the open interval (0, 1)')

class Likelihood(ABC):

    def __init__(self, prior_strength=1.0):
        if False:
            print('Hello World!')
        '\n        Abstract class for a likelihood model.\n        '
        self.prior_strength = prior_strength
        self.ignore_attrs_equality = []

    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor):
        if False:
            return 10
        '\n        Computes a loss from a `model_output`, which represents the parameters of a given probability\n        distribution for every ground truth value in `target`, and the `target` itself.\n        '
        params_out = self._params_from_output(model_output)
        loss = self._nllloss(params_out, target)
        prior_params = self._prior_params
        use_prior = prior_params is not None and any((p is not None for p in prior_params))
        if use_prior:
            out_distr = self._distr_from_params(params_out)
            device = params_out[0].device
            prior_params = tuple((torch.tensor(prior_params[i]).to(device) if prior_params[i] is not None else params_out[i] for i in range(len(prior_params))))
            prior_distr = self._distr_from_params(prior_params)
            loss += self.prior_strength * torch.mean(kl_divergence(prior_distr, out_distr))
        return loss

    def _nllloss(self, params_out, target):
        if False:
            while True:
                i = 10
        '\n        This is the basic way to compute the NLL loss. It can be overwritten by likelihoods for which\n        PyTorch proposes a numerically better NLL loss.\n        '
        out_distr = self._distr_from_params(params_out)
        return -out_distr.log_prob(target).mean()

    @property
    def _prior_params(self):
        if False:
            while True:
                i = 10
        '\n        Has to be overwritten by the Likelihood objects supporting specifying a prior distribution on the\n        outputs. If it returns None, no prior will be used and the model will be trained with plain maximum likelihood.\n        '
        return None

    @abstractmethod
    def _distr_from_params(self, params: Tuple) -> torch.distributions.Distribution:
        if False:
            print('Hello World!')
        '\n        Returns a torch distribution built with the specified params\n        '
        pass

    @abstractmethod
    def _params_from_output(self, model_output: torch.Tensor) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        if False:
            i = 10
            return i + 15
        '\n        Returns the distribution parameters, obtained from the raw model outputs\n        (e.g. applies softplus or sigmoids to get parameters in the expected domains).\n        '
        pass

    @abstractmethod
    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Samples a prediction from the probability distributions defined by the specific likelihood model\n        and the parameters given in `model_output`.\n        '
        pass

    def predict_likelihood_parameters(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        '\n        Returns the distribution parameters as a single Tensor, extracted from the raw model outputs.\n        '
        params = self._params_from_output(model_output)
        if isinstance(params, torch.Tensor):
            return params
        else:
            (num_samples, n_times, n_components, n_params) = model_output.shape
            return torch.stack(params, dim=3).reshape((num_samples, n_times, n_components * n_params))

    @abstractmethod
    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            print('Hello World!')
        '\n        Generates names for the parameters of the Likelihood.\n        '
        pass

    def _likelihood_generate_components_names(self, input_series: TimeSeries, parameter_names: List[str]) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        return [f'{tgt_name}_{param_n}' for tgt_name in input_series.components for param_n in parameter_names]

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        if False:
            print('Hello World!')
        '\n        Returns the number of parameters that define the probability distribution for one single\n        target value.\n        '
        pass

    @abstractmethod
    def simplified_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns a simplified name, used to compare Likelihood and LikelihoodMixin_ instances'
        pass

    def __eq__(self, other) -> bool:
        if False:
            while True:
                i = 10
        '\n        Defines (in)equality between two likelihood objects, ignore the attributes listed in\n        self.ignore_attrs_equality or inheriting from torch.nn.Module.\n        '
        if type(other) is type(self):
            other_state = {k: v for (k, v) in other.__dict__.items() if k not in self.ignore_attrs_equality and (not isinstance(v, nn.Module))}
            self_state = {k: v for (k, v) in self.__dict__.items() if k not in self.ignore_attrs_equality and (not isinstance(v, nn.Module))}
            return other_state == self_state
        else:
            return False

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return the class and parameters of the instance in a nice format'
        cls_name = self.__class__.__name__
        init_signature = inspect.signature(self.__class__.__init__)
        params_string = ', '.join([f'{str(v)}' for (_, v) in init_signature.parameters.items() if str(v) != 'self'])
        return f'{cls_name}({params_string})'

class GaussianLikelihood(Likelihood):

    def __init__(self, prior_mu=None, prior_sigma=None, prior_strength=1.0, beta_nll=0.0):
        if False:
            return 10
        '\n        Univariate Gaussian distribution.\n\n        https://en.wikipedia.org/wiki/Normal_distribution\n\n        Instead of the pure negative log likelihood (NLL) loss, the loss function used\n        is the :math:`\\beta`-NLL loss [1]_, parameterized by ``beta_nll`` in (0, 1).\n        For ``beta_nll=0`` it is equivalent to NLL, however larger values of ``beta_nll`` can\n        mitigate issues with NLL causing effective under-sampling of poorly fit regions\n        during training. ``beta_nll=1`` provides the same gradient for the mean as the MSE loss.\n\n        - Univariate continuous distribution.\n        - Support: :math:`\\mathbb{R}`.\n        - Parameters: mean :math:`\\mu \\in \\mathbb{R}`, standard deviation :math:`\\sigma > 0`.\n\n        Parameters\n        ----------\n        prior_mu\n            mean of the prior Gaussian distribution (default: None).\n        prior_sigma\n            standard deviation (or scale) of the prior Gaussian distribution (default: None)\n        prior_strength\n            strength of the loss regularisation induced by the prior\n        beta_nll\n            The parameter :math:`0 \\leq \\beta \\leq 1` of the :math:`\\beta`-NLL loss [1]_.\n            Default: 0. (equivalent to NLL)\n\n        References\n        ----------\n        .. [1] Seitzer et al.,\n               "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks"\n               https://arxiv.org/abs/2203.09168\n        '
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.beta_nll = beta_nll
        _check_strict_positive(self.prior_sigma, 'sigma')
        self.nllloss = nn.GaussianNLLLoss(reduction='none' if self.beta_nll > 0.0 else 'mean', full=True)
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    def _nllloss(self, params_out, target):
        if False:
            for i in range(10):
                print('nop')
        (means_out, sigmas_out) = params_out
        cont_var = sigmas_out.contiguous() ** 2
        loss = self.nllloss(means_out.contiguous(), target.contiguous(), cont_var)
        if self.beta_nll > 0.0:
            loss = (loss * cont_var.detach() ** self.beta_nll).mean()
        return loss

    @property
    def _prior_params(self):
        if False:
            while True:
                i = 10
        return (self.prior_mu, self.prior_sigma)

    def _distr_from_params(self, params):
        if False:
            while True:
                i = 10
        (mu, sigma) = params
        return _Normal(mu, sigma)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        (mu, sigma) = self._params_from_output(model_output)
        return torch.normal(mu, sigma)

    @property
    def num_parameters(self) -> int:
        if False:
            i = 10
            return i + 15
        return 2

    def _params_from_output(self, model_output):
        if False:
            return 10
        mu = model_output[:, :, :, 0]
        sigma = self.softplus(model_output[:, :, :, 1])
        return (mu, sigma)

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._likelihood_generate_components_names(input_series, ['mu', 'sigma'])

    def simplified_name(self) -> str:
        if False:
            while True:
                i = 10
        return 'gaussian'

class PoissonLikelihood(Likelihood):

    def __init__(self, prior_lambda=None, prior_strength=1.0):
        if False:
            while True:
                i = 10
        '\n        Poisson distribution. Can typically be used to model event counts during time intervals, when the events\n        happen independently of the time since the last event.\n\n        https://en.wikipedia.org/wiki/Poisson_distribution\n\n        - Univariate discrete distribution\n        - Support: :math:`\\mathbb{N}_0` (natural numbers including 0).\n        - Parameter: rate :math:`\\lambda > 0`.\n\n        Parameters\n        ----------\n        prior_lambda\n            rate :math:`\\lambda` of the prior Poisson distribution (default: None)\n        prior_strength\n            strength of the loss regularisation induced by the prior\n        '
        self.prior_lambda = prior_lambda
        _check_strict_positive(self.prior_lambda, 'lambda')
        self.nllloss = nn.PoissonNLLLoss(log_input=False, full=True)
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    def _nllloss(self, params_out, target):
        if False:
            for i in range(10):
                print('nop')
        lambda_out = params_out
        return self.nllloss(lambda_out, target)

    @property
    def _prior_params(self):
        if False:
            return 10
        return (self.prior_lambda,)

    def _distr_from_params(self, params):
        if False:
            while True:
                i = 10
        lmbda = params[0]
        return _Poisson(lmbda)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        model_lambda = self._params_from_output(model_output)
        return torch.poisson(model_lambda)

    @property
    def num_parameters(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 1

    def _params_from_output(self, model_output):
        if False:
            print('Hello World!')
        lmbda = self.softplus(model_output.squeeze(dim=-1))
        return lmbda

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            while True:
                i = 10
        return self._likelihood_generate_components_names(input_series, ['lambda'])

    def simplified_name(self) -> str:
        if False:
            print('Hello World!')
        return 'poisson'

class NegativeBinomialLikelihood(Likelihood):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Negative Binomial distribution.\n\n        https://en.wikipedia.org/wiki/Negative_binomial_distribution\n\n        It does not support priors.\n\n        - Univariate discrete distribution.\n        - Support: :math:`\\mathbb{N}_0` (natural numbers including 0).\n        - Parameters: number of failures :math:`r > 0`, success probability :math:`p \\in (0, 1)`.\n\n        Behind the scenes the distribution is reparameterized so that the actual outputs of the\n        network are in terms of the mean :math:`\\mu` and shape :math:`\\alpha`.\n        '
        self.softplus = nn.Softplus()
        super().__init__()

    @property
    def _prior_params(self):
        if False:
            print('Hello World!')
        return None

    @staticmethod
    def _get_r_and_p_from_mu_and_alpha(mu, alpha):
        if False:
            return 10
        r = 1.0 / alpha
        p = r / (mu + r)
        return (r, p)

    def _distr_from_params(self, params):
        if False:
            return 10
        (mu, alpha) = params
        (r, p) = NegativeBinomialLikelihood._get_r_and_p_from_mu_and_alpha(mu, alpha)
        return _NegativeBinomial(r, p)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        (mu, alpha) = self._params_from_output(model_output)
        (r, p) = NegativeBinomialLikelihood._get_r_and_p_from_mu_and_alpha(mu, alpha)
        distr = _NegativeBinomial(r, p)
        return distr.sample()

    def predict_likelihood_parameters(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        'Overwrite the parent since the parameters are extracted in two steps.'
        (mu, alpha) = self._params_from_output(model_output)
        (r, p) = NegativeBinomialLikelihood._get_r_and_p_from_mu_and_alpha(mu, alpha)
        return torch.cat([r, p], dim=-1)

    def _params_from_output(self, model_output):
        if False:
            while True:
                i = 10
        mu = self.softplus(model_output[:, :, :, 0])
        alpha = self.softplus(model_output[:, :, :, 1])
        return (mu, alpha)

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            i = 10
            return i + 15
        return self._likelihood_generate_components_names(input_series, ['r', 'p'])

    @property
    def num_parameters(self) -> int:
        if False:
            print('Hello World!')
        return 2

    def simplified_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'negativebinomial'

class BernoulliLikelihood(Likelihood):

    def __init__(self, prior_p=None, prior_strength=1.0):
        if False:
            while True:
                i = 10
        '\n        Bernoulli distribution.\n\n        https://en.wikipedia.org/wiki/Bernoulli_distribution\n\n        - Univariate discrete distribution.\n        - Support: :math:`\\{0, 1\\}`.\n        - Parameter: probability :math:`p \\in (0, 1)`.\n\n        Parameters\n        ----------\n        prior_p\n            probability :math:`p` of the prior Bernoulli distribution (default: None)\n        prior_strength\n            strength of the loss regularisation induced by the prior\n        '
        self.prior_p = prior_p
        _check_in_open_0_1_intvl(self.prior_p, 'p')
        self.sigmoid = nn.Sigmoid()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        if False:
            print('Hello World!')
        return (self.prior_p,)

    def _distr_from_params(self, params):
        if False:
            while True:
                i = 10
        p = params[0]
        return _Bernoulli(p)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        model_p = self._params_from_output(model_output)
        return torch.bernoulli(model_p)

    @property
    def num_parameters(self) -> int:
        if False:
            return 10
        return 1

    def _params_from_output(self, model_output: torch.Tensor):
        if False:
            while True:
                i = 10
        p = self.sigmoid(model_output.squeeze(dim=-1))
        return p

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            print('Hello World!')
        return self._likelihood_generate_components_names(input_series, ['p'])

    def simplified_name(self) -> str:
        if False:
            while True:
                i = 10
        return 'bernoulli'

class BetaLikelihood(Likelihood):

    def __init__(self, prior_alpha=None, prior_beta=None, prior_strength=1.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Beta distribution.\n\n        https://en.wikipedia.org/wiki/Beta_distribution\n\n        - Univariate continuous distribution.\n        - Support: open interval :math:`(0,1)`\n        - Parameters: shape parameters :math:`\\alpha > 0` and :math:`\\beta > 0`.\n\n        Parameters\n        ----------\n        prior_alpha\n            shape parameter :math:`\\alpha` of the Beta distribution, strictly positive (default: None)\n        prior_beta\n            shape parameter :math:`\\beta` distribution, strictly positive (default: None)\n        prior_strength\n            strength of the loss regularisation induced by the prior\n        '
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        _check_strict_positive(self.prior_alpha, 'alpha')
        _check_strict_positive(self.prior_beta, 'beta')
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        if False:
            i = 10
            return i + 15
        return (self.prior_alpha, self.prior_beta)

    def _distr_from_params(self, params):
        if False:
            i = 10
            return i + 15
        (alpha, beta) = params
        return _Beta(alpha, beta)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        (alpha, beta) = self._params_from_output(model_output)
        distr = _Beta(alpha, beta)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        if False:
            while True:
                i = 10
        return 2

    def _params_from_output(self, model_output):
        if False:
            for i in range(10):
                print('nop')
        alpha = self.softplus(model_output[:, :, :, 0])
        beta = self.softplus(model_output[:, :, :, 1])
        return (alpha, beta)

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            i = 10
            return i + 15
        return self._likelihood_generate_components_names(input_series, ['alpha', 'beta'])

    def simplified_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'beta'

class CauchyLikelihood(Likelihood):

    def __init__(self, prior_xzero=None, prior_gamma=None, prior_strength=1.0):
        if False:
            print('Hello World!')
        '\n        Cauchy Distribution.\n\n        https://en.wikipedia.org/wiki/Cauchy_distribution\n\n        - Univariate continuous distribution.\n        - Support: :math:`\\mathbb{R}`.\n        - Parameters: location :math:`x_0 \\in \\mathbb{R}`, scale :math:`\\gamma > 0`.\n\n        Due to its fat tails, this distribution is typically harder to estimate,\n        and your mileage may vary. Also be aware that it typically\n        requires a large value for `num_samples` for sampling predictions.\n\n        Parameters\n        ----------\n        prior_xzero\n            location parameter :math:`x_0` of the Cauchy distribution (default: None)\n        prior_gamma\n            scale parameter :math:`\\gamma` of the Cauchy distribution, strictly positive (default: None)\n        prior_strength\n            strength of the loss regularisation induced by the prior\n        '
        self.prior_xzero = prior_xzero
        self.prior_gamma = prior_gamma
        _check_strict_positive(self.prior_gamma, 'gamma')
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        if False:
            i = 10
            return i + 15
        return (self.prior_xzero, self.prior_gamma)

    def _distr_from_params(self, params):
        if False:
            while True:
                i = 10
        (xzero, gamma) = params
        return _Cauchy(xzero, gamma)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        (xzero, gamma) = self._params_from_output(model_output)
        distr = _Cauchy(xzero, gamma)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        if False:
            print('Hello World!')
        return 2

    def _params_from_output(self, model_output):
        if False:
            print('Hello World!')
        xzero = model_output[:, :, :, 0]
        gamma = self.softplus(model_output[:, :, :, 1])
        gamma[gamma < MIN_CAUCHY_GAMMA_SAMPLING] = MIN_CAUCHY_GAMMA_SAMPLING
        return (xzero, gamma)

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            return 10
        return self._likelihood_generate_components_names(input_series, ['xzero', 'gamma'])

    def simplified_name(self) -> str:
        if False:
            print('Hello World!')
        return 'cauchy'

class ContinuousBernoulliLikelihood(Likelihood):

    def __init__(self, prior_lambda=None, prior_strength=1.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Continuous Bernoulli distribution.\n\n        https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution\n\n        - Univariate continuous distribution.\n        - Support: open interval :math:`(0, 1)`.\n        - Parameter: shape :math:`\\lambda \\in (0,1)`\n\n        Parameters\n        ----------\n        prior_lambda\n            shape :math:`\\lambda` of the prior Continuous Bernoulli distribution (default: None)\n        prior_strength\n            strength of the loss regularisation induced by the prior\n        '
        self.prior_lambda = prior_lambda
        _check_in_open_0_1_intvl(self.prior_lambda, 'lambda')
        self.sigmoid = nn.Sigmoid()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        if False:
            i = 10
            return i + 15
        return (self.prior_lambda,)

    def _distr_from_params(self, params):
        if False:
            while True:
                i = 10
        lmbda = params[0]
        return _ContinuousBernoulli(lmbda)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        model_lmbda = self._params_from_output(model_output)
        distr = _ContinuousBernoulli(model_lmbda)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        if False:
            while True:
                i = 10
        return 1

    def _params_from_output(self, model_output: torch.Tensor):
        if False:
            return 10
        lmbda = self.sigmoid(model_output.squeeze(dim=-1))
        return lmbda

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            return 10
        return self._likelihood_generate_components_names(input_series, ['lambda'])

    def simplified_name(self) -> str:
        if False:
            return 10
        return 'continuousbernoulli'

class DirichletLikelihood(Likelihood):

    def __init__(self, prior_alphas=None, prior_strength=1.0):
        if False:
            print('Hello World!')
        '\n        Dirichlet distribution.\n\n        https://en.wikipedia.org/wiki/Dirichlet_distribution\n\n        - Multivariate continuous distribution, modeling all components of a time series jointly.\n        - Support: The :math:`K`-dimensional simplex for series of dimension :math:`K`, i.e.,\n          :math:`x_1, ..., x_K \\text{ with } x_i \\in (0,1),\\; \\sum_i^K{x_i}=1`.\n        - Parameter: concentrations :math:`\\alpha_1, ..., \\alpha_K` with :math:`\\alpha_i > 0`.\n\n        Parameters\n        ----------\n        prior_alphas\n            concentrations parameters :math:`\\alpha` of the prior Dirichlet distribution.\n        prior_strength\n            strength of the loss regularisation induced by the prior\n        '
        self.prior_alphas = prior_alphas
        _check_strict_positive(self.prior_alphas)
        self.softmax = nn.Softmax(dim=2)
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        if False:
            return 10
        return (self.prior_alphas,)

    def _distr_from_params(self, params: Tuple):
        if False:
            return 10
        alphas = params[0]
        return _Dirichlet(alphas)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        alphas = self._params_from_output(model_output)
        distr = _Dirichlet(alphas)
        return distr.sample()

    def predict_likelihood_parameters(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        alphas = self._params_from_output(model_output)
        return alphas

    @property
    def num_parameters(self) -> int:
        if False:
            while True:
                i = 10
        return 1

    def _params_from_output(self, model_output):
        if False:
            return 10
        alphas = self.softmax(model_output.squeeze(dim=-1))
        return alphas

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            return 10
        return self._likelihood_generate_components_names(input_series, ['alpha'])

    def simplified_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'dirichlet'

class ExponentialLikelihood(Likelihood):

    def __init__(self, prior_lambda=None, prior_strength=1.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Exponential distribution.\n\n        https://en.wikipedia.org/wiki/Exponential_distribution\n\n        - Univariate continuous distribution.\n        - Support: :math:`\\mathbb{R}_{>0}`.\n        - Parameter: rate :math:`\\lambda > 0`.\n\n        Parameters\n        ----------\n        prior_lambda\n            rate :math:`\\lambda` of the prior exponential distribution (default: None).\n        prior_strength\n            strength of the loss regularisation induced by the prior\n        '
        self.prior_lambda = prior_lambda
        _check_strict_positive(self.prior_lambda, 'lambda')
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        if False:
            return 10
        return (self.prior_lambda,)

    def _distr_from_params(self, params: Tuple):
        if False:
            return 10
        lmbda = params[0]
        return _Exponential(lmbda)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        lmbda = self._params_from_output(model_output)
        distr = _Exponential(lmbda)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 1

    def _params_from_output(self, model_output: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        lmbda = self.softplus(model_output.squeeze(dim=-1))
        return lmbda

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            print('Hello World!')
        return self._likelihood_generate_components_names(input_series, ['lambda'])

    def simplified_name(self) -> str:
        if False:
            return 10
        return 'exponential'

class GammaLikelihood(Likelihood):

    def __init__(self, prior_alpha=None, prior_beta=None, prior_strength=1.0):
        if False:
            return 10
        '\n        Gamma distribution.\n\n        https://en.wikipedia.org/wiki/Gamma_distribution\n\n        - Univariate continuous distribution\n        - Support: :math:`\\mathbb{R}_{>0}`.\n        - Parameters: shape :math:`\\alpha > 0` and rate :math:`\\beta > 0`.\n\n        Parameters\n        ----------\n        prior_alpha\n            shape :math:`\\alpha` of the prior gamma distribution (default: None).\n        prior_beta\n            rate :math:`\\beta` of the prior gamma distribution (default: None).\n        prior_strength\n            strength of the loss regularisation induced by the prior\n        '
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        _check_strict_positive(self.prior_alpha, 'alpha')
        _check_strict_positive(self.prior_beta, 'beta')
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        if False:
            while True:
                i = 10
        return (self.prior_alpha, self.prior_beta)

    def _distr_from_params(self, params: Tuple):
        if False:
            while True:
                i = 10
        (alpha, beta) = params
        return _Gamma(alpha, beta)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        (alpha, beta) = self._params_from_output(model_output)
        distr = _Gamma(alpha, beta)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 2

    def _params_from_output(self, model_output: torch.Tensor):
        if False:
            i = 10
            return i + 15
        alpha = self.softplus(model_output[:, :, :, 0])
        beta = self.softplus(model_output[:, :, :, 1])
        return (alpha, beta)

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            return 10
        return self._likelihood_generate_components_names(input_series, ['alpha', 'beta'])

    def simplified_name(self) -> str:
        if False:
            while True:
                i = 10
        return 'gamma'

class GeometricLikelihood(Likelihood):

    def __init__(self, prior_p=None, prior_strength=1.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Geometric distribution.\n\n        https://en.wikipedia.org/wiki/Geometric_distribution\n\n        - Univariate discrete distribution\n        - Support: :math:`\\mathbb{N}_0` (natural numbers including 0).\n        - Parameter: success probability :math:`p \\in (0, 1)`.\n\n        Parameters\n        ----------\n        prior_p\n            success probability :math:`p` of the prior geometric distribution (default: None)\n        prior_strength\n            strength of the loss regularisation induced by the prior\n        '
        self.prior_p = prior_p
        _check_in_open_0_1_intvl(self.prior_p, 'p')
        self.sigmoid = nn.Sigmoid()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        if False:
            return 10
        return (self.prior_p,)

    def _distr_from_params(self, params: Tuple):
        if False:
            print('Hello World!')
        p = params[0]
        return _Geometric(p)

    def sample(self, model_output) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        p = self._params_from_output(model_output)
        distr = _Geometric(p)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        if False:
            print('Hello World!')
        return 1

    def _params_from_output(self, model_output: torch.Tensor):
        if False:
            print('Hello World!')
        p = self.sigmoid(model_output.squeeze(dim=-1))
        return p

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            i = 10
            return i + 15
        return self._likelihood_generate_components_names(input_series, ['p'])

    def simplified_name(self) -> str:
        if False:
            return 10
        return 'geometric'

class GumbelLikelihood(Likelihood):

    def __init__(self, prior_mu=None, prior_beta=None, prior_strength=1.0):
        if False:
            print('Hello World!')
        '\n        Gumbel distribution.\n\n        https://en.wikipedia.org/wiki/Gumbel_distribution\n\n        - Univariate continuous distribution\n        - Support: :math:`\\mathbb{R}`.\n        - Parameters: location :math:`\\mu \\in \\mathbb{R}` and scale :math:`\\beta > 0`.\n\n        Parameters\n        ----------\n        prior_mu\n            location :math:`\\mu` of the prior Gumbel distribution (default: None).\n        prior_beta\n            scale :math:`\\beta` of the prior Gumbel distribution (default: None).\n        prior_strength\n            strength of the loss regularisation induced by the prior\n        '
        self.prior_mu = prior_mu
        self.prior_beta = prior_beta
        _check_strict_positive(self.prior_beta)
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.prior_mu, self.prior_beta)

    def _distr_from_params(self, params: Tuple):
        if False:
            while True:
                i = 10
        (mu, beta) = params
        return _Gumbel(mu, beta)

    def sample(self, model_output) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        (mu, beta) = self._params_from_output(model_output)
        distr = _Gumbel(mu, beta)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        if False:
            print('Hello World!')
        return 2

    def _params_from_output(self, model_output: torch.Tensor):
        if False:
            while True:
                i = 10
        mu = model_output[:, :, :, 0]
        beta = self.softplus(model_output[:, :, :, 1])
        return (mu, beta)

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            while True:
                i = 10
        return self._likelihood_generate_components_names(input_series, ['mu', 'beta'])

    def simplified_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'gumbel'

class HalfNormalLikelihood(Likelihood):

    def __init__(self, prior_sigma=None, prior_strength=1.0):
        if False:
            return 10
        '\n        Half-normal distribution.\n\n        https://en.wikipedia.org/wiki/Half-normal_distribution\n\n        - Univariate continuous distribution.\n        - Support: :math:`\\mathbb{R}_{>0}`.\n        - Parameter: rate :math:`\\sigma > 0`.\n\n        Parameters\n        ----------\n        prior_sigma\n            standard deviation :math:`\\sigma` of the prior half-normal distribution (default: None).\n        prior_strength\n            strength of the loss regularisation induced by the prior\n        '
        self.prior_sigma = prior_sigma
        _check_strict_positive(self.prior_sigma, 'sigma')
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.prior_sigma,)

    def _distr_from_params(self, params: Tuple):
        if False:
            print('Hello World!')
        sigma = params[0]
        return _HalfNormal(sigma)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        sigma = self._params_from_output(model_output)
        distr = _HalfNormal(sigma)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        if False:
            i = 10
            return i + 15
        return 1

    def _params_from_output(self, model_output: torch.Tensor):
        if False:
            while True:
                i = 10
        sigma = self.softplus(model_output.squeeze(dim=-1))
        return sigma

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            i = 10
            return i + 15
        return self._likelihood_generate_components_names(input_series, ['sigma'])

    def simplified_name(self) -> str:
        if False:
            print('Hello World!')
        return 'halfnormal'

class LaplaceLikelihood(Likelihood):

    def __init__(self, prior_mu=None, prior_b=None, prior_strength=1.0):
        if False:
            i = 10
            return i + 15
        '\n        Laplace distribution.\n\n        https://en.wikipedia.org/wiki/Laplace_distribution\n\n        - Univariate continuous distribution\n        - Support: :math:`\\mathbb{R}`.\n        - Parameters: location :math:`\\mu \\in \\mathbb{R}` and scale :math:`b > 0`.\n\n        Parameters\n        ----------\n        prior_mu\n            location :math:`\\mu` of the prior Laplace distribution (default: None).\n        prior_b\n            scale :math:`b` of the prior Laplace distribution (default: None).\n        prior_strength\n            strength of the loss regularisation induced by the prior\n        '
        self.prior_mu = prior_mu
        self.prior_b = prior_b
        _check_strict_positive(self.prior_b)
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        if False:
            return 10
        return (self.prior_mu, self.prior_b)

    def _distr_from_params(self, params: Tuple):
        if False:
            return 10
        (mu, b) = params
        return _Laplace(mu, b)

    def sample(self, model_output) -> torch.Tensor:
        if False:
            print('Hello World!')
        (mu, b) = self._params_from_output(model_output)
        distr = _Laplace(mu, b)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        if False:
            return 10
        return 2

    def _params_from_output(self, model_output: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        mu = model_output[:, :, :, 0]
        b = self.softplus(model_output[:, :, :, 1])
        return (mu, b)

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            print('Hello World!')
        return self._likelihood_generate_components_names(input_series, ['mu', 'b'])

    def simplified_name(self) -> str:
        if False:
            print('Hello World!')
        return 'laplace'

class LogNormalLikelihood(Likelihood):

    def __init__(self, prior_mu=None, prior_sigma=None, prior_strength=1.0):
        if False:
            i = 10
            return i + 15
        '\n        Log-normal distribution.\n\n        https://en.wikipedia.org/wiki/Log-normal_distribution\n\n        - Univariate continuous distribution.\n        - Support: :math:`\\mathbb{R}_{>0}`.\n        - Parameters: :math:`\\mu \\in \\mathbb{R}` and :math:`\\sigma > 0`.\n\n        Parameters\n        ----------\n        prior_mu\n            parameter :math:`\\mu` of the prior log-normal distribution (default: None).\n        prior_sigma\n            parameter :math:`\\sigma` of the prior log-normal distribution (default: None)\n        prior_strength\n            strength of the loss regularisation induced by the prior\n        '
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        _check_strict_positive(self.prior_sigma, 'sigma')
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        if False:
            while True:
                i = 10
        return (self.prior_mu, self.prior_sigma)

    def _distr_from_params(self, params):
        if False:
            for i in range(10):
                print('nop')
        (mu, sigma) = params
        return _LogNormal(mu, sigma)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        (mu, sigma) = self._params_from_output(model_output)
        distr = _LogNormal(mu, sigma)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        if False:
            while True:
                i = 10
        return 2

    def _params_from_output(self, model_output):
        if False:
            for i in range(10):
                print('nop')
        mu = model_output[:, :, :, 0]
        sigma = self.softplus(model_output[:, :, :, 1])
        return (mu, sigma)

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            i = 10
            return i + 15
        return self._likelihood_generate_components_names(input_series, ['mu', 'sigma'])

    def simplified_name(self) -> str:
        if False:
            while True:
                i = 10
        return 'lognormal'

class WeibullLikelihood(Likelihood):

    def __init__(self, prior_strength=1.0):
        if False:
            print('Hello World!')
        '\n        Weibull distribution.\n\n        https://en.wikipedia.org/wiki/Weibull_distribution\n\n        - Univariate continuous distribution\n        - Support: :math:`\\mathbb{R}_{>0}`.\n        - Parameters: scale :math:`\\lambda > 0` and concentration :math:`k > 0`.\n\n        It does not support priors.\n\n        Parameters\n        ----------\n        prior_strength\n            strength of the loss regularisation induced by the prior\n        '
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        if False:
            print('Hello World!')
        return None

    def _distr_from_params(self, params: Tuple):
        if False:
            while True:
                i = 10
        (lmba, k) = params
        return _Weibull(lmba, k)

    def sample(self, model_output) -> torch.Tensor:
        if False:
            print('Hello World!')
        (lmbda, k) = self._params_from_output(model_output)
        distr = _Weibull(lmbda, k)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        if False:
            return 10
        return 2

    def _params_from_output(self, model_output: torch.Tensor):
        if False:
            while True:
                i = 10
        lmbda = self.softplus(model_output[:, :, :, 0])
        k = self.softplus(model_output[:, :, :, 1])
        return (lmbda, k)

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            while True:
                i = 10
        return self._likelihood_generate_components_names(input_series, ['lambda', 'k'])

    def simplified_name(self) -> str:
        if False:
            while True:
                i = 10
        return 'weibull'

class QuantileRegression(Likelihood):

    def __init__(self, quantiles: Optional[List[float]]=None):
        if False:
            return 10
        '\n        The "likelihood" corresponding to quantile regression.\n        It uses the Quantile Loss Metric for custom quantiles centered around q=0.5.\n\n        This class can be used as any other Likelihood objects even though it is not\n        representing the likelihood of a well defined distribution.\n\n        Parameters\n        ----------\n        quantiles\n            list of quantiles\n        '
        super().__init__()
        if quantiles is None:
            self.quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        else:
            self.quantiles = sorted(quantiles)
        _check_quantiles(self.quantiles)
        self._median_idx = self.quantiles.index(0.5)
        self.first = True
        self.quantiles_tensor = None
        self.ignore_attrs_equality = ['first', 'quantiles_tensor']

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Sample uniformly between [0, 1] (for each batch example) and return the linear interpolation between the fitted\n        quantiles closest to the sampled value.\n\n        model_output is of shape (batch_size, n_timesteps, n_components, n_quantiles)\n        '
        device = model_output.device
        (num_samples, n_timesteps, n_components, n_quantiles) = model_output.shape
        probs = torch.rand(size=(num_samples, n_timesteps, n_components, 1)).to(device)
        probas = probs.unsqueeze(-2)
        p = torch.tile(probas, (1, 1, 1, n_quantiles, 1)).transpose(4, 3)
        tquantiles = torch.tensor(self.quantiles).reshape((1, 1, 1, -1)).to(device)
        left_idx = torch.sum(p > tquantiles, dim=-1)
        right_idx = left_idx + 1
        repeat_count = [1] * n_quantiles
        repeat_count[0] = 2
        repeat_count[-1] = 2
        repeat_count = torch.tensor(repeat_count).to(device)
        shifted_output = torch.repeat_interleave(model_output, repeat_count, dim=-1)
        left_value = torch.gather(shifted_output, index=left_idx, dim=-1)
        right_value = torch.gather(shifted_output, index=right_idx, dim=-1)
        ext_quantiles = [0.0] + self.quantiles + [1.0]
        expanded_q = torch.tile(torch.tensor(ext_quantiles), left_idx.shape).to(device)
        left_q = torch.gather(expanded_q, index=left_idx, dim=-1)
        right_q = torch.gather(expanded_q, index=right_idx, dim=-1)
        weights = (probs - left_q) / (right_q - left_q)
        inter = left_value + weights * (right_value - left_value)
        return inter.squeeze(-1)

    def predict_likelihood_parameters(self, model_output: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        'Overwrite parent method since QuantileRegression is not a Likelihood per-se and\n        parameters must be extracted differently.'
        (num_samples, n_timesteps, n_components, n_quantiles) = model_output.shape
        params = model_output.reshape(num_samples, n_timesteps, n_components * n_quantiles)
        return params

    @property
    def num_parameters(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self.quantiles)

    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor):
        if False:
            i = 10
            return i + 15
        '\n        We are re-defining a custom loss (which is not a likelihood loss) compared to Likelihood\n\n        Parameters\n        ----------\n        model_output\n            must be of shape (batch_size, n_timesteps, n_target_variables, n_quantiles)\n        target\n            must be of shape (n_samples, n_timesteps, n_target_variables)\n        '
        dim_q = 3
        (batch_size, length) = model_output.shape[:2]
        device = model_output.device
        if self.first:
            raise_if_not(len(model_output.shape) == 4 and len(target.shape) == 3 and (model_output.shape[:2] == target.shape[:2]), 'mismatch between predicted and target shape')
            raise_if_not(model_output.shape[dim_q] == len(self.quantiles), 'mismatch between number of predicted quantiles and target quantiles')
            self.quantiles_tensor = torch.tensor(self.quantiles).to(device)
            self.first = False
        errors = target.unsqueeze(-1) - model_output
        losses = torch.max((self.quantiles_tensor - 1) * errors, self.quantiles_tensor * errors)
        return losses.sum(dim=dim_q).mean()

    def _distr_from_params(self, params: Tuple) -> None:
        if False:
            return 10
        return None

    def _params_from_output(self, model_output: torch.Tensor) -> None:
        if False:
            while True:
                i = 10
        return None

    def likelihood_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Each component have their own quantiles'
        return [f'{tgt_name}_q{quantile:.2f}' for tgt_name in input_series.components for quantile in self.quantiles]

    def simplified_name(self) -> str:
        if False:
            while True:
                i = 10
        return 'quantile'
' TODO\nTo make it work, we\'ll have to change our models so they optionally accept an absolute\nnumber of parameters, instead of num_parameters per component.\n\nfrom torch.distributions import MultivariateNormal as _MultivariateNormal\n    class MultivariateNormal(Likelihood):\n        def __init__(\n            self, dim: int, prior_mu=None, prior_covmat=None, prior_strength=1.0\n        ):\n            self.dim = dim\n            self.prior_mu = prior_mu\n            self.prior_covmat = prior_covmat\n            if self.prior_mu is not None:\n                raise_if_not(\n                    len(self.prior_mu) == self.dim,\n                    "The provided prior_mu must have a size matching the "\n                    "provided dimension.",\n                )\n            if self.prior_covmat is not None:\n                raise_if_not(\n                    self.prior_covmat.shape == (self.dim, self.dim),\n                    "The provided prior on the covariaance "\n                    "matrix must have size (dim, dim).",\n                )\n                _check_strict_positive(self.prior_covmat.flatten(), "covariance matrix")\n\n            self.softplus = nn.Softplus()\n            super().__init__(prior_strength)\n\n        @property\n        def _prior_params(self):\n            return self.prior_mu, self.prior_covmat\n\n        def _distr_from_params(self, params: Tuple):\n            mu, covmat = params\n            return _MultivariateNormal(mu, covmat)\n\n        def sample(self, model_output: torch.Tensor):\n            mu, covmat = self._params_from_output(model_output)\n            distr = _MultivariateNormal(mu, covmat)\n            return distr.sample()\n\n        @property\n        def num_parameters(self) -> int:\n            return int(self.dim + (self.dim ** 2 - self.dim) / 2)\n\n        def _params_from_output(self, model_output: torch.Tensor):\n            device = model_output.device\n            mu = model_output[:, :, : self.dim]\n            covmat_coefs = self.softplus(model_output[:, :, self.dim :])\n\n            print("model output: {}".format(model_output.shape))\n\n            # build covariance matrix\n            covmat = torch.zeros(\n                (model_output.shape[0], model_output.shape[1], self.dim, self.dim)\n            ).to(device)\n            tril_indices = torch.tril_indices(\n                row=self.dim, col=self.dim, offset=1, device=device\n            )\n            covmat[tril_indices[0], tril_indices[1]] = covmat_coefs\n            covmat[tril_indices[1], tril_indices[0]] = covmat_coefs\n            covmat[range(self.dim), range(self.dim)] = 1.0\n'