import math
import warnings
import torch
import pyro
from pyro import poutine
from pyro.contrib.oed.search import Search
from pyro.contrib.util import lexpand
from pyro.infer import SVI, EmpiricalMarginal, Importance
from pyro.infer.autoguide.utils import mean_field_entropy
from pyro.util import torch_isinf, torch_isnan
__all__ = ['laplace_eig', 'vi_eig', 'nmc_eig', 'donsker_varadhan_eig', 'posterior_eig', 'marginal_eig', 'lfire_eig', 'vnmc_eig']

def laplace_eig(model, design, observation_labels, target_labels, guide, loss, optim, num_steps, final_num_samples, y_dist=None, eig=True, **prior_entropy_kwargs):
    if False:
        print('Hello World!')
    '\n    Estimates the expected information gain (EIG) by making repeated Laplace approximations to the posterior.\n\n    :param function model: Pyro stochastic function taking `design` as only argument.\n    :param torch.Tensor design: Tensor of possible designs.\n    :param list observation_labels: labels of sample sites to be regarded as observables.\n    :param list target_labels: labels of sample sites to be regarded as latent variables of interest, i.e. the sites\n        that we wish to gain information about.\n    :param function guide: Pyro stochastic function corresponding to `model`.\n    :param loss: a Pyro loss such as `pyro.infer.Trace_ELBO().differentiable_loss`.\n    :param optim: optimizer for the loss\n    :param int num_steps: Number of gradient steps to take per sampled pseudo-observation.\n    :param int final_num_samples: Number of `y` samples (pseudo-observations) to take.\n    :param y_dist: Distribution to sample `y` from- if `None` we use the Bayesian marginal distribution.\n    :param bool eig: Whether to compute the EIG or the average posterior entropy (APE). The EIG is given by\n        `EIG = prior entropy - APE`. If `True`, the prior entropy will be estimated analytically,\n        or by Monte Carlo as appropriate for the `model`. If `False` the APE is returned.\n    :param dict prior_entropy_kwargs: parameters for estimating the prior entropy: `num_prior_samples` indicating the\n        number of samples for a MC estimate of prior entropy, and `mean_field` indicating if an analytic form for\n        a mean-field prior should be tried.\n    :return: EIG estimate, optionally includes full optimization history\n    :rtype: torch.Tensor\n    '
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if target_labels is not None and isinstance(target_labels, str):
        target_labels = [target_labels]
    ape = _laplace_vi_ape(model, design, observation_labels, target_labels, guide, loss, optim, num_steps, final_num_samples, y_dist=y_dist)
    return _eig_from_ape(model, design, target_labels, ape, eig, prior_entropy_kwargs)

def _eig_from_ape(model, design, target_labels, ape, eig, prior_entropy_kwargs):
    if False:
        i = 10
        return i + 15
    mean_field = prior_entropy_kwargs.get('mean_field', True)
    if eig:
        if mean_field:
            try:
                prior_entropy = mean_field_entropy(model, [design], whitelist=target_labels)
            except NotImplemented:
                prior_entropy = monte_carlo_entropy(model, design, target_labels, **prior_entropy_kwargs)
        else:
            prior_entropy = monte_carlo_entropy(model, design, target_labels, **prior_entropy_kwargs)
        return prior_entropy - ape
    else:
        return ape

def _laplace_vi_ape(model, design, observation_labels, target_labels, guide, loss, optim, num_steps, final_num_samples, y_dist=None):
    if False:
        for i in range(10):
            print('nop')

    def posterior_entropy(y_dist, design):
        if False:
            for i in range(10):
                print('nop')
        y = pyro.sample('conditioning_y', y_dist)
        y_dict = {label: y[i, ...] for (i, label) in enumerate(observation_labels)}
        conditioned_model = pyro.condition(model, data=y_dict)
        guide.train()
        svi = SVI(conditioned_model, guide=guide, loss=loss, optim=optim)
        with poutine.block():
            for _ in range(num_steps):
                svi.step(design)
        with poutine.block():
            final_loss = loss(conditioned_model, guide, design)
            guide.finalize(final_loss, target_labels)
            entropy = mean_field_entropy(guide, [design], whitelist=target_labels)
        return entropy
    if y_dist is None:
        y_dist = EmpiricalMarginal(Importance(model, num_samples=final_num_samples).run(design), sites=observation_labels)
    loss_dist = EmpiricalMarginal(Search(posterior_entropy).run(y_dist, design))
    ape = loss_dist.mean
    return ape

def vi_eig(model, design, observation_labels, target_labels, vi_parameters, is_parameters, y_dist=None, eig=True, **prior_entropy_kwargs):
    if False:
        return 10
    '.. deprecated:: 0.4.1\n        Use `posterior_eig` instead.\n\n    Estimates the expected information gain (EIG) using variational inference (VI).\n\n    The APE is defined as\n\n        :math:`APE(d)=E_{Y\\sim p(y|\\theta, d)}[H(p(\\theta|Y, d))]`\n\n    where :math:`H[p(x)]` is the `differential entropy\n    <https://en.wikipedia.org/wiki/Differential_entropy>`_.\n    The APE is related to expected information gain (EIG) by the equation\n\n        :math:`EIG(d)=H[p(\\theta)]-APE(d)`\n\n    in particular, minimising the APE is equivalent to maximising EIG.\n\n    :param function model: A pyro model accepting `design` as only argument.\n    :param torch.Tensor design: Tensor representation of design\n    :param list observation_labels: A subset of the sample sites\n        present in `model`. These sites are regarded as future observations\n        and other sites are regarded as latent variables over which a\n        posterior is to be inferred.\n    :param list target_labels: A subset of the sample sites over which the posterior\n        entropy is to be measured.\n    :param dict vi_parameters: Variational inference parameters which should include:\n        `optim`: an instance of :class:`pyro.Optim`, `guide`: a guide function\n        compatible with `model`, `num_steps`: the number of VI steps to make,\n        and `loss`: the loss function to use for VI\n    :param dict is_parameters: Importance sampling parameters for the\n        marginal distribution of :math:`Y`. May include `num_samples`: the number\n        of samples to draw from the marginal.\n    :param pyro.distributions.Distribution y_dist: (optional) the distribution\n        assumed for the response variable :math:`Y`\n    :param bool eig: Whether to compute the EIG or the average posterior entropy (APE). The EIG is given by\n        `EIG = prior entropy - APE`. If `True`, the prior entropy will be estimated analytically,\n        or by Monte Carlo as appropriate for the `model`. If `False` the APE is returned.\n    :param dict prior_entropy_kwargs: parameters for estimating the prior entropy: `num_prior_samples` indicating the\n        number of samples for a MC estimate of prior entropy, and `mean_field` indicating if an analytic form for\n        a mean-field prior should be tried.\n    :return: EIG estimate, optionally includes full optimization history\n    :rtype: torch.Tensor\n\n    '
    warnings.warn('`vi_eig` is deprecated in favour of the amortized version: `posterior_eig`.', DeprecationWarning)
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if target_labels is not None and isinstance(target_labels, str):
        target_labels = [target_labels]
    ape = _vi_ape(model, design, observation_labels, target_labels, vi_parameters, is_parameters, y_dist=y_dist)
    return _eig_from_ape(model, design, target_labels, ape, eig, prior_entropy_kwargs)

def _vi_ape(model, design, observation_labels, target_labels, vi_parameters, is_parameters, y_dist=None):
    if False:
        print('Hello World!')
    svi_num_steps = vi_parameters.pop('num_steps')

    def posterior_entropy(y_dist, design):
        if False:
            while True:
                i = 10
        y = pyro.sample('conditioning_y', y_dist)
        y_dict = {label: y[i, ...] for (i, label) in enumerate(observation_labels)}
        conditioned_model = pyro.condition(model, data=y_dict)
        svi = SVI(conditioned_model, **vi_parameters)
        with poutine.block():
            for _ in range(svi_num_steps):
                svi.step(design)
        with poutine.block():
            guide = vi_parameters['guide']
            entropy = mean_field_entropy(guide, [design], whitelist=target_labels)
        return entropy
    if y_dist is None:
        y_dist = EmpiricalMarginal(Importance(model, **is_parameters).run(design), sites=observation_labels)
    loss_dist = EmpiricalMarginal(Search(posterior_entropy).run(y_dist, design))
    loss = loss_dist.mean
    return loss

def nmc_eig(model, design, observation_labels, target_labels=None, N=100, M=10, M_prime=None, independent_priors=False):
    if False:
        for i in range(10):
            print('nop')
    "Nested Monte Carlo estimate of the expected information\n    gain (EIG). The estimate is, when there are not any random effects,\n\n    .. math::\n\n        \\frac{1}{N}\\sum_{n=1}^N \\log p(y_n | \\theta_n, d) -\n        \\frac{1}{N}\\sum_{n=1}^N \\log \\left(\\frac{1}{M}\\sum_{m=1}^M p(y_n | \\theta_m, d)\\right)\n\n    where :math:`\\theta_n, y_n \\sim p(\\theta, y | d)` and :math:`\\theta_m \\sim p(\\theta)`.\n    The estimate in the presence of random effects is\n\n    .. math::\n\n        \\frac{1}{N}\\sum_{n=1}^N  \\log \\left(\\frac{1}{M'}\\sum_{m=1}^{M'}\n        p(y_n | \\theta_n, \\widetilde{\\theta}_{nm}, d)\\right)-\n        \\frac{1}{N}\\sum_{n=1}^N \\log \\left(\\frac{1}{M}\\sum_{m=1}^{M}\n        p(y_n | \\theta_m, \\widetilde{\\theta}_{m}, d)\\right)\n\n    where :math:`\\widetilde{\\theta}` are the random effects with\n    :math:`\\widetilde{\\theta}_{nm} \\sim p(\\widetilde{\\theta}|\\theta=\\theta_n)` and\n    :math:`\\theta_m,\\widetilde{\\theta}_m \\sim p(\\theta,\\widetilde{\\theta})`.\n    The latter form is used when `M_prime != None`.\n\n    :param function model: A pyro model accepting `design` as only argument.\n    :param torch.Tensor design: Tensor representation of design\n    :param list observation_labels: A subset of the sample sites\n        present in `model`. These sites are regarded as future observations\n        and other sites are regarded as latent variables over which a\n        posterior is to be inferred.\n    :param list target_labels: A subset of the sample sites over which the posterior\n        entropy is to be measured.\n    :param int N: Number of outer expectation samples.\n    :param int M: Number of inner expectation samples for `p(y|d)`.\n    :param int M_prime: Number of samples for `p(y | theta, d)` if required.\n    :param bool independent_priors: Only used when `M_prime` is not `None`. Indicates whether the prior distributions\n        for the target variables and the nuisance variables are independent. In this case, it is not necessary to\n        sample the targets conditional on the nuisance variables.\n    :return: EIG estimate, optionally includes full optimization history\n    :rtype: torch.Tensor\n    "
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    expanded_design = lexpand(design, N)
    trace = poutine.trace(model).get_trace(expanded_design)
    trace.compute_log_prob()
    if M_prime is not None:
        y_dict = {l: lexpand(trace.nodes[l]['value'], M_prime) for l in observation_labels}
        theta_dict = {l: lexpand(trace.nodes[l]['value'], M_prime) for l in target_labels}
        theta_dict.update(y_dict)
        conditional_model = pyro.condition(model, data=theta_dict)
        if independent_priors:
            reexpanded_design = lexpand(design, M_prime, 1)
        else:
            reexpanded_design = lexpand(design, M_prime, N)
        retrace = poutine.trace(conditional_model).get_trace(reexpanded_design)
        retrace.compute_log_prob()
        conditional_lp = sum((retrace.nodes[l]['log_prob'] for l in observation_labels)).logsumexp(0) - math.log(M_prime)
    else:
        conditional_lp = sum((trace.nodes[l]['log_prob'] for l in observation_labels))
    y_dict = {l: lexpand(trace.nodes[l]['value'], M) for l in observation_labels}
    conditional_model = pyro.condition(model, data=y_dict)
    reexpanded_design = lexpand(design, M, 1)
    retrace = poutine.trace(conditional_model).get_trace(reexpanded_design)
    retrace.compute_log_prob()
    marginal_lp = sum((retrace.nodes[l]['log_prob'] for l in observation_labels)).logsumexp(0) - math.log(M)
    terms = conditional_lp - marginal_lp
    nonnan = (~torch.isnan(terms)).sum(0).type_as(terms)
    terms[torch.isnan(terms)] = 0.0
    return terms.sum(0) / nonnan

def donsker_varadhan_eig(model, design, observation_labels, target_labels, num_samples, num_steps, T, optim, return_history=False, final_design=None, final_num_samples=None):
    if False:
        while True:
            i = 10
    '\n    Donsker-Varadhan estimate of the expected information gain (EIG).\n\n    The Donsker-Varadhan representation of EIG is\n\n    .. math::\n\n        \\sup_T E_{p(y, \\theta | d)}[T(y, \\theta)] - \\log E_{p(y|d)p(\\theta)}[\\exp(T(\\bar{y}, \\bar{\\theta}))]\n\n    where :math:`T` is any (measurable) function.\n\n    This methods optimises the loss function over a pre-specified class of\n    functions `T`.\n\n    :param function model: A pyro model accepting `design` as only argument.\n    :param torch.Tensor design: Tensor representation of design\n    :param list observation_labels: A subset of the sample sites\n        present in `model`. These sites are regarded as future observations\n        and other sites are regarded as latent variables over which a\n        posterior is to be inferred.\n    :param list target_labels: A subset of the sample sites over which the posterior\n        entropy is to be measured.\n    :param int num_samples: Number of samples per iteration.\n    :param int num_steps: Number of optimization steps.\n    :param function or torch.nn.Module T: optimisable function `T` for use in the\n        Donsker-Varadhan loss function.\n    :param pyro.optim.Optim optim: Optimiser to use.\n    :param bool return_history: If `True`, also returns a tensor giving the loss function\n        at each step of the optimization.\n    :param torch.Tensor final_design: The final design tensor to evaluate at. If `None`, uses\n        `design`.\n    :param int final_num_samples: The number of samples to use at the final evaluation, If `None,\n        uses `num_samples`.\n    :return: EIG estimate, optionally includes full optimization history\n    :rtype: torch.Tensor or tuple\n    '
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss = _donsker_varadhan_loss(model, T, observation_labels, target_labels)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history, final_design, final_num_samples)

def posterior_eig(model, design, observation_labels, target_labels, num_samples, num_steps, guide, optim, return_history=False, final_design=None, final_num_samples=None, eig=True, prior_entropy_kwargs={}, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Posterior estimate of expected information gain (EIG) computed from the average posterior entropy (APE)\n    using :math:`EIG(d) = H[p(\\theta)] - APE(d)`. See [1] for full details.\n\n    The posterior representation of APE is\n\n        :math:`\\sup_{q}\\ E_{p(y, \\theta | d)}[\\log q(\\theta | y, d)]`\n\n    where :math:`q` is any distribution on :math:`\\theta`.\n\n    This method optimises the loss over a given `guide` family representing :math:`q`.\n\n    [1] Foster, Adam, et al. "Variational Bayesian Optimal Experimental Design." arXiv preprint arXiv:1903.05480 (2019).\n\n    :param function model: A pyro model accepting `design` as only argument.\n    :param torch.Tensor design: Tensor representation of design\n    :param list observation_labels: A subset of the sample sites\n        present in `model`. These sites are regarded as future observations\n        and other sites are regarded as latent variables over which a\n        posterior is to be inferred.\n    :param list target_labels: A subset of the sample sites over which the posterior\n        entropy is to be measured.\n    :param int num_samples: Number of samples per iteration.\n    :param int num_steps: Number of optimization steps.\n    :param function guide: guide family for use in the (implicit) posterior estimation.\n        The parameters of `guide` are optimised to maximise the posterior\n        objective.\n    :param pyro.optim.Optim optim: Optimiser to use.\n    :param bool return_history: If `True`, also returns a tensor giving the loss function\n        at each step of the optimization.\n    :param torch.Tensor final_design: The final design tensor to evaluate at. If `None`, uses\n        `design`.\n    :param int final_num_samples: The number of samples to use at the final evaluation, If `None,\n        uses `num_samples`.\n    :param bool eig: Whether to compute the EIG or the average posterior entropy (APE). The EIG is given by\n        `EIG = prior entropy - APE`. If `True`, the prior entropy will be estimated analytically,\n        or by Monte Carlo as appropriate for the `model`. If `False` the APE is returned.\n    :param dict prior_entropy_kwargs: parameters for estimating the prior entropy: `num_prior_samples` indicating the\n        number of samples for a MC estimate of prior entropy, and `mean_field` indicating if an analytic form for\n        a mean-field prior should be tried.\n    :return: EIG estimate, optionally includes full optimization history\n    :rtype: torch.Tensor or tuple\n    '
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    ape = _posterior_ape(model, design, observation_labels, target_labels, num_samples, num_steps, guide, optim, *args, return_history=return_history, final_design=final_design, final_num_samples=final_num_samples, **kwargs)
    return _eig_from_ape(model, design, target_labels, ape, eig, prior_entropy_kwargs)

def _posterior_ape(model, design, observation_labels, target_labels, num_samples, num_steps, guide, optim, return_history=False, final_design=None, final_num_samples=None, *args, **kwargs):
    if False:
        while True:
            i = 10
    loss = _posterior_loss(model, guide, observation_labels, target_labels, *args, **kwargs)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history, final_design, final_num_samples)

def marginal_eig(model, design, observation_labels, target_labels, num_samples, num_steps, guide, optim, return_history=False, final_design=None, final_num_samples=None):
    if False:
        return 10
    'Estimate EIG by estimating the marginal entropy :math:`p(y|d)`. See [1] for full details.\n\n    The marginal representation of EIG is\n\n        :math:`\\inf_{q}\\ E_{p(y, \\theta | d)}\\left[\\log \\frac{p(y | \\theta, d)}{q(y | d)} \\right]`\n\n    where :math:`q` is any distribution on :math:`y`. A variational family for :math:`q` is specified in the `guide`.\n\n    .. warning :: This method does **not** estimate the correct quantity in the presence of random effects.\n\n    [1] Foster, Adam, et al. "Variational Bayesian Optimal Experimental Design." arXiv preprint arXiv:1903.05480 (2019).\n\n    :param function model: A pyro model accepting `design` as only argument.\n    :param torch.Tensor design: Tensor representation of design\n    :param list observation_labels: A subset of the sample sites\n        present in `model`. These sites are regarded as future observations\n        and other sites are regarded as latent variables over which a\n        posterior is to be inferred.\n    :param list target_labels: A subset of the sample sites over which the posterior\n        entropy is to be measured.\n    :param int num_samples: Number of samples per iteration.\n    :param int num_steps: Number of optimization steps.\n    :param function guide: guide family for use in the marginal estimation.\n        The parameters of `guide` are optimised to maximise the log-likelihood objective.\n    :param pyro.optim.Optim optim: Optimiser to use.\n    :param bool return_history: If `True`, also returns a tensor giving the loss function\n        at each step of the optimization.\n    :param torch.Tensor final_design: The final design tensor to evaluate at. If `None`, uses\n        `design`.\n    :param int final_num_samples: The number of samples to use at the final evaluation, If `None,\n        uses `num_samples`.\n    :return: EIG estimate, optionally includes full optimization history\n    :rtype: torch.Tensor or tuple\n    '
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss = _marginal_loss(model, guide, observation_labels, target_labels)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history, final_design, final_num_samples)

def marginal_likelihood_eig(model, design, observation_labels, target_labels, num_samples, num_steps, marginal_guide, cond_guide, optim, return_history=False, final_design=None, final_num_samples=None):
    if False:
        for i in range(10):
            print('nop')
    'Estimates EIG by estimating the marginal entropy, that of :math:`p(y|d)`,\n    *and* the conditional entropy, of :math:`p(y|\\theta, d)`, both via Gibbs\' Inequality. See [1] for full details.\n\n    [1] Foster, Adam, et al. "Variational Bayesian Optimal Experimental Design." arXiv preprint arXiv:1903.05480 (2019).\n\n    :param function model: A pyro model accepting `design` as only argument.\n    :param torch.Tensor design: Tensor representation of design\n    :param list observation_labels: A subset of the sample sites\n        present in `model`. These sites are regarded as future observations\n        and other sites are regarded as latent variables over which a\n        posterior is to be inferred.\n    :param list target_labels: A subset of the sample sites over which the posterior\n        entropy is to be measured.\n    :param int num_samples: Number of samples per iteration.\n    :param int num_steps: Number of optimization steps.\n    :param function marginal_guide: guide family for use in the marginal estimation.\n        The parameters of `guide` are optimised to maximise the log-likelihood objective.\n    :param function cond_guide: guide family for use in the likelihood (conditional) estimation.\n        The parameters of `guide` are optimised to maximise the log-likelihood objective.\n    :param pyro.optim.Optim optim: Optimiser to use.\n    :param bool return_history: If `True`, also returns a tensor giving the loss function\n        at each step of the optimization.\n    :param torch.Tensor final_design: The final design tensor to evaluate at. If `None`, uses\n        `design`.\n    :param int final_num_samples: The number of samples to use at the final evaluation, If `None,\n        uses `num_samples`.\n    :return: EIG estimate, optionally includes full optimization history\n    :rtype: torch.Tensor or tuple\n    '
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss = _marginal_likelihood_loss(model, marginal_guide, cond_guide, observation_labels, target_labels)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history, final_design, final_num_samples)

def lfire_eig(model, design, observation_labels, target_labels, num_y_samples, num_theta_samples, num_steps, classifier, optim, return_history=False, final_design=None, final_num_samples=None):
    if False:
        for i in range(10):
            print('nop')
    'Estimates the EIG using the method of Likelihood-Free Inference by Ratio Estimation (LFIRE) as in [1].\n    LFIRE is run separately for several samples of :math:`\\theta`.\n\n    [1] Kleinegesse, Steven, and Michael Gutmann. "Efficient Bayesian Experimental Design for Implicit Models."\n    arXiv preprint arXiv:1810.09912 (2018).\n\n    :param function model: A pyro model accepting `design` as only argument.\n    :param torch.Tensor design: Tensor representation of design\n    :param list observation_labels: A subset of the sample sites\n        present in `model`. These sites are regarded as future observations\n        and other sites are regarded as latent variables over which a\n        posterior is to be inferred.\n    :param list target_labels: A subset of the sample sites over which the posterior\n        entropy is to be measured.\n    :param int num_y_samples: Number of samples to take in :math:`y` for each :math:`\\theta`.\n    :param: int num_theta_samples: Number of initial samples in :math:`\\theta` to take. The likelihood ratio\n                                   is estimated by LFIRE for each sample.\n    :param int num_steps: Number of optimization steps.\n    :param function classifier: a Pytorch or Pyro classifier used to distinguish between samples of :math:`y` under\n                                :math:`p(y|d)` and samples under :math:`p(y|\\theta,d)` for some :math:`\\theta`.\n    :param pyro.optim.Optim optim: Optimiser to use.\n    :param bool return_history: If `True`, also returns a tensor giving the loss function\n        at each step of the optimization.\n    :param torch.Tensor final_design: The final design tensor to evaluate at. If `None`, uses\n        `design`.\n    :param int final_num_samples: The number of samples to use at the final evaluation, If `None,\n        uses `num_samples`.\n    :return: EIG estimate, optionally includes full optimization history\n    :rtype: torch.Tensor or tuple\n    '
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    expanded_design = lexpand(design, num_theta_samples)
    trace = poutine.trace(model).get_trace(expanded_design)
    theta_dict = {l: trace.nodes[l]['value'] for l in target_labels}
    cond_model = pyro.condition(model, data=theta_dict)
    loss = _lfire_loss(model, cond_model, classifier, observation_labels, target_labels)
    out = opt_eig_ape_loss(expanded_design, loss, num_y_samples, num_steps, optim, return_history, final_design, final_num_samples)
    if return_history:
        return (out[0], out[1].sum(0) / num_theta_samples)
    else:
        return out.sum(0) / num_theta_samples

def vnmc_eig(model, design, observation_labels, target_labels, num_samples, num_steps, guide, optim, return_history=False, final_design=None, final_num_samples=None):
    if False:
        print('Hello World!')
    'Estimates the EIG using Variational Nested Monte Carlo (VNMC). The VNMC estimate [1] is\n\n    .. math::\n\n        \\frac{1}{N}\\sum_{n=1}^N \\left[ \\log p(y_n | \\theta_n, d) -\n         \\log \\left(\\frac{1}{M}\\sum_{m=1}^M \\frac{p(\\theta_{mn})p(y_n | \\theta_{mn}, d)}\n         {q(\\theta_{mn} | y_n)} \\right) \\right]\n\n    where :math:`q(\\theta | y)` is the learned variational posterior approximation and\n    :math:`\\theta_n, y_n \\sim p(\\theta, y | d)`, :math:`\\theta_{mn} \\sim q(\\theta|y=y_n)`.\n\n    As :math:`N \\to \\infty` this is an upper bound on EIG. We minimise this upper bound by stochastic gradient\n    descent.\n\n    .. warning :: This method cannot be used in the presence of random effects.\n\n    [1] Foster, Adam, et al. "Variational Bayesian Optimal Experimental Design." arXiv preprint arXiv:1903.05480 (2019).\n\n    :param function model: A pyro model accepting `design` as only argument.\n    :param torch.Tensor design: Tensor representation of design\n    :param list observation_labels: A subset of the sample sites\n        present in `model`. These sites are regarded as future observations\n        and other sites are regarded as latent variables over which a\n        posterior is to be inferred.\n    :param list target_labels: A subset of the sample sites over which the posterior\n        entropy is to be measured.\n    :param tuple num_samples: Number of (:math:`N, M`) samples per iteration.\n    :param int num_steps: Number of optimization steps.\n    :param function guide: guide family for use in the posterior estimation.\n        The parameters of `guide` are optimised to minimise the VNMC upper bound.\n    :param pyro.optim.Optim optim: Optimiser to use.\n    :param bool return_history: If `True`, also returns a tensor giving the loss function\n        at each step of the optimization.\n    :param torch.Tensor final_design: The final design tensor to evaluate at. If `None`, uses\n        `design`.\n    :param tuple final_num_samples: The number of (:math:`N, M`) samples to use at the final evaluation, If `None,\n        uses `num_samples`.\n    :return: EIG estimate, optionally includes full optimization history\n    :rtype: torch.Tensor or tuple\n    '
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss = _vnmc_eig_loss(model, guide, observation_labels, target_labels)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history, final_design, final_num_samples)

def opt_eig_ape_loss(design, loss_fn, num_samples, num_steps, optim, return_history=False, final_design=None, final_num_samples=None):
    if False:
        for i in range(10):
            print('nop')
    if final_design is None:
        final_design = design
    if final_num_samples is None:
        final_num_samples = num_samples
    params = None
    history = []
    for step in range(num_steps):
        if params is not None:
            pyro.infer.util.zero_grads(params)
        with poutine.trace(param_only=True) as param_capture:
            (agg_loss, loss) = loss_fn(design, num_samples, evaluation=return_history)
        params = set((site['value'].unconstrained() for site in param_capture.trace.nodes.values()))
        if torch.isnan(agg_loss):
            raise ArithmeticError('Encountered NaN loss in opt_eig_ape_loss')
        agg_loss.backward(retain_graph=True)
        if return_history:
            history.append(loss)
        optim(params)
        try:
            optim.step()
        except AttributeError:
            pass
    (_, loss) = loss_fn(final_design, final_num_samples, evaluation=True)
    if return_history:
        return (torch.stack(history), loss)
    else:
        return loss

def monte_carlo_entropy(model, design, target_labels, num_prior_samples=1000):
    if False:
        print('Hello World!')
    'Computes a Monte Carlo estimate of the entropy of `model` assuming that each of sites in `target_labels` is\n    independent and the entropy is to be computed for that subset of sites only.\n    '
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    expanded_design = lexpand(design, num_prior_samples)
    trace = pyro.poutine.trace(model).get_trace(expanded_design)
    trace.compute_log_prob()
    lp = sum((trace.nodes[l]['log_prob'] for l in target_labels))
    return -lp.sum(0) / num_prior_samples

def _donsker_varadhan_loss(model, T, observation_labels, target_labels):
    if False:
        print('Hello World!')
    'DV loss: to evaluate directly use `donsker_varadhan_eig` setting `num_steps=0`.'
    ewma_log = EwmaLog(alpha=0.9)

    def loss_fn(design, num_particles, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            pyro.module('T', T)
        except AssertionError:
            pass
        expanded_design = lexpand(design, num_particles)
        unshuffled_trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: unshuffled_trace.nodes[l]['value'] for l in observation_labels}
        conditional_model = pyro.condition(model, data=y_dict)
        shuffled_trace = poutine.trace(conditional_model).get_trace(expanded_design)
        T_joint = T(expanded_design, unshuffled_trace, observation_labels, target_labels)
        T_independent = T(expanded_design, shuffled_trace, observation_labels, target_labels)
        joint_expectation = T_joint.sum(0) / num_particles
        A = T_independent - math.log(num_particles)
        (s, _) = torch.max(A, dim=0)
        independent_expectation = s + ewma_log((A - s).exp().sum(dim=0), s)
        loss = joint_expectation - independent_expectation
        agg_loss = -loss.sum()
        return (agg_loss, loss)
    return loss_fn

def _posterior_loss(model, guide, observation_labels, target_labels, analytic_entropy=False):
    if False:
        return 10
    'Posterior loss: to evaluate directly use `posterior_eig` setting `num_steps=0`, `eig=False`.'

    def loss_fn(design, num_particles, evaluation=False, **kwargs):
        if False:
            return 10
        expanded_design = lexpand(design, num_particles)
        trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: trace.nodes[l]['value'] for l in observation_labels}
        theta_dict = {l: trace.nodes[l]['value'] for l in target_labels}
        conditional_guide = pyro.condition(guide, data=theta_dict)
        cond_trace = poutine.trace(conditional_guide).get_trace(y_dict, expanded_design, observation_labels, target_labels)
        cond_trace.compute_log_prob()
        if evaluation and analytic_entropy:
            loss = mean_field_entropy(guide, [y_dict, expanded_design, observation_labels, target_labels], whitelist=target_labels).sum(0) / num_particles
            agg_loss = loss.sum()
        else:
            terms = -sum((cond_trace.nodes[l]['log_prob'] for l in target_labels))
            (agg_loss, loss) = _safe_mean_terms(terms)
        return (agg_loss, loss)
    return loss_fn

def _marginal_loss(model, guide, observation_labels, target_labels):
    if False:
        return 10
    'Marginal loss: to evaluate directly use `marginal_eig` setting `num_steps=0`.'

    def loss_fn(design, num_particles, evaluation=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        expanded_design = lexpand(design, num_particles)
        trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: trace.nodes[l]['value'] for l in observation_labels}
        conditional_guide = pyro.condition(guide, data=y_dict)
        cond_trace = poutine.trace(conditional_guide).get_trace(expanded_design, observation_labels, target_labels)
        cond_trace.compute_log_prob()
        terms = -sum((cond_trace.nodes[l]['log_prob'] for l in observation_labels))
        if evaluation:
            trace.compute_log_prob()
            terms += sum((trace.nodes[l]['log_prob'] for l in observation_labels))
        return _safe_mean_terms(terms)
    return loss_fn

def _marginal_likelihood_loss(model, marginal_guide, likelihood_guide, observation_labels, target_labels):
    if False:
        return 10
    'Marginal_likelihood loss: to evaluate directly use `marginal_likelihood_eig` setting `num_steps=0`.'

    def loss_fn(design, num_particles, evaluation=False, **kwargs):
        if False:
            i = 10
            return i + 15
        expanded_design = lexpand(design, num_particles)
        trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: trace.nodes[l]['value'] for l in observation_labels}
        theta_dict = {l: trace.nodes[l]['value'] for l in target_labels}
        qyd = pyro.condition(marginal_guide, data=y_dict)
        marginal_trace = poutine.trace(qyd).get_trace(expanded_design, observation_labels, target_labels)
        marginal_trace.compute_log_prob()
        qythetad = pyro.condition(likelihood_guide, data=y_dict)
        cond_trace = poutine.trace(qythetad).get_trace(theta_dict, expanded_design, observation_labels, target_labels)
        cond_trace.compute_log_prob()
        terms = -sum((marginal_trace.nodes[l]['log_prob'] for l in observation_labels))
        if evaluation:
            terms += sum((cond_trace.nodes[l]['log_prob'] for l in observation_labels))
        else:
            terms -= sum((cond_trace.nodes[l]['log_prob'] for l in observation_labels))
        return _safe_mean_terms(terms)
    return loss_fn

def _lfire_loss(model_marginal, model_conditional, h, observation_labels, target_labels):
    if False:
        while True:
            i = 10
    'LFIRE loss: to evaluate directly use `lfire_eig` setting `num_steps=0`.'

    def loss_fn(design, num_particles, evaluation=False, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            pyro.module('h', h)
        except AssertionError:
            pass
        expanded_design = lexpand(design, num_particles)
        model_conditional_trace = poutine.trace(model_conditional).get_trace(expanded_design)
        if not evaluation:
            model_marginal_trace = poutine.trace(model_marginal).get_trace(expanded_design)
            h_joint = h(expanded_design, model_conditional_trace, observation_labels, target_labels)
            h_independent = h(expanded_design, model_marginal_trace, observation_labels, target_labels)
            terms = torch.nn.functional.softplus(-h_joint) + torch.nn.functional.softplus(h_independent)
            return _safe_mean_terms(terms)
        else:
            h_joint = h(expanded_design, model_conditional_trace, observation_labels, target_labels)
            return _safe_mean_terms(h_joint)
    return loss_fn

def _vnmc_eig_loss(model, guide, observation_labels, target_labels):
    if False:
        for i in range(10):
            print('nop')
    'VNMC loss: to evaluate directly use `vnmc_eig` setting `num_steps=0`.'

    def loss_fn(design, num_particles, evaluation=False, **kwargs):
        if False:
            i = 10
            return i + 15
        (N, M) = num_particles
        expanded_design = lexpand(design, N)
        trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: lexpand(trace.nodes[l]['value'], M) for l in observation_labels}
        reexpanded_design = lexpand(expanded_design, M)
        conditional_guide = pyro.condition(guide, data=y_dict)
        guide_trace = poutine.trace(conditional_guide).get_trace(y_dict, reexpanded_design, observation_labels, target_labels)
        theta_y_dict = {l: guide_trace.nodes[l]['value'] for l in target_labels}
        theta_y_dict.update(y_dict)
        guide_trace.compute_log_prob()
        modelp = pyro.condition(model, data=theta_y_dict)
        model_trace = poutine.trace(modelp).get_trace(reexpanded_design)
        model_trace.compute_log_prob()
        terms = -sum((guide_trace.nodes[l]['log_prob'] for l in target_labels))
        terms += sum((model_trace.nodes[l]['log_prob'] for l in target_labels))
        terms += sum((model_trace.nodes[l]['log_prob'] for l in observation_labels))
        terms = -terms.logsumexp(0) + math.log(M)
        if evaluation:
            trace.compute_log_prob()
            terms += sum((trace.nodes[l]['log_prob'] for l in observation_labels))
        return _safe_mean_terms(terms)
    return loss_fn

def _safe_mean_terms(terms):
    if False:
        i = 10
        return i + 15
    mask = torch.isnan(terms) | (terms == float('-inf')) | (terms == float('inf'))
    if terms.dtype is torch.float32:
        nonnan = (~mask).sum(0).float()
    elif terms.dtype is torch.float64:
        nonnan = (~mask).sum(0).double()
    terms[mask] = 0.0
    loss = terms.sum(0) / nonnan
    agg_loss = loss.sum()
    return (agg_loss, loss)

def xexpx(a):
    if False:
        print('Hello World!')
    'Computes `a*exp(a)`.\n\n    This function makes the outputs more stable when the inputs of this function converge to :math:`-\\infty`.\n\n    :param torch.Tensor a:\n    :return: Equivalent of `a*torch.exp(a)`.\n    '
    mask = a == float('-inf')
    y = a * torch.exp(a)
    y[mask] = 0.0
    return y

class _EwmaLogFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, ewma):
        if False:
            while True:
                i = 10
        ctx.save_for_backward(ewma)
        return input.log()

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            print('Hello World!')
        (ewma,) = ctx.saved_tensors
        return (grad_output / ewma, None)
_ewma_log_fn = _EwmaLogFn.apply

class EwmaLog:
    """Logarithm function with exponentially weighted moving average
    for gradients.

    For input `inputs` this function return :code:`inputs.log()`. However, it
    computes the gradient as

        :math:`\\frac{\\sum_{t=0}^{T-1} \\alpha^t}{\\sum_{t=0}^{T-1} \\alpha^t x_{T-t}}`

    where :math:`x_t` are historical input values passed to this function,
    :math:`x_T` being the most recently seen value.

    This gradient may help with numerical stability when the sequence of
    inputs to the function form a convergent sequence.
    """

    def __init__(self, alpha):
        if False:
            while True:
                i = 10
        self.alpha = alpha
        self.ewma = 0.0
        self.n = 0
        self.s = 0.0

    def __call__(self, inputs, s, dim=0, keepdim=False):
        if False:
            return 10
        'Updates the moving average, and returns :code:`inputs.log()`.'
        self.n += 1
        if torch_isnan(self.ewma) or torch_isinf(self.ewma):
            ewma = inputs
        else:
            ewma = inputs * (1.0 - self.alpha) / (1 - self.alpha ** self.n) + torch.exp(self.s - s) * self.ewma * (self.alpha - self.alpha ** self.n) / (1 - self.alpha ** self.n)
        self.ewma = ewma.detach()
        self.s = s.detach()
        return _ewma_log_fn(inputs, ewma)