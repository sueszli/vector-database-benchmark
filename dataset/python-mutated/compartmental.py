import functools
import logging
import operator
import re
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from functools import reduce
from timeit import default_timer
import torch
from torch.distributions import biject_to, constraints
from torch.distributions.utils import lazy_property
import pyro.distributions as dist
import pyro.distributions.hmm
import pyro.poutine as poutine
from pyro.distributions.transforms import HaarTransform
from pyro.infer import MCMC, NUTS, SVI, JitTrace_ELBO, SMCFilter, Trace_ELBO, infer_discrete
from pyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoMultivariateNormal, AutoNormal, init_to_generated, init_to_value
from pyro.infer.mcmc import ArrowheadMassMatrix
from pyro.infer.reparam import HaarReparam, SplitReparam
from pyro.infer.smcfilter import SMCFailed
from pyro.infer.util import is_validation_enabled
from pyro.optim import ClippedAdam
from pyro.poutine.util import site_is_factor, site_is_subsample
from pyro.util import warn_if_nan
from .distributions import set_approx_log_prob_tol, set_approx_sample_thresh, set_relaxed_distributions
from .util import align_samples, cat2, clamp, quantize, quantize_enumerate
logger = logging.getLogger(__name__)

def _require_double_precision():
    if False:
        while True:
            i = 10
    if torch.get_default_dtype() != torch.float64:
        warnings.warn('CompartmentalModel is unstable for dtypes less than torch.float64; try torch.set_default_dtype(torch.float64)', RuntimeWarning)

@contextmanager
def _disallow_latent_variables(section_name):
    if False:
        while True:
            i = 10
    if not is_validation_enabled():
        yield
        return
    with poutine.trace() as tr:
        yield
    for (name, site) in tr.trace.nodes.items():
        if site['type'] == 'sample' and (not site['is_observed']):
            raise NotImplementedError('{} contained latent variable {}'.format(section_name, name))

class CompartmentalModel(ABC):
    """
    Abstract base class for discrete-time discrete-value stochastic
    compartmental models.

    Derived classes must implement methods :meth:`initialize` and
    :meth:`transition`. Derived classes may optionally implement
    :meth:`global_model`, :meth:`compute_flows`, and :meth:`heuristic`.

    Example usage::

        # First implement a concrete derived class.
        class MyModel(CompartmentalModel):
            def __init__(self, ...): ...
            def global_model(self): ...
            def initialize(self, params): ...
            def transition(self, params, state, t): ...

        # Run inference to fit the model to data.
        model = MyModel(...)
        model.fit_svi(num_samples=100)  # or .fit_mcmc(...)
        R0 = model.samples["R0"]  # An example parameter.
        print("R0 = {:0.3g} ± {:0.3g}".format(R0.mean(), R0.std()))

        # Predict latent variables.
        samples = model.predict()

        # Forecast forward.
        samples = model.predict(forecast=30)

        # You can assess future interventions (applied after ``duration``) by
        # storing them as attributes that are read by your derived methods.
        model.my_intervention = False
        samples1 = model.predict(forecast=30)
        model.my_intervention = True
        samples2 = model.predict(forecast=30)
        effect = samples2["my_result"].mean() - samples1["my_result"].mean()
        print("average effect = {:0.3g}".format(effect))

    An example workflow is to use cheaper approximate inference while finding
    good model structure and priors, then move to more accurate but more
    expensive inference once the model is plausible.

    1.  Start with ``.fit_svi(guide_rank=0, num_steps=2000)`` for cheap
        inference while you search for a good model.
    2.  Additionally infer long-range correlations by moving to a low-rank
        multivariate normal guide via ``.fit_svi(guide_rank=None,
        num_steps=5000)``.
    3.  Optionally additionally infer non-Gaussian posterior by moving to the
        more expensive (but still approximate via moment matching)
        ``.fit_mcmc(num_quant_bins=1, num_samples=10000, num_chains=2)``.
    4.  Optionally improve fit around small counts by moving the the more
        expensive enumeration-based algorithm ``.fit_mcmc(num_quant_bins=4,
        num_samples=10000, num_chains=2)`` (GPU recommended).

    :ivar dict samples: Dictionary of posterior samples.
    :param list compartments: A list of strings of compartment names.
    :param int duration: The number of discrete time steps in this model.
    :param population: Either the total population of a single-region model or
        a tensor of each region's population in a regional model.
    :type population: int or torch.Tensor
    :param tuple approximate: Names of compartments for which pointwise
        approximations should be provided in :meth:`transition`, e.g. if you
        specify ``approximate=("I")`` then the ``state["I_approx"]`` will be a
        continuous-valued non-enumerated point estimate of ``state["I"]``.
        Approximations are useful to reduce computational cost. Approximations
        are continuous-valued with support ``(-0.5, population + 0.5)``.
    """

    def __init__(self, compartments, duration, population, *, approximate=()):
        if False:
            print('Hello World!')
        super().__init__()
        assert isinstance(duration, int)
        assert duration >= 1
        self.duration = duration
        if isinstance(population, torch.Tensor):
            assert population.dim() == 1
            assert (population >= 1).all()
            self.is_regional = True
            self.max_plate_nesting = 2
        else:
            assert isinstance(population, int)
            assert population >= 2
            self.is_regional = False
            self.max_plate_nesting = 1
        self.population = population
        compartments = tuple(compartments)
        assert all((isinstance(name, str) for name in compartments))
        assert len(compartments) == len(set(compartments))
        self.compartments = compartments
        assert isinstance(approximate, tuple)
        assert all((name in compartments for name in approximate))
        self.approximate = approximate
        self.samples = {}
        self._clear_plates()

    @property
    def time_plate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A ``pyro.plate`` for the time dimension.\n        '
        if self._time_plate is None:
            self._time_plate = pyro.plate('time', self.duration, dim=-2 if self.is_regional else -1)
        return self._time_plate

    @property
    def region_plate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Either a ``pyro.plate`` or a trivial ``ExitStack`` depending on whether\n        this model ``.is_regional``.\n        '
        if self._region_plate is None:
            if self.is_regional:
                self._region_plate = pyro.plate('region', len(self.population), dim=-1)
            else:
                self._region_plate = ExitStack()
        return self._region_plate

    def _clear_plates(self):
        if False:
            i = 10
            return i + 15
        self._time_plate = None
        self._region_plate = None

    @lazy_property
    def full_mass(self):
        if False:
            i = 10
            return i + 15
        '\n        A list of a single tuple of the names of global random variables.\n        '
        with torch.no_grad(), poutine.block(), poutine.trace() as tr:
            self.global_model()
        return [tuple((name for (name, site) in tr.trace.iter_stochastic_nodes() if not site_is_subsample(site)))]

    @lazy_property
    def series(self):
        if False:
            return 10
        '\n        A frozenset of names of sample sites that are sampled each time step.\n        '
        with torch.no_grad(), poutine.block():
            params = self.global_model()
            prev = self.initialize(params)
            for name in self.approximate:
                prev[name + '_approx'] = prev[name]
            curr = prev.copy()
            with poutine.trace() as tr:
                self.transition(params, curr, 0)
        return frozenset((re.match('(.*)_0', name).group(1) for (name, site) in tr.trace.nodes.items() if site['type'] == 'sample' if not site_is_subsample(site)))

    def global_model(self):
        if False:
            return 10
        '\n        Samples and returns any global parameters.\n\n        :returns: An arbitrary object of parameters (e.g. ``None`` or a tuple).\n        '
        return None

    @abstractmethod
    def initialize(self, params):
        if False:
            return 10
        '\n        Returns initial counts in each compartment.\n\n        :param params: The global params returned by :meth:`global_model`.\n        :returns: A dict mapping compartment name to initial value.\n        :rtype: dict\n        '
        raise NotImplementedError

    @abstractmethod
    def transition(self, params, state, t):
        if False:
            i = 10
            return i + 15
        '\n        Forward generative process for dynamics.\n\n        This inputs a current ``state`` and stochastically updates that\n        state in-place.\n\n        Note that this method is called under multiple different\n        interpretations, including batched and vectorized interpretations.\n        During :meth:`generate` this is called to generate a single sample.\n        During :meth:`heuristic` this is called to generate a batch of samples\n        for SMC.  During :meth:`fit_mcmc` this is called both in vectorized form\n        (vectorizing over time) and in sequential form (for a single time\n        step); both forms enumerate over discrete latent variables.  During\n        :meth:`predict` this is called to forecast a batch of samples,\n        conditioned on posterior samples for the time interval\n        ``[0:duration]``.\n\n        :param params: The global params returned by :meth:`global_model`.\n        :param dict state: A dictionary mapping compartment name to current\n            tensor value. This should be updated in-place.\n        :param t: A time-like index. During inference ``t`` may be either a\n            slice (for vectorized inference) or an integer time index. During\n            prediction ``t`` will be integer time index.\n        :type t: int or slice\n        '
        raise NotImplementedError

    def finalize(self, params, prev, curr):
        if False:
            i = 10
            return i + 15
        '\n        Optional method for likelihoods that depend on entire time series.\n\n        This should be used only for non-factorizable likelihoods that couple\n        states across time. Factorizable likelihoods should instead be added to\n        the :meth:`transition` method, thereby enabling their use in\n        :meth:`heuristic` initialization. Since this method is called only\n        after the last time step, it is not used in :meth:`heuristic`\n        initialization.\n\n        .. warning:: This currently does not support latent variables.\n\n        :param params: The global params returned by :meth:`global_model`.\n        :param dict prev:\n        :param dict curr: Dictionaries mapping compartment name to tensor of\n            entire time series. These two parameters are offset by 1 step,\n            thereby making it easy to compute time series of fluxes. For\n            quantized inference, this uses the approximate point estimates, so\n            users must request any needed time series in :meth:`__init__`, e.g.\n            by calling ``super().__init__(..., approximate=("I", "E"))`` if\n            likelihood depends on the ``I`` and ``E`` time series.\n        '
        pass

    def compute_flows(self, prev, curr, t):
        if False:
            while True:
                i = 10
        '\n        Computes flows between compartments, given compartment populations\n        before and after time step t.\n\n        The default implementation assumes sequential flows terminating in an\n        implicit compartment named "R". For example if::\n\n            compartment_names = ("S", "E", "I")\n\n        the default implementation computes at time step ``t = 9``::\n\n            flows["S2E_9"] = prev["S"] - curr["S"]\n            flows["E2I_9"] = prev["E"] - curr["E"] + flows["S2E_9"]\n            flows["I2R_9"] = prev["I"] - curr["I"] + flows["E2I_9"]\n\n        For more complex flows (non-sequential, branching, looping,\n        duplicating, etc.), users may override this method.\n\n        :param dict state: A dictionary mapping compartment name to current\n            tensor value. This should be updated in-place.\n        :param t: A time-like index. During inference ``t`` may be either a\n            slice (for vectorized inference) or an integer time index. During\n            prediction ``t`` will be integer time index.\n        :type t: int or slice\n        :returns: A dict mapping flow name to tensor value.\n        :rtype: dict\n        '
        flows = {}
        flow = 0
        for (source, destin) in zip(self.compartments, self.compartments[1:] + ('R',)):
            flow = prev[source] - curr[source] + flow
            flows['{}2{}_{}'.format(source, destin, t)] = flow
        return flows

    @torch.no_grad()
    @set_approx_sample_thresh(1000)
    def generate(self, fixed={}):
        if False:
            print('Hello World!')
        '\n        Generate data from the prior.\n\n        :pram dict fixed: A dictionary of parameters on which to condition.\n            These must be top-level parentless nodes, i.e. have no\n            upstream stochastic dependencies.\n        :returns: A dictionary mapping sample site name to sampled value.\n        :rtype: dict\n        '
        fixed = {k: torch.as_tensor(v) for (k, v) in fixed.items()}
        model = self._generative_model
        model = poutine.condition(model, fixed)
        trace = poutine.trace(model).get_trace()
        samples = OrderedDict(((name, site['value']) for (name, site) in trace.nodes.items() if site['type'] == 'sample'))
        self._concat_series(samples, trace)
        return samples

    def fit_svi(self, *, num_samples=100, num_steps=2000, num_particles=32, learning_rate=0.1, learning_rate_decay=0.01, betas=(0.8, 0.99), haar=True, init_scale=0.01, guide_rank=0, jit=False, log_every=200, **options):
        if False:
            print('Hello World!')
        '\n        Runs stochastic variational inference to generate posterior samples.\n\n        This runs :class:`~pyro.infer.svi.SVI`, setting the ``.samples``\n        attribute on completion.\n\n        This approximate inference method is useful for quickly iterating on\n        probabilistic models.\n\n        :param int num_samples: Number of posterior samples to draw from the\n            trained guide. Defaults to 100.\n        :param int num_steps: Number of :class:`~pyro.infer.svi.SVI` steps.\n        :param int num_particles: Number of :class:`~pyro.infer.svi.SVI`\n            particles per step.\n        :param int learning_rate: Learning rate for the\n            :class:`~pyro.optim.clipped_adam.ClippedAdam` optimizer.\n        :param int learning_rate_decay: Learning rate for the\n            :class:`~pyro.optim.clipped_adam.ClippedAdam` optimizer. Note this\n            is decay over the entire schedule, not per-step decay.\n        :param tuple betas: Momentum parameters for the\n            :class:`~pyro.optim.clipped_adam.ClippedAdam` optimizer.\n        :param bool haar: Whether to use a Haar wavelet reparameterizer.\n        :param int guide_rank: Rank of the auto normal guide. If zero (default)\n            use an :class:`~pyro.infer.autoguide.AutoNormal` guide. If a\n            positive integer or None, use an\n            :class:`~pyro.infer.autoguide.AutoLowRankMultivariateNormal` guide.\n            If the string "full", use an\n            :class:`~pyro.infer.autoguide.AutoMultivariateNormal` guide. These\n            latter two require more ``num_steps`` to fit.\n        :param float init_scale: Initial scale of the\n            :class:`~pyro.infer.autoguide.AutoLowRankMultivariateNormal` guide.\n        :param bool jit: Whether to use a jit compiled ELBO.\n        :param int log_every: How often to log svi losses.\n        :param int heuristic_num_particles: Passed to :meth:`heuristic` as\n            ``num_particles``. Defaults to 1024.\n        :returns: Time series of SVI losses (useful to diagnose convergence).\n        :rtype: list\n        '
        self.relaxed = True
        self.num_quant_bins = 1
        if haar:
            time_dim = -2 if self.is_regional else -1
            dims = {'auxiliary': time_dim}
            supports = {'auxiliary': constraints.interval(-0.5, self.population + 0.5)}
            for (name, (fn, is_regional)) in self._non_compartmental.items():
                dims[name] = time_dim - fn.event_dim
                supports[name] = fn.support
            haar = _HaarSplitReparam(0, self.duration, dims, supports)
        heuristic_options = {k.replace('heuristic_', ''): options.pop(k) for k in list(options) if k.startswith('heuristic_')}
        assert not options, 'unrecognized options: {}'.format(', '.join(options))
        init_strategy = self._heuristic(haar, **heuristic_options)
        logger.info('Running inference...')
        model = self._relaxed_model
        if haar:
            model = haar.reparam(model)
        if guide_rank == 0:
            guide = AutoNormal(model, init_loc_fn=init_strategy, init_scale=init_scale)
        elif guide_rank == 'full':
            guide = AutoMultivariateNormal(model, init_loc_fn=init_strategy, init_scale=init_scale)
        elif guide_rank is None or isinstance(guide_rank, int):
            guide = AutoLowRankMultivariateNormal(model, init_loc_fn=init_strategy, init_scale=init_scale, rank=guide_rank)
        else:
            raise ValueError('Invalid guide_rank: {}'.format(guide_rank))
        Elbo = JitTrace_ELBO if jit else Trace_ELBO
        elbo = Elbo(max_plate_nesting=self.max_plate_nesting, num_particles=num_particles, vectorize_particles=True, ignore_jit_warnings=True)
        optim = ClippedAdam({'lr': learning_rate, 'betas': betas, 'lrd': learning_rate_decay ** (1 / num_steps)})
        svi = SVI(model, guide, optim, elbo)
        start_time = default_timer()
        losses = []
        for step in range(1 + num_steps):
            loss = svi.step() / self.duration
            if step % log_every == 0:
                logger.info('step {} loss = {:0.4g}'.format(step, loss))
            losses.append(loss)
        elapsed = default_timer() - start_time
        logger.info('SVI took {:0.1f} seconds, {:0.1f} step/sec'.format(elapsed, (1 + num_steps) / elapsed))
        with torch.no_grad():
            particle_plate = pyro.plate('particles', num_samples, dim=-1 - self.max_plate_nesting)
            guide_trace = poutine.trace(particle_plate(guide)).get_trace()
            model_trace = poutine.trace(poutine.replay(particle_plate(model), guide_trace)).get_trace()
            self.samples = {name: site['value'] for (name, site) in model_trace.nodes.items() if site['type'] == 'sample' if not site['is_observed'] if not site_is_subsample(site)}
            if haar:
                haar.aux_to_user(self.samples)
        assert all((v.size(0) == num_samples for v in self.samples.values())), {k: tuple(v.shape) for (k, v) in self.samples.items()}
        return losses

    @set_approx_log_prob_tol(0.1)
    def fit_mcmc(self, **options):
        if False:
            return 10
        '\n        Runs NUTS inference to generate posterior samples.\n\n        This uses the :class:`~pyro.infer.mcmc.nuts.NUTS` kernel to run\n        :class:`~pyro.infer.mcmc.api.MCMC`, setting the ``.samples``\n        attribute on completion.\n\n        This uses an asymptotically exact enumeration-based model when\n        ``num_quant_bins > 1``, and a cheaper moment-matched approximate model\n        when ``num_quant_bins == 1``.\n\n        :param \\*\\*options: Options passed to\n            :class:`~pyro.infer.mcmc.api.MCMC`. The remaining options are\n            pulled out and have special meaning.\n        :param int num_samples: Number of posterior samples to draw via mcmc.\n            Defaults to 100.\n        :param int max_tree_depth: (Default 5). Max tree depth of the\n            :class:`~pyro.infer.mcmc.nuts.NUTS` kernel.\n        :param full_mass: Specification of mass matrix of the\n            :class:`~pyro.infer.mcmc.nuts.NUTS` kernel. Defaults to full mass\n            over global random variables.\n        :param bool arrowhead_mass: Whether to treat ``full_mass`` as the head\n            of an arrowhead matrix versus simply as a block. Defaults to False.\n        :param int num_quant_bins: If greater than 1, use asymptotically exact\n            inference via local enumeration over this many quantization bins.\n            If equal to 1, use continuous-valued relaxed approximate inference.\n            Note that computational cost is exponential in `num_quant_bins`.\n            Defaults to 1 for relaxed inference.\n        :param bool haar: Whether to use a Haar wavelet reparameterizer.\n            Defaults to True.\n        :param int haar_full_mass: Number of low frequency Haar components to\n            include in the full mass matrix. If ``haar=False`` then this is\n            ignored. Defaults to 10.\n        :param int heuristic_num_particles: Passed to :meth:`heuristic` as\n            ``num_particles``. Defaults to 1024.\n        :returns: An MCMC object for diagnostics, e.g. ``MCMC.summary()``.\n        :rtype: ~pyro.infer.mcmc.api.MCMC\n        '
        _require_double_precision()
        num_samples = options.setdefault('num_samples', 100)
        num_chains = options.setdefault('num_chains', 1)
        self.num_quant_bins = options.pop('num_quant_bins', 1)
        assert isinstance(self.num_quant_bins, int)
        assert self.num_quant_bins >= 1
        self.relaxed = self.num_quant_bins == 1
        haar = options.pop('haar', False)
        haar_full_mass = options.pop('haar_full_mass', 10)
        full_mass = options.pop('full_mass', self.full_mass)
        assert isinstance(haar, bool)
        assert isinstance(haar_full_mass, int) and haar_full_mass >= 0
        assert isinstance(full_mass, (bool, list))
        haar_full_mass = min(haar_full_mass, self.duration)
        if not haar:
            haar_full_mass = 0
        if full_mass is True:
            haar_full_mass = 0
        elif haar_full_mass >= self.duration:
            full_mass = True
            haar_full_mass = 0
        if haar:
            time_dim = -2 if self.is_regional else -1
            dims = {'auxiliary': time_dim}
            supports = {'auxiliary': constraints.interval(-0.5, self.population + 0.5)}
            for (name, (fn, is_regional)) in self._non_compartmental.items():
                dims[name] = time_dim - fn.event_dim
                supports[name] = fn.support
            haar = _HaarSplitReparam(haar_full_mass, self.duration, dims, supports)
        if haar_full_mass:
            assert full_mass and isinstance(full_mass, list)
            full_mass = full_mass[:]
            full_mass[0] += tuple((name + '_haar_split_0' for name in sorted(dims)))
        heuristic_options = {k.replace('heuristic_', ''): options.pop(k) for k in list(options) if k.startswith('heuristic_')}
        init_strategy = init_to_generated(generate=functools.partial(self._heuristic, haar, **heuristic_options))
        logger.info('Running inference...')
        model = self._relaxed_model if self.relaxed else self._quantized_model
        if haar:
            model = haar.reparam(model)
        kernel = NUTS(model, full_mass=full_mass, init_strategy=init_strategy, max_plate_nesting=self.max_plate_nesting, jit_compile=options.pop('jit_compile', False), jit_options=options.pop('jit_options', None), ignore_jit_warnings=options.pop('ignore_jit_warnings', True), target_accept_prob=options.pop('target_accept_prob', 0.8), max_tree_depth=options.pop('max_tree_depth', 5))
        if options.pop('arrowhead_mass', False):
            kernel.mass_matrix_adapter = ArrowheadMassMatrix()
        options.setdefault('disable_validation', None)
        mcmc = MCMC(kernel, **options)
        mcmc.run()
        self.samples = mcmc.get_samples()
        if haar:
            haar.aux_to_user(self.samples)
        model = self._relaxed_model if self.relaxed else self._quantized_model
        self.samples = align_samples(self.samples, model, particle_dim=-1 - self.max_plate_nesting)
        assert all((v.size(0) == num_samples * num_chains for v in self.samples.values())), {k: tuple(v.shape) for (k, v) in self.samples.items()}
        return mcmc

    @torch.no_grad()
    @set_approx_log_prob_tol(0.1)
    @set_approx_sample_thresh(10000)
    def predict(self, forecast=0):
        if False:
            while True:
                i = 10
        '\n        Predict latent variables and optionally forecast forward.\n\n        This may be run only after :meth:`fit_mcmc` and draws the same\n        ``num_samples`` as passed to :meth:`fit_mcmc`.\n\n        :param int forecast: The number of time steps to forecast forward.\n        :returns: A dictionary mapping sample site name (or compartment name)\n            to a tensor whose first dimension corresponds to sample batching.\n        :rtype: dict\n        '
        if self.num_quant_bins > 1:
            _require_double_precision()
        if not self.samples:
            raise RuntimeError('Missing samples, try running .fit_mcmc() first')
        samples = self.samples
        num_samples = len(next(iter(samples.values())))
        particle_plate = pyro.plate('particles', num_samples, dim=-1 - self.max_plate_nesting)
        logger.info('Predicting latent variables for {} time steps...'.format(self.duration))
        model = self._sequential_model
        model = poutine.condition(model, samples)
        model = particle_plate(model)
        if not self.relaxed:
            model = infer_discrete(model, first_available_dim=-2 - self.max_plate_nesting)
        trace = poutine.trace(model).get_trace()
        samples = OrderedDict(((name, site['value'].expand(site['fn'].shape())) for (name, site) in trace.nodes.items() if site['type'] == 'sample' if not site_is_subsample(site) if not site_is_factor(site)))
        assert all((v.size(0) == num_samples for v in samples.values())), {k: tuple(v.shape) for (k, v) in samples.items()}
        if forecast:
            logger.info('Forecasting {} steps ahead...'.format(forecast))
            model = self._generative_model
            model = poutine.condition(model, samples)
            model = particle_plate(model)
            trace = poutine.trace(model).get_trace(forecast)
            samples = OrderedDict(((name, site['value']) for (name, site) in trace.nodes.items() if site['type'] == 'sample' if not site_is_subsample(site) if not site_is_factor(site)))
        self._concat_series(samples, trace, forecast)
        assert all((v.size(0) == num_samples for v in samples.values())), {k: tuple(v.shape) for (k, v) in samples.items()}
        return samples

    @torch.no_grad()
    @set_approx_log_prob_tol(0.1)
    @set_approx_sample_thresh(100)
    def heuristic(self, num_particles=1024, ess_threshold=0.5, retries=10):
        if False:
            i = 10
            return i + 15
        '\n        Finds an initial feasible guess of all latent variables, consistent\n        with observed data. This is needed because not all hypotheses are\n        feasible and HMC needs to start at a feasible solution to progress.\n\n        The default implementation attempts to find a feasible state using\n        :class:`~pyro.infer.smcfilter.SMCFilter` with proprosals from the\n        prior.  However this method may be overridden in cases where SMC\n        performs poorly e.g. in high-dimensional models.\n\n        :param int num_particles: Number of particles used for SMC.\n        :param float ess_threshold: Effective sample size threshold for SMC.\n        :returns: A dictionary mapping sample site name to tensor value.\n        :rtype: dict\n        '
        model = _SMCModel(self)
        guide = _SMCGuide(self)
        for attempt in range(1, 1 + retries):
            smc = SMCFilter(model, guide, num_particles=num_particles, ess_threshold=ess_threshold, max_plate_nesting=self.max_plate_nesting)
            try:
                smc.init()
                for t in range(1, self.duration):
                    smc.step()
                break
            except SMCFailed as e:
                if attempt == retries:
                    raise
                logger.info('{}. Retrying...'.format(e))
                continue
        i = int(smc.state._log_weights.max(0).indices)
        init = {key: value[i, 0] for (key, value) in smc.state.items()}
        init = self.generate(init)
        aux = torch.stack([init[name] for name in self.compartments], dim=0)
        init['auxiliary'] = clamp(aux, min=0.5, max=self.population - 0.5)
        return init

    def _heuristic(self, haar, **options):
        if False:
            i = 10
            return i + 15
        with poutine.block():
            init_values = self.heuristic(**options)
        assert isinstance(init_values, dict)
        assert 'auxiliary' in init_values, '.heuristic() did not define auxiliary value'
        logger.info('Heuristic init: {}'.format(', '.join(('{}={:0.3g}'.format(k, v.item()) for (k, v) in sorted(init_values.items()) if v.numel() == 1))))
        return init_to_value(values=init_values, fallback=None)

    def _concat_series(self, samples, trace, forecast=0):
        if False:
            while True:
                i = 10
        '\n        Concatenate sequential time series into tensors, in-place.\n\n        :param dict samples: A dictionary of samples.\n        '
        time_dim = -2 if self.is_regional else -1
        for name in set(self.compartments).union(self.series):
            pattern = name + '_[0-9]+'
            series = []
            for key in list(samples):
                if re.match(pattern, key):
                    series.append(samples.pop(key))
            if not series:
                continue
            assert len(series) == self.duration + forecast
            series = torch.broadcast_tensors(*map(torch.as_tensor, series))
            dim = time_dim - trace.nodes[name + '_0']['fn'].event_dim
            if series[0].dim() >= -dim:
                samples[name] = torch.cat(series, dim=dim)
            else:
                samples[name] = torch.stack(series)

    @lazy_property
    @torch.no_grad()
    def _non_compartmental(self):
        if False:
            while True:
                i = 10
        '\n        A dict mapping name -> (distribution, is_regional) for all\n        non-compartmental sites in :meth:`transition`. For simple models this\n        is often empty; for time-heterogeneous models this may contain\n        time-local latent variables.\n        '
        with torch.no_grad(), poutine.block():
            params = self.global_model()
            prev = self.initialize(params)
            for name in self.approximate:
                prev[name + '_approx'] = prev[name]
            curr = prev.copy()
            with poutine.trace() as tr:
                self.transition(params, curr, 0)
            flows = self.compute_flows(prev, curr, 0)
        result = OrderedDict()
        for (name, site) in tr.trace.iter_stochastic_nodes():
            if name in flows or site_is_subsample(site):
                continue
            assert name.endswith('_0'), name
            name = name[:-2]
            assert name in self.series, name
            is_regional = any((f.name == 'region' for f in site['cond_indep_stack']))
            result[name] = (site['fn'], is_regional)
        return result

    def _sample_auxiliary(self):
        if False:
            return 10
        '\n        Sample both compartmental and non-compartmental auxiliary variables.\n        '
        C = len(self.compartments)
        T = self.duration
        R_shape = getattr(self.population, 'shape', ())
        shape = (C, T) + R_shape
        auxiliary = pyro.sample('auxiliary', dist.Uniform(-0.5, self.population + 0.5).mask(False).expand(shape).to_event())
        extra_dims = auxiliary.dim() - len(shape)
        non_compartmental = OrderedDict()
        for (name, (fn, is_regional)) in self._non_compartmental.items():
            fn = dist.ImproperUniform(fn.support, fn.batch_shape, fn.event_shape)
            shape = (T,)
            if self.is_regional:
                shape += R_shape if is_regional else (1,)
            non_compartmental[name] = pyro.sample(name, fn.expand(shape).to_event())
        if extra_dims:
            shape = auxiliary.shape[:1] + auxiliary.shape[extra_dims:]
            auxiliary = auxiliary.reshape(shape)
            for (name, value) in non_compartmental.items():
                shape = value.shape[:1] + value.shape[extra_dims:]
                non_compartmental[name] = value.reshape(shape)
        return (auxiliary, non_compartmental)

    def _transition_bwd(self, params, prev, curr, t):
        if False:
            i = 10
            return i + 15
        '\n        Helper to collect probabilty factors from .transition() conditioned on\n        previous and current enumerated states.\n        '
        cond_data = {'{}_{}'.format(k, t): v for (k, v) in curr.items()}
        cond_data.update(self.compute_flows(prev, curr, t))
        with poutine.condition(data=cond_data):
            state = prev.copy()
            self.transition(params, state, t)
        if is_validation_enabled():
            for key in self.compartments:
                if not torch.allclose(state[key], curr[key]):
                    raise ValueError("Incorrect state['{}'] update in .transition(), check that .transition() matches .compute_flows().".format(key))

    def _generative_model(self, forecast=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Forward generative model used for simulation and forecasting.\n        '
        params = self.global_model()
        state = self.initialize(params)
        state = {k: v if isinstance(v, torch.Tensor) else torch.tensor(float(v)) for (k, v) in state.items()}
        for t in range(self.duration + forecast):
            for name in self.approximate:
                state[name + '_approx'] = state[name]
            self.transition(params, state, t)
            with self.region_plate:
                for name in self.compartments:
                    pyro.deterministic('{}_{}'.format(name, t), state[name], event_dim=0)
        self._clear_plates()

    def _sequential_model(self):
        if False:
            i = 10
            return i + 15
        '\n        Sequential model used to sample latents in the interval [0:duration].\n        This is compatible with both quantized and relaxed inference.\n        This method is called only inside particle_plate.\n        This method is used only for prediction.\n        '
        C = len(self.compartments)
        T = self.duration
        R_shape = getattr(self.population, 'shape', ())
        num_samples = len(next(iter(self.samples.values())))
        params = self.global_model()
        (auxiliary, non_compartmental) = self._sample_auxiliary()
        assert auxiliary.shape == (num_samples, C, T) + R_shape, (auxiliary.shape, (num_samples, C, T) + R_shape)
        aux = [aux.unbind(2) for aux in auxiliary.unsqueeze(1).unbind(2)]
        curr = self.initialize(params)
        for t in poutine.markov(range(T)):
            with self.region_plate:
                (prev, curr) = (curr, {})
                for (name, value) in non_compartmental.items():
                    curr[name] = value[:, t:t + 1]
                for (c, name) in enumerate(self.compartments):
                    curr[name] = quantize('{}_{}'.format(name, t), aux[c][t], min=0, max=self.population, num_quant_bins=self.num_quant_bins)
                    if name in self.approximate:
                        curr[name + '_approx'] = aux[c][t]
                        prev.setdefault(name + '_approx', prev[name])
            self._transition_bwd(params, prev, curr, t)
        self._clear_plates()

    def _quantized_model(self):
        if False:
            while True:
                i = 10
        '\n        Quantized vectorized model used for parallel-scan enumerated inference.\n        This method is called only outside particle_plate.\n        '
        C = len(self.compartments)
        T = self.duration
        Q = self.num_quant_bins
        R_shape = getattr(self.population, 'shape', ())
        params = self.global_model()
        (auxiliary, non_compartmental) = self._sample_auxiliary()
        (curr, logp) = quantize_enumerate(auxiliary, min=0, max=self.population, num_quant_bins=self.num_quant_bins)
        curr = OrderedDict(zip(self.compartments, curr.unbind(0)))
        logp = OrderedDict(zip(self.compartments, logp.unbind(0)))
        curr.update(non_compartmental)
        init = self.initialize(params)
        prev = {}
        for (name, value) in init.items():
            if name in self.compartments:
                if isinstance(value, torch.Tensor):
                    value = value[..., None]
                prev[name] = cat2(value, curr[name][:-1], dim=-3 if self.is_regional else -2)
            else:
                prev[name] = cat2(init[name], curr[name][:-1], dim=-curr[name].dim())

        def enum_reshape(tensor, position):
            if False:
                while True:
                    i = 10
            assert tensor.size(-1) == Q
            assert tensor.dim() <= self.max_plate_nesting + 2
            tensor = tensor.permute(tensor.dim() - 1, *range(tensor.dim() - 1))
            shape = [Q] + [1] * (position + self.max_plate_nesting - (tensor.dim() - 2))
            shape.extend(tensor.shape[1:])
            return tensor.reshape(shape)
        for (e, name) in enumerate(self.compartments):
            curr[name] = enum_reshape(curr[name], e)
            logp[name] = enum_reshape(logp[name], e)
            prev[name] = enum_reshape(prev[name], e + C)
        for name in self.approximate:
            aux = auxiliary[self.compartments.index(name)]
            curr[name + '_approx'] = aux
            prev[name + '_approx'] = cat2(init[name], aux[:-1], dim=-2 if self.is_regional else -1)
        with poutine.block(), poutine.trace() as tr:
            with self.time_plate:
                t = slice(0, T, 1)
                self._transition_bwd(params, prev, curr, t)
        tr.trace.compute_log_prob()
        for (name, site) in tr.trace.nodes.items():
            if site['type'] == 'sample':
                log_prob = site['log_prob']
                if log_prob.dim() <= self.max_plate_nesting:
                    pyro.factor('transition_' + name, site['log_prob_sum'])
                    continue
                if self.is_regional and log_prob.shape[-1:] != R_shape:
                    log_prob = log_prob.expand(log_prob.shape[:-1] + R_shape) / R_shape[0]
                logp[name] = site['log_prob']
        logp = reduce(operator.add, logp.values())
        logp = logp.reshape(Q ** C, Q ** C, T, -1)
        logp = logp.permute(3, 2, 0, 1).squeeze(0)
        logp = pyro.distributions.hmm._sequential_logmatmulexp(logp)
        logp = logp.reshape(-1, Q ** C * Q ** C).logsumexp(-1).sum()
        warn_if_nan(logp)
        pyro.factor('transition', logp)
        prev = {name: prev[name + '_approx'] for name in self.approximate}
        curr = {name: curr[name + '_approx'] for name in self.approximate}
        with _disallow_latent_variables('.finalize()'):
            self.finalize(params, prev, curr)
        self._clear_plates()

    @set_relaxed_distributions()
    def _relaxed_model(self):
        if False:
            i = 10
            return i + 15
        '\n        Relaxed vectorized model used for continuous inference.\n        This method may be called either inside or outside particle_plate.\n        '
        T = self.duration
        params = self.global_model()
        (auxiliary, non_compartmental) = self._sample_auxiliary()
        particle_dims = auxiliary.dim() - (3 if self.is_regional else 2)
        assert particle_dims in (0, 1)
        curr = dict(zip(self.compartments, auxiliary.unbind(particle_dims)))
        curr.update(non_compartmental)
        prev = {}
        for (name, value) in self.initialize(params).items():
            dim = particle_dims - curr[name].dim()
            t = (slice(None),) * particle_dims + (slice(0, -1),)
            prev[name] = cat2(value, curr[name][t], dim=dim)
        for name in self.approximate:
            curr[name + '_approx'] = curr[name]
            prev[name + '_approx'] = prev[name]
        with self.time_plate:
            t = slice(0, T, 1)
            self._transition_bwd(params, prev, curr, t)
        with _disallow_latent_variables('.finalize()'):
            self.finalize(params, prev, curr)
        self._clear_plates()

class _SMCModel:
    """
    Helper to initialize a CompartmentalModel to a feasible initial state.
    """

    def __init__(self, model):
        if False:
            print('Hello World!')
        assert isinstance(model, CompartmentalModel)
        self.model = model

    def init(self, state):
        if False:
            print('Hello World!')
        with poutine.trace() as tr:
            params = self.model.global_model()
        for (name, site) in tr.trace.nodes.items():
            if site['type'] == 'sample':
                state[name] = site['value']
        self.t = 0
        state.update(self.model.initialize(params))
        self.step(state)

    def step(self, state):
        if False:
            i = 10
            return i + 15
        with poutine.block(), poutine.condition(data=state):
            params = self.model.global_model()
        with poutine.trace() as tr:
            extended_state = dict(state)
            for name in self.model.approximate:
                extended_state[name + '_approx'] = state[name]
            self.model.transition(params, extended_state, self.t)
            for name in self.model.approximate:
                del extended_state[name + '_approx']
            state.update(extended_state)
        for (name, site) in tr.trace.nodes.items():
            if site['type'] == 'sample' and (not site['is_observed']):
                state[name] = site['value']
        self.t += 1

class _SMCGuide(_SMCModel):
    """
    Like _SMCModel but does not update state and does not observe.
    """

    def init(self, state):
        if False:
            while True:
                i = 10
        super().init(state.copy())

    def step(self, state):
        if False:
            return 10
        with poutine.block(hide_types=['observe']):
            super().step(state.copy())

class _HaarSplitReparam:
    """
    Wrapper around ``HaarReparam`` and ``SplitReparam`` to additionally convert
    sample dicts between user-facing and auxiliary coordinates.
    """

    def __init__(self, split, duration, dims, supports):
        if False:
            while True:
                i = 10
        assert 0 <= split < duration
        self.split = split
        self.duration = duration
        self.dims = dims
        self.supports = supports

    def __bool__(self):
        if False:
            while True:
                i = 10
        return True

    def reparam(self, model):
        if False:
            while True:
                i = 10
        '\n        Wrap a model with ``poutine.reparam``.\n        '
        config = {}
        for (name, dim) in self.dims.items():
            config[name] = HaarReparam(dim=dim, flip=True)
        model = poutine.reparam(model, config)
        if self.split:
            splits = [self.split, self.duration - self.split]
            config = {}
            for (name, dim) in self.dims.items():
                config[name + '_haar'] = SplitReparam(splits, dim=dim)
            model = poutine.reparam(model, config)
        return model

    def aux_to_user(self, samples):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert from auxiliary samples to user-facing samples, in-place.\n        '
        if self.split:
            for (name, dim) in self.dims.items():
                samples[name + '_haar'] = torch.cat([samples.pop(name + '_haar_split_0'), samples.pop(name + '_haar_split_1')], dim=dim)
        for (name, dim) in self.dims.items():
            x = samples.pop(name + '_haar')
            x = HaarTransform(dim=dim, flip=True).inv(x)
            x = biject_to(self.supports[name])(x)
            samples[name] = x