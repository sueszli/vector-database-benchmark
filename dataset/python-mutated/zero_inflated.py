import time
import torch
from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _reshape_weights
from ._distribution import Distribution

class ZeroInflated(Distribution):
    """A wrapper for a zero-inflated distribution.

	Some discrete distributions, e.g. Poisson or negative binomial, are used
	to model data that has many more zeroes in it than one would expect from
	the true signal itself. Potentially, this is because data collection devices
	fail or other gaps exist in the data. A zero-inflated distribution is
	essentially a mixture of these zero values and the real underlying
	distribution.

	Accordingly, this class serves as a wrapper that can be dropped in for
	other probability distributions and makes them "zero-inflated". It is
	similar to a mixture model between the distribution passed in and a dirac
	delta distribution, except that the mixture happens independently for each
	distribution as well as for each example.


	Parameters
	----------
	distribution: pomegranate.distributions.Distribution
		A pomegranate distribution object. It should probably be a discrete
		distribution, but does not technically have to be.

	priors: tuple, numpy.ndarray, torch.Tensor, or None. shape=(2,), optional
		The prior probabilities over the given distribution and the dirac
		delta component. Default is None.

	max_iter: int, optional
		The number of iterations to do in the EM step of fitting the
		distribution. Default is 10.

	tol: float, optional
		The threshold at which to stop during fitting when the improvement
		goes under. Default is 0.1.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	verbose: bool, optional
		Whether to print the improvement and timings during training.
	"""

    def __init__(self, distribution, priors=None, max_iter=10, tol=0.1, inertia=0.0, frozen=False, check_data=False, verbose=False):
        if False:
            i = 10
            return i + 15
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'ZeroInflated'
        self.distribution = distribution
        self.priors = _check_parameter(_cast_as_parameter(priors), 'priors', min_value=0, max_value=1, ndim=1, value_sum=1.0)
        self.verbose = verbose
        self._initialized = distribution._initialized is True
        self.d = distribution.d if self._initialized else None
        self.max_iter = max_iter
        self.tol = tol
        if self.priors is None and self.d is not None:
            self.priors = _cast_as_parameter(torch.ones(self.d, device=self.device) / 2)
        self._reset_cache()

    def _initialize(self, X):
        if False:
            i = 10
            return i + 15
        'Initialize the probability distribution.\n\n\t\tThis method is meant to only be called internally. It initializes the\n\t\tparameters of the distribution and stores its dimensionality. For more\n\t\tcomplex methods, this function will do more.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, numpy.ndarray, torch.Tensor, shape=(1, self.d)\n\t\t\tThe data to use to initialize the model.\n\t\t'
        self.distribution._initialize(X.shape[1])
        self.distribution.fit(X)
        self.priors = _cast_as_parameter(torch.ones(X.shape[1], device=self.device) / 2)
        self._initialized = True
        super()._initialize(X.shape[1])

    def _reset_cache(self):
        if False:
            return 10
        'Reset the internally stored statistics.\n\n\t\tThis method is meant to only be called internally. It resets the\n\t\tstored statistics used to update the model parameters as well as\n\t\trecalculates the cached values meant to speed up log probability\n\t\tcalculations.\n\t\t'
        if self._initialized == False:
            return
        self.register_buffer('_w_sum', torch.zeros(self.d, 2, device=self.device))
        self.register_buffer('_log_priors', torch.log(self.priors))

    def _emission_matrix(self, X):
        if False:
            print('Hello World!')
        'Return the emission/responsibility matrix.\n\n\t\tThis method returns the log probability of each example under each\n\t\tdistribution contained in the model with the log prior probability\n\t\tof each component added.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to evaluate. \n\n\t\n\t\tReturns\n\t\t-------\n\t\te: torch.Tensor, shape=(-1, self.k)\n\t\t\tA set of log probabilities for each example under each distribution.\n\t\t'
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, shape=(-1, self.d))
        e = torch.empty(X.shape[0], self.d, 2, device=self.device)
        e[:, :, 0] = self._log_priors.unsqueeze(0)
        e[:, :, 0] += self.distribution.log_probability(X).unsqueeze(1)
        e[:, :, 1] = torch.log(1 - self.priors).unsqueeze(0)
        e[:, :, 1] += torch.where(X == 0, 0, float('-inf'))
        return e

    def fit(self, X, sample_weight=None):
        if False:
            i = 10
            return i + 15
        "Fit the model to optionally weighted examples.\n\n\t\tThis method implements the core of the learning process. For a\n\t\tzero-inflated distribution, this involves performing EM until the\n\t\tdistribution being fit converges.\n\n\t\tThis method is largely a wrapper around the `summarize` and\n\t\t`from_summaries` methods. It's primary contribution is serving as a\n\t\tloop around these functions and to monitor convergence.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to evaluate. \n\n\t\tsample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional\n\t\t\tA set of weights for the examples. This can be either of shape\n\t\t\t(-1, self.d) or a vector of shape (-1,). Default is ones.\n\n\n\t\tReturns\n\t\t-------\n\t\tself\n\t\t"
        logp = None
        for i in range(self.max_iter):
            start_time = time.time()
            last_logp = logp
            logp = self.summarize(X, sample_weight=sample_weight)
            if i > 0:
                improvement = logp - last_logp
                duration = time.time() - start_time
                if self.verbose:
                    print('[{}] Improvement: {}, Time: {:4.4}s'.format(i, improvement, duration))
                if improvement < self.tol:
                    break
            self.from_summaries()
        self._reset_cache()
        return self

    def summarize(self, X, sample_weight=None):
        if False:
            i = 10
            return i + 15
        'Extract the sufficient statistics from a batch of data.\n\n\t\tThis method calculates the sufficient statistics from optionally\n\t\tweighted data and adds them to the stored cache. The examples must be\n\t\tgiven in a 2D format. Sample weights can either be provided as one\n\t\tvalue per example or as a 2D matrix of weights for each feature in\n\t\teach example.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to summarize.\n\n\t\tsample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional\n\t\t\tA set of weights for the examples. This can be either of shape\n\t\t\t(-1, self.d) or a vector of shape (-1,). Default is ones.\n\t\t'
        X = _cast_as_tensor(X)
        if not self._initialized:
            self._initialize(X)
        _check_parameter(X, 'X', ndim=2, shape=(-1, self.d))
        sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight, dtype=torch.float32), device=self.device)
        e = self._emission_matrix(X)
        logp = torch.logsumexp(e, dim=2, keepdims=True)
        y = torch.exp(e - logp)
        self.distribution.summarize(X, y[:, :, 0] * sample_weight)
        if not self.frozen:
            self._w_sum += torch.sum(y * sample_weight.unsqueeze(-1), dim=(0, 1))
        return torch.sum(logp)

    def from_summaries(self):
        if False:
            return 10
        'Update the model parameters given the extracted statistics.\n\n\t\tThis method uses calculated statistics from calls to the `summarize`\n\t\tmethod to update the distribution parameters. Hyperparameters for the\n\t\tupdate are passed in at initialization time.\n\n\t\tNote: Internally, a call to `fit` is just a successive call to the\n\t\t`summarize` method followed by the `from_summaries` method.\n\t\t'
        self.distribution.from_summaries()
        if self.frozen == True:
            return
        priors = self._w_sum[:, 0] / torch.sum(self._w_sum, dim=-1)
        _update_parameter(self.priors, priors, self.inertia)
        self._reset_cache()