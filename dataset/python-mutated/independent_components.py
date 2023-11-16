import torch
from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _reshape_weights
from ._distribution import Distribution

class IndependentComponents(Distribution):
    """An independent components distribution object.

	A distribution made up of independent, univariate, distributions that each
	model a single feature in the data. This means that instead of using a
	single type of distribution to model all of the features in your data, you
	use one distribution per feature. Note that this will likely be slower
	than using a single distribution because the amount of batching possible
	will go down significantly.

	There are two ways to initialize this object. The first is to pass in a
	set of distributions that are all initialized with parameters, at which
	point this distribution can be immediately used for inference. The second
	is to pass in a set of distributions that are not initialized with
	parameters, and then call either `fit` or `summary` + `from_summaries` to
	learn the parameters of all the distributions.


	Parameters
	----------
	distributions: list, tuple, numpy.ndarray, torch.Tensor, shape=(d,)
		An ordered iterable containing all of the distributions, one per
		feature, that will be used.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

    def __init__(self, distributions, check_data=False):
        if False:
            i = 10
            return i + 15
        super().__init__(inertia=0.0, frozen=False, check_data=check_data)
        self.name = 'IndependentComponents'
        if len(distributions) <= 1:
            raise ValueError('Must pass in at least 2 distributions.')
        for distribution in distributions:
            if not isinstance(distribution, Distribution):
                raise ValueError('All passed in distributions must ' + 'inherit from the Distribution object.')
        self.distributions = distributions
        self._initialized = all((d._initialized for d in distributions))
        self.d = len(distributions)
        self._reset_cache()

    def _initialize(self, d):
        if False:
            print('Hello World!')
        'Initialize the probability distribution.\n\n\t\tThis method is meant to only be called internally. It initializes the\n\t\tparameters of the distribution and stores its dimensionality. For more\n\t\tcomplex methods, this function will do more.\n\n\n\t\tParameters\n\t\t----------\n\t\td: int\n\t\t\tThe dimensionality the distribution is being initialized to.\n\t\t'
        for distribution in self.distributions:
            distribution._initialize(d)
        self._initialized = True

    def _reset_cache(self):
        if False:
            i = 10
            return i + 15
        'Reset the internally stored statistics.\n\n\t\tThis method is meant to only be called internally. It resets the\n\t\tstored statistics used to update the model parameters as well as\n\t\trecalculates the cached values meant to speed up log probability\n\t\tcalculations.\n\t\t'
        if self._initialized == False:
            return
        for distribution in self.distributions:
            distribution._reset_cache()

    def sample(self, n):
        if False:
            print('Hello World!')
        'Sample from the probability distribution.\n\n\t\tThis method will return `n` samples generated from the underlying\n\t\tprobability distribution.\n\n\n\t\tParameters\n\t\t----------\n\t\tn: int\n\t\t\tThe number of samples to generate.\n\t\t\n\n\t\tReturns\n\t\t-------\n\t\tX: torch.tensor, shape=(n, self.d)\n\t\t\tRandomly generated samples.\n\t\t'
        return torch.hstack([d.sample(n) for d in self.distributions])

    def log_probability(self, X):
        if False:
            i = 10
            return i + 15
        'Calculate the log probability of each example.\n\n\t\tThis method calculates the log probability of each example given the\n\t\tparameters of the distribution. The examples must be given in a 2D\n\t\tformat.\n\n\t\tNote: This differs from some other log probability calculation\n\t\tfunctions, like those in torch.distributions, because it is not\n\t\treturning the log probability of each feature independently, but rather\n\t\tthe total log probability of the entire example.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to evaluate.\n\n\n\t\tReturns\n\t\t-------\n\t\tlogp: torch.Tensor, shape=(-1,)\n\t\t\tThe log probability of each example.\n\t\t'
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, shape=(-1, self.d))
        logp = torch.zeros(X.shape[0])
        for (i, d) in enumerate(self.distributions):
            if isinstance(X, torch.masked.MaskedTensor):
                logp.add_(d.log_probability(X[:, i:i + 1])._masked_data)
            else:
                logp.add_(d.log_probability(X[:, i:i + 1]))
        return logp

    def summarize(self, X, sample_weight=None):
        if False:
            while True:
                i = 10
        'Extract the sufficient statistics from a batch of data.\n\n\t\tThis method calculates the sufficient statistics from optionally\n\t\tweighted data and adds them to the stored cache. The examples must be\n\t\tgiven in a 2D format. Sample weights can either be provided as one\n\t\tvalue per example or as a 2D matrix of weights for each feature in\n\t\teach example.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to summarize.\n\n\t\tsample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional\n\t\t\tA set of weights for the examples. This can be either of shape\n\t\t\t(-1, self.d) or a vector of shape (-1,). Default is ones.\n\t\t'
        if self.frozen == True:
            return
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, shape=(-1, self.d))
        sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight, dtype=torch.float32), device=self.device)
        for (i, d) in enumerate(self.distributions):
            d.summarize(X[:, i:i + 1], sample_weight=sample_weight[:, i:i + 1])

    def from_summaries(self):
        if False:
            i = 10
            return i + 15
        'Update the model parameters given the extracted statistics.\n\n\t\tThis method uses calculated statistics from calls to the `summarize`\n\t\tmethod to update the distribution parameters. Hyperparameters for the\n\t\tupdate are passed in at initialization time.\n\n\t\tNote: Internally, a call to `fit` is just a successive call to the\n\t\t`summarize` method followed by the `from_summaries` method.\n\t\t'
        if self.frozen == True:
            return
        for distribution in self.distributions:
            distribution.from_summaries()