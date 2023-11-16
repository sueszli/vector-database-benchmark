import torch
from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _update_parameter
from .._utils import _check_parameter
from ._distribution import Distribution

class DiracDelta(Distribution):
    """A dirac delta distribution object.

	A dirac delta distribution is a probability distribution that has its entire
	density at zero. This distribution assumes that each feature is independent
	of the others. This means that, in practice, it will assign a zero
	probability if any value in an example is non-zero.

	There are two ways to initialize this object. The first is to pass in
	the tensor of alpha values representing the probability to return given a
	zero value, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the probability
	parameter will be learned from data.


	Parameters
	----------
	alphas: list, numpy.ndarray, torch.Tensor or None, shape=(d,), optional
		The probability parameters for each feature. Default is None.

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

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

    def __init__(self, alphas=None, inertia=0.0, frozen=False, check_data=True):
        if False:
            i = 10
            return i + 15
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'DiracDelta'
        self.alphas = _check_parameter(_cast_as_parameter(alphas), 'alphas', min_value=0.0, ndim=1)
        self._initialized = alphas is not None
        self.d = len(self.alphas) if self._initialized else None
        self._reset_cache()

    def _initialize(self, d):
        if False:
            i = 10
            return i + 15
        'Initialize the probability distribution.\n\n\t\tThis method is meant to only be called internally. It initializes the\n\t\tparameters of the distribution and stores its dimensionality. For more\n\t\tcomplex methods, this function will do more.\n\n\n\t\tParameters\n\t\t----------\n\t\td: int\n\t\t\tThe dimensionality the distribution is being initialized to.\n\t\t'
        self.alphas = _cast_as_parameter(torch.ones(d, device=self.device))
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        if False:
            for i in range(10):
                print('nop')
        'Reset the internally stored statistics.\n\n\t\tThis method is meant to only be called internally. It resets the\n\t\tstored statistics used to update the model parameters as well as\n\t\trecalculates the cached values meant to speed up log probability\n\t\tcalculations.\n\t\t'
        if self._initialized == False:
            return
        self.register_buffer('_log_alphas', torch.log(self.alphas))

    def log_probability(self, X):
        if False:
            while True:
                i = 10
        'Calculate the log probability of each example.\n\n\t\tThis method calculates the log probability of each example given the\n\t\tparameters of the distribution. The examples must be given in a 2D\n\t\tformat. \n\n\t\tNote: This differs from some other log probability calculation\n\t\tfunctions, like those in torch.distributions, because it is not\n\t\treturning the log probability of each feature independently, but rather\n\t\tthe total log probability of the entire example.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to evaluate.\n\n\n\t\tReturns\n\t\t-------\n\t\tlogp: torch.Tensor, shape=(-1,)\n\t\t\tThe log probability of each example.\n\t\t'
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        return torch.sum(torch.where(X == 0.0, self._log_alphas, float('-inf')), dim=-1)

    def summarize(self, X, sample_weight=None):
        if False:
            while True:
                i = 10
        'Extract the sufficient statistics from a batch of data.\n\n\t\tThis method calculates the sufficient statistics from optionally\n\t\tweighted data and adds them to the stored cache. The examples must be\n\t\tgiven in a 2D format. Sample weights can either be provided as one\n\t\tvalue per example or as a 2D matrix of weights for each feature in\n\t\teach example.\n\n\t\tFor a dirac delta distribution, there are no statistics to extract.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to summarize.\n\n\t\tsample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional\n\t\t\tA set of weights for the examples. This can be either of shape\n\t\t\t(-1, self.d) or a vector of shape (-1,). Default is ones.\n\t\t'
        if self.frozen == True:
            return
        (X, sample_weight) = super().summarize(X, sample_weight=sample_weight)

    def from_summaries(self):
        if False:
            return 10
        'Update the model parameters given the extracted statistics.\n\n\t\tThis method uses calculated statistics from calls to the `summarize`\n\t\tmethod to update the distribution parameters. Hyperparameters for the\n\t\tupdate are passed in at initialization time.\n\n\t\tFor a dirac delta distribution, there are no updates.\n\n\t\tNote: Internally, a call to `fit` is just a successive call to the\n\t\t`summarize` method followed by the `from_summaries` method.\n\t\t'
        return