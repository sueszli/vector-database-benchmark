import math
import torch
from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _update_parameter
from .._utils import _check_parameter
from .normal import Normal

class StudentT(Normal):
    """A Student T distribution.

	A Student T distribution models the probability of a variable occuring under
	a bell-shaped curve with heavy tails. Basically, this is a version of the
	normal distribution that is less resistant to outliers.  It is described by 
	a vector of mean values and a vector of variance values. This
	distribution can assume that features are independent of the others if
	the covariance type is 'diag' or 'sphere', but if the type is 'full' then
	the features are not independent.

	There are two ways to initialize this object. The first is to pass in
	the tensor of probablity parameters, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the probability
	parameter will be learned from data.


	Parameters
	----------
	means: list, numpy.ndarray, torch.Tensor or None, shape=(d,), optional
		The mean values of the distributions. Default is None.

	covs: list, numpy.ndarray, torch.Tensor, or None, optional
		The variances and covariances of the distribution. If covariance_type
		is 'full', the shape should be (self.d, self.d); if 'diag', the shape
		should be (self.d,); if 'sphere', it should be (1,). Note that this is
		the variances or covariances in all settings, and not the standard
		deviation, as may be more common for diagonal covariance matrices.
		Default is None.

	covariance_type: str, optional
		The type of covariance matrix. Must be one of 'full', 'diag', or
		'sphere'. Default is 'full'. 

	min_cov: float or None, optional
		The minimum variance or covariance.

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

    def __init__(self, dofs, means=None, covs=None, covariance_type='diag', min_cov=None, inertia=0.0, frozen=False, check_data=True):
        if False:
            for i in range(10):
                print('nop')
        self.name = 'StudentT'
        dofs = _check_parameter(_cast_as_tensor(dofs), 'dofs', min_value=1, ndim=0, dtypes=(torch.int32, torch.int64))
        self.dofs = dofs
        super().__init__(means=means, covs=covs, min_cov=min_cov, covariance_type=covariance_type, inertia=inertia, frozen=frozen, check_data=check_data)
        del self.dofs
        self.register_buffer('dofs', _cast_as_tensor(dofs))
        self.register_buffer('_lgamma_dofsp1', torch.lgamma((dofs + 1) / 2.0))
        self.register_buffer('_lgamma_dofs', torch.lgamma(dofs / 2.0))

    def _reset_cache(self):
        if False:
            return 10
        'Reset the internally stored statistics.\n\n\t\tThis method is meant to only be called internally. It resets the\n\t\tstored statistics used to update the model parameters as well as\n\t\trecalculates the cached values meant to speed up log probability\n\t\tcalculations.\n\t\t'
        super()._reset_cache()
        if self._initialized == False:
            return
        self.register_buffer('_log_sqrt_dofs_pi_cov', torch.log(torch.sqrt(self.dofs * math.pi * self.covs)))

    def sample(self, n):
        if False:
            print('Hello World!')
        'Sample from the probability distribution.\n\n\t\tThis method will return `n` samples generated from the underlying\n\t\tprobability distribution.\n\n\n\t\tParameters\n\t\t----------\n\t\tn: int\n\t\t\tThe number of samples to generate.\n\t\t\n\n\t\tReturns\n\t\t-------\n\t\tX: torch.tensor, shape=(n, self.d)\n\t\t\tRandomly generated samples.\n\t\t'
        return torch.distributions.StudentT(self.means, self.covs).sample([n])

    def log_probability(self, X):
        if False:
            i = 10
            return i + 15
        'Calculate the log probability of each example.\n\n\t\tThis method calculates the log probability of each example given the\n\t\tparameters of the distribution. The examples must be given in a 2D\n\t\tformat.\n\n\t\tNote: This differs from some other log probability calculation\n\t\tfunctions, like those in torch.distributions, because it is not\n\t\treturning the log probability of each feature independently, but rather\n\t\tthe total log probability of the entire example.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to evaluate.\n\n\n\t\tReturns\n\t\t-------\n\t\tlogp: torch.Tensor, shape=(-1,)\n\t\t\tThe log probability of each example.\n\t\t'
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        t = (X - self.means) ** 2 / self.covs
        return torch.sum(self._lgamma_dofsp1 - self._lgamma_dofs - self._log_sqrt_dofs_pi_cov - (self.dofs + 1) / 2.0 * torch.log(1 + t / self.dofs), dim=-1)